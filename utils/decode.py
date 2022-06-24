import os
import torch
import torch.nn as nn
from torchvision.ops import nms
import colorsys
import torch
from utils.config import _C as cfg
from nets.yolov3 import YOLOV3
import numpy as np
from utils.utils import cvtColor, preprocess_input, resize_image
from PIL import ImageDraw, ImageFont

class Decoder():
    def __init__(self, cfg, mode='predice'):
        # self.__dict__.update(cfg)
        self.anchors = np.array(cfg.model.anchors).reshape(-1, 2)
        self.num_box = cfg.model.num_box
        self.num_classes = cfg.data.num_classes
        self.anchor_mask = cfg.model.anchor_mask
        self.input_shape = cfg.data.input_shape
        self.cuda = cfg.predict.cuda if mode == 'predict' else cfg.map.cuda
        self.conf_thresh = cfg.predict.conf_thresh if mode == 'predict' else cfg.map.conf_thresh
        self.nms_thresh = cfg.predict.nms_thresh if mode == 'predict' else cfg.map.nms_thresh

        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

    def decode(self, preds, image_shape):
        boxes = self.get_box(preds)  # 1, 10647, 25
        results = self.non_maximum_supression(boxes, image_shape)
        return results

    def get_box(self, preds):
        outputs = []
        for i in range(len(preds)):
            pred = preds[i]  # b, 75, 13, 13
            batch = pred.size()[0]
            output_height, output_width = pred.size()[-2:]
 
            # b, 3, 13, 13, 25
            pred = pred.view(-1, self.num_box, self.num_classes+5,
                             output_height, output_width).permute(0, 1, 3, 4, 2).contiguous()

            stride_x = self.input_shape[1] // output_width
            stride_y = self.input_shape[0] // output_height
            # 3, 2
            scaled_anchors = torch.Tensor(
                [[self.anchors[self.anchor_mask[i]][j][0] /
                  stride_x, self.anchors[self.anchor_mask[i]][j][1] / stride_y] for j in range(self.num_box)]
            ).type(self.FloatTensor)
            
            # repeat(a, b...)沿着指定的维度复制
            pw = scaled_anchors[:, 0].unsqueeze(1).repeat(
                batch, 1).repeat(1, 1, output_height*output_width).view(
                    batch, self.num_box, output_height, output_width
            )
            ph = scaled_anchors[:, 1].unsqueeze(1).repeat(
                batch, 1).repeat(1, 1,output_height*output_width).view(
                    batch, self.num_box, output_height, output_width
            )
            cx = torch.linspace(0, output_width-1, output_width).repeat(
                output_height, 1).repeat(self.num_box, 1, 1,).repeat(batch, 1, 1, 1)
            cy = torch.linspace(0, output_height-1, output_height).repeat(
                output_width, 1).t().repeat(self.num_box, 1, 1,).repeat(batch, 1, 1, 1)
            if self.cuda:
                cx = cx.cuda()
                cy = cy.cuda()

            bx = torch.sigmoid(pred[..., 0]) + cx   # b, 3, 13, 13
            by = torch.sigmoid(pred[..., 1]) + cy
            bw = torch.exp(pred[..., 2]) * pw
            bh = torch.exp(pred[..., 3]) * ph

            conf = torch.sigmoid(pred[..., 4])
            cls_prob = torch.sigmoid(pred[..., 5:])  # b, 3, 13, 13, 20

            output = torch.cat((
                bx.view(batch, -1, 1) / output_width,
                by.view(batch, -1, 1) / output_height,
                bw.view(batch, -1, 1) / output_width,
                bh.view(batch, -1, 1) / output_height,
                conf.view(batch, -1, 1),
                cls_prob.view(batch, -1, self.num_classes)
            ), dim=-1)  # b, 507, 25
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def non_maximum_supression(self, boxes, image_shape):
        # boxes: b, k, 25

        # 转换为xyxy
        batch = boxes.size()[0]
        num_box = boxes.size()[1]

        box_xyxy = torch.zeros((batch, num_box, 4))
        if self.cuda:
            box_xyxy = box_xyxy.cuda()
        box_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
        box_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
        box_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
        box_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2

        results = []
        cls_probs, cls_ids = torch.max(boxes[:, :, 5:], dim=-1, keepdim=True)   # b, k, 1
        for i in range(batch):
            results.append([])

            decoded_box = torch.cat((
                box_xyxy[i, :, :4], 
                boxes[i, :, 4].unsqueeze(-1), 
                cls_probs[i], 
                cls_ids[i].type(self.FloatTensor)
                ), dim=-1
            )   # k, 7
            conf_keep = (decoded_box[:, 4] * decoded_box[:, 5]) >= self.conf_thresh
            
            decoded_box_conf = decoded_box[conf_keep]
            for c in range(self.num_classes):
                decoded_box_c_conf = decoded_box_conf[decoded_box_conf[:, 6] == c]
                nms_keep = nms(
                    decoded_box_c_conf[:, :4],
                    decoded_box_c_conf[:, 4] * decoded_box_c_conf[:, 5],
                    self.nms_thresh
                )
                decoded_box_c_nms = decoded_box_c_conf[nms_keep]    # m, 7
                if self.cuda:
                    results[-1].extend(decoded_box_c_nms.cpu().numpy())
                else:
                    results[-1].extend(decoded_box_c_nms.numpy())
            if len(results[-1]) != 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / \
                    2, (results[-1][:, 2:4] - results[-1][:, 0:2])
                results[-1][:, :4] = self.yolo_correct_box(box_xy, box_wh, image_shape)

        return results

    def yolo_correct_box(self, box_xy, box_wh, image_shape):

        box_yx = box_xy[:, ::-1]
        box_hw = box_wh[:, ::-1]
        box_yx, box_hw = np.array(box_yx), np.array(box_hw)

        ul = box_yx - (box_hw / 2.)
        br = box_yx + (box_hw / 2.)

        box_yxyx = np.concatenate((ul, br), axis=1)
        box_yxyx *= np.concatenate((image_shape, image_shape),)
        return box_yxyx

class ModelUtil():
    def __init__(self, cfg, mode='predict'):
        super(ModelUtil, self).__init__()
        self.model = YOLOV3(cfg)
        self.cuda = cfg.predict.cuda if mode == 'predict' else cfg.map.cuda
        self.checkpoint = cfg.predict.checkpoint if mode == 'predict' else cfg.map.checkpoint
        self.load_weights(self.checkpoint)
        
        self.num_classes = cfg.data.num_classes
        self.class_names = cfg.data.class_names
        self.input_shape = cfg.data.input_shape
        self.decoder = Decoder(cfg, mode)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    
    def load_weights(self, checkpoint):
        print("=> loading weights from {}".format(checkpoint))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if np.shape(v) == np.shape(model_dict[k])}
        # pretrained_dict = {}
        # for k, v in pretrained_dict_.items():
        #     if np.shape(v) == np.shape(model_dict[k]):
        #         pretrained_dict[k] = v
        #     else:
        #         print(k)
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.model.eval()
        if self.cuda:
            self.model = self.model.cuda()
        print("=> loaded weights from {}".format(checkpoint))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), False)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            preds = self.model(images)
            results = self.decoder.decode(preds, image_shape)

            if len(results[0]) == 0:
                return image
            
            box_coords = results[0][:, :4]
            conf = results[0][:, 4] * results[0][:, 5]
            cls_id = results[0][:, -1]


        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        for i, c in list(enumerate(cls_id)):
            c = int(c)
            predicted_class = self.class_names[c]
            box             = box_coords[i]
            score           = conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
   
    def get_map_txt(self, image_id, image, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), False)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            preds = self.model(images)
            results = self.decoder.decode(preds, image_shape)
            
            if len(results[0]) == 0:
                return 

            box_coords = results[0][:, :4]
            conf = results[0][:, 4] * results[0][:, 5]
            cls_id = results[0][:, -1]

        for i, c in list(enumerate(cls_id)):
            c = int(c)
            predicted_class = self.class_names[c]
            box             = box_coords[i]
            score           = str(conf[i])

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

        
            if predicted_class not in self.class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 