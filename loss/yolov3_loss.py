import math
from re import S
from turtle import forward
from matplotlib.pyplot import box
import torch
import torch.nn as nn
import numpy as np


class YOLOV3Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.anchors = np.array(cfg.model.anchors).reshape(-1, 2)
        self.num_box = cfg.model.num_box
        self.num_classes = cfg.data.num_classes
        self.anchors_mask = cfg.model.anchor_mask
        self.input_shape = cfg.data.input_shape
        self.cuda = cfg.cuda
        self.ignore_thresh = cfg.loss.ignore_thresh
        self.loc_ratio = cfg.loss.loc_ratio
        self.cls_ratio = cfg.loss.cls_ratio
        self.conf_ratio = cfg.loss.conf_ratio
        self.balance = cfg.loss.balance
        self.bbox_attrs = 5 + self.num_classes
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + \
            (result > t_max).float() * t_max
        return result

    def BCELoss(self, pred, target):
        """
            preds: b, i, j, k
            targets: b, i, j, k
        """
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - \
            (1.0 - target) * torch.log(1.0 - pred)
        return output

    def MSELoss(self, preds, targets):
        loss = torch.pow((preds - targets), 2)
        return loss

    def forward(self, i, pred, targets):
        """
            pred: b,3*25, 13, 13
            targets: list(torch.Tensor), k,
        """
        output_height, output_width = pred.size()[-2:]
        pred = pred.view(-1, self.num_box, self.num_classes+5,
                         output_height, output_width).permute(0, 1, 3, 4, 2).contiguous()

        stride_y, stride_x = self.input_shape[0] / \
            output_height, self.input_shape[1] / output_width
        scaled_anchors = np.array([[i[0] / stride_x, i[1] / stride_y]
                                   for i in self.anchors])  # 9, 2

        #   y_pred: b, 3, 13, 13, 25
        #   obj_mask: b, 3, 13, 13
        #   loss_scale: b, 3, 13, 13
        y_true, obj_mask, loss_scale = self.get_target(
            i, targets, scaled_anchors, output_height, output_width
        )

        # 先要获得正负样本，就先得对模型输出进行解码
        obj_mask, _ = self.assign_label(i, pred, targets, obj_mask,
                                          output_height, output_width, scaled_anchors)

        if self.cuda:
            y_true = y_true.cuda()
            obj_mask = obj_mask.cuda()
            loss_scale = loss_scale.cuda()

        loss_scale = 2 - loss_scale
        obj_mask = y_true[..., 4] == 1
        
        # obj_mask = obj_mask == 1
        pos_num = torch.sum(obj_mask)

        
        x = torch.sigmoid(pred[..., 0])  # torch.Size([b, 3, 13, 13])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        conf = torch.sigmoid(pred[..., 4])
        cls_prob = torch.sigmoid(pred[..., 5:])

        total_loss = 0
        losses = torch.Tensor([0, 0, 0, 0])
        if pos_num > 0:
            
            loss_x = torch.mean(
                (self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * loss_scale[obj_mask]))
            loss_y = torch.mean(
                (self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * loss_scale[obj_mask]))

            loss_w = torch.mean(
                (self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * loss_scale[obj_mask]))
            loss_h = torch.mean(
                (self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * loss_scale[obj_mask]))

            loc_loss = (loss_x + loss_y + loss_h + loss_w) * 0.1
            cls_loss = torch.mean(
                (self.BCELoss(cls_prob[obj_mask], y_true[..., 5:][obj_mask])))
            losses[2] = loc_loss
            losses[3] = cls_loss
            total_loss += loc_loss * self.loc_ratio + cls_loss * self.cls_ratio
        
        conf_loss = torch.mean((self.BCELoss(conf,
                                y_true[..., 4]))) * self.balance[i]
        total_loss += conf_loss * self.conf_ratio
        losses[0] = total_loss
        losses[1] = conf_loss
        # losses = [total_loss, conf_loss, loc_loss, cls_loss]
        return losses

    def get_target(self, i, targets, anchors, output_height, output_width):
        # anchors 是映射到特征层上的w和h
        batch = len(targets)
        y_true = torch.zeros((
            batch, self.num_box, output_height, output_width, 5+self.num_classes
        ), requires_grad=False)

        obj_mask = torch.zeros((
            batch, self.num_box, output_height, output_width),
            requires_grad=False)
        loss_scale = torch.zeros((
            batch, self.num_box, output_height, output_width),
            requires_grad=False)

        for b in range(batch):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])  # k, 5
            # 归一化转为实际值
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * output_width
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * output_height
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            gt_box_coord = torch.cat(
                (torch.zeros((batch_target.size()[0], 2)), batch_target[:, 2:4]), dim=1)  # k, 4

            gt_anchors = torch.cat(
                (torch.zeros((len(self.anchors), 2)), torch.FloatTensor(anchors)), dim=1)  # 9, 4

            box_anchor_maxiou_idx = torch.argmax(
                self.iou(gt_box_coord, gt_anchors), dim=-1)  # k,

            for t, anchor_id in enumerate(box_anchor_maxiou_idx):
                if anchor_id not in self.anchors_mask[i]:
                    continue
                c = batch_target[t, 4].long()
                anchor_pos_id = self.anchors_mask[i].index(
                    anchor_id)  # ∈0, 1, 2
                # .long() is equivalent to .to(torch.int64)
                gridx = torch.floor(batch_target[t, 0]).long()
                gridy = torch.floor(batch_target[t, 1]).long()

                y_true[b, anchor_pos_id, gridy, gridx,
                       0] = batch_target[t, 0] - gridx.float()
                y_true[b, anchor_pos_id, gridy, gridx,
                       1] = batch_target[t, 1] - gridy.float()
                y_true[b, anchor_pos_id, gridy, gridx, 2] = math.log(
                    batch_target[t, 2] / anchors[anchor_id][0]
                )
                y_true[b, anchor_pos_id, gridy, gridx, 3] = math.log(
                    batch_target[t, 3] / anchors[anchor_id][1]
                )
                y_true[b, anchor_pos_id, gridy, gridx, 5 + c] = 1
                y_true[b, anchor_pos_id, gridy, gridx, 4] = 1

                obj_mask[b, anchor_pos_id, gridy, gridx] = 1

                loss_scale[b, anchor_pos_id, gridy, gridx] = batch_target[t, 2] * batch_target[t, 3] / output_height / output_width

        return y_true, obj_mask, loss_scale

    def iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / \
            2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / \
            2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / \
            2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / \
            2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:,
                                                     3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:,
                                                     3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(
            A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(
            A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3] -
                  box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3] -
                  box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]
        # box1_xyxy = np.zeros_like(box1)
        # box2_xyxy = np.zeros_like(box2)
        # box1_xyxy[:, 0] = box1[:, 0] - box1[:, 2] / 2
        # box1_xyxy[:, 1] = box1[:, 1] - box1[:, 3] / 2
        # box1_xyxy[:, 2] = box1[:, 0] + box1[:, 2] / 2
        # box1_xyxy[:, 3] = box1[:, 1] + box1[:, 3] / 2

        # box2_xyxy[:, 0] = box2[:, 0] - box2[:, 2] / 2
        # box2_xyxy[:, 1] = box2[:, 1] - box2[:, 3] / 2
        # box2_xyxy[:, 2] = box2[:, 0] + box2[:, 2] / 2
        # box2_xyxy[:, 3] = box2[:, 1] + box2[:, 3] / 2

        # tl = np.maximum(box1_xyxy[:, None, :2], box2_xyxy[:, :2])
        # br = np.minimum(box1_xyxy[:, None, 2:], box2_xyxy[:, 2:])
        # area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        # area_a = np.prod(box1_xyxy[:, 2:] - box1_xyxy[:, :2], axis=1)
        # area_b = np.prod(box2_xyxy[:, 2:] - box2_xyxy[:, :2], axis=1)
        # return area_i / (area_a[:, None] + area_b - area_i)

    def assign_label(self, i, pred, targets, obj_mask, output_height, output_width, scaled_anchors):
        # pred: b,3,13,13,25
        # 先解码
        batch = pred.size()[0]
        cx = torch.linspace(0, output_width-1, output_width).repeat(
            output_height, 1).repeat(self.num_box, 1, 1).repeat(
                batch, 1, 1, 1).type(self.FloatTensor)

        cy = torch.linspace(0, output_height-1, output_height).repeat(
            output_width, 1).t().repeat(self.num_box, 1, 1).repeat(
                batch, 1, 1, 1).type(self.FloatTensor)

        pw = self.FloatTensor(scaled_anchors[self.anchors_mask[i]][:, 0]).unsqueeze(1).repeat(
            batch, 1).repeat(1, 1, output_height*output_width).view(
            batch, self.num_box, output_height, output_width
        )

        ph = self.FloatTensor(scaled_anchors[self.anchors_mask[i]][:, 1]).unsqueeze(1).repeat(
            batch, 1).repeat(1, 1, output_height*output_width).view(
            batch, self.num_box, output_height, output_width
        )

        pred_box = torch.zeros_like(pred).type(self.FloatTensor)
        pred_box[..., 0] = torch.sigmoid(pred[..., 0]) + cx
        pred_box[..., 1] = torch.sigmoid(pred[..., 1]) + cy
        pred_box[..., 2] = torch.exp(pred[..., 2]) * pw
        pred_box[..., 3] = torch.exp(pred[..., 3]) * ph

        pred_box[..., 4] = torch.sigmoid(pred[..., 4])
        pred_box[..., 5:] = torch.sigmoid(pred[..., 5:])

        # 再遍历
        for b in range(batch):
            if len(targets[b]) == 0:
                continue

            pred_box_for_ignore = pred_box[b].view((-1, pred_box.size()[-1]))

            # 映射到对应特征层上的xyxy
            batch_target = torch.zeros_like(targets[b])  # device会和targets[b]一样
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * output_width
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * output_height
            batch_target = batch_target[:, :4]

            # 3*13*13, k,
            iou = self.iou(pred_box_for_ignore[:, :4], batch_target, )
            iou_max, _ = torch.max(iou, dim=1)  # 3*13*13
            iou_max = iou_max.view(pred_box.size()[1:4])  # 3, 13, 13,
            obj_mask[b][iou_max > self.ignore_thresh] = 1
        return obj_mask, pred_box
