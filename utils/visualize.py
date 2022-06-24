import colorsys
import os
import numpy as np
from tqdm import tqdm
from PIL import ImageFont, Image, ImageDraw
from utils import get_classes


def draw_annotation(img_path, annot_path, save_path, classes_path, batch=False):
    class_names, num_classes = get_classes(classes_path)
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    if batch:
        print('start draw ground truth in batch way')
        annot = open(annot_path, 'r', encoding='UTF-8').readlines()
        num = 0

        for line in tqdm(annot):
            line = line.split()

            image = Image.open(line[0],)
            if np.shape(image) != 3 or np.shape(image)[2] != 3:
                image = image.convert('RGB')
            img_name = os.path.basename(line[0])

            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = int(
                max((image.size[0] + image.size[1]) // np.mean(np.array(image.size[:2])), 1))

            for box in line[1:]:
                left, top, right, bottom, cls_id= box.split(',')

                top     = max(0, int(top))
                left    = max(0, int(left))
                bottom  = min(image.size[1], int(bottom))
                right   = min(image.size[0], int(right))

                label = '{}'.format(class_names[int(cls_id)])
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, left, top, right, bottom)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[int(cls_id)])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[int(cls_id)])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            if not os.path.exists(save_path):
                os.makedirs(os.path.join(save_path), exist_ok=True)
            image.save(os.path.join(save_path, img_name))
            num += 1
        print('draw {} ground truth in batch way done!'.format(num))
    else:
        img_name = os.path.basename(img_path)
        image = Image.open(img_path)
        annot = open(annot_path, 'r').readlines()

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(
            max((image.size[0] + image.size[1]) // np.mean(np.array(image.size[:2])), 1))

        for line in annot:
            line = line.split()
            if os.path.basename(line[0]) == img_name:
                for box in line[1:]:
                    left, top, right, bottom, cls_id= box.split(',')

                    top     = max(0, int(top))
                    left    = max(0, int(left))
                    bottom  = min(image.size[1], int(bottom))
                    right   = min(image.size[0], int(right))

                    label = '{}'.format(class_names[int(cls_id)])
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    label = label.encode('utf-8')
                    print(label, top, left, bottom, right)

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for i in range(thickness):
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[int(cls_id)])
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[int(cls_id)])
                    draw.text(text_origin, str(label,'UTF-8'), fill=(0,0,0), font=font)
                    del draw
                
                image.show()
if __name__ == '__main__':
    draw_annotation(
        r'D:\keypointsProject\implementation\object_dection\YOLOV3\yolov3\VOCdevkit\VOC2007\JPEGImages', 
        r'D:\keypointsProject\implementation\object_dection\YOLOV3\yolov3\VOCdevkit\VOC2007\train.txt',
        r'D:\keypointsProject\implementation\object_dection\YOLOV3\yolov3\VOCdevkit\VOC2007\GroundTruth\train',
        r'D:\keypointsProject\implementation\object_dection\YOLOV3\yolov3\VOCdevkit\VOC2007\voc_classes.txt',
        True
    )