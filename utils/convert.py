import os
import random

from tqdm import tqdm
import xml.etree.ElementTree as ET

op = 0
project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

data_year = '2007'
classes = open(os.path.join(project_root, 'VOCdevkit/VOC{}/voc_classes.txt'.format(data_year)),'r').read().strip().split()
data_root = os.path.join(project_root, 'VOCdevkit/VOC{}'.format(data_year))


def split_data(mode, data_type,): 
    random.seed(0)
    if mode == 0 or mode == 1:
        print("Generate txt in ImageSets.")

        save_dir = os.path.join(data_root, 'ImageSets/Main')
        trainval_percent = 0.9
        train_percent = 0.9


        xml_files = os.listdir(os.path.join(data_root, 'Annotations'))
        trainval_num = int(len(xml_files) * trainval_percent)
        train_num = int(trainval_num * train_percent)

        trainval_files = random.sample(xml_files, trainval_num)
        train_files = random.sample(trainval_files, train_num)

        print('train samples: {}'.format(len(train_files)))
        print('val samples: {}'.format(len(trainval_files) - len(train_files)))

        ftrainval = open(os.path.join(save_dir, 'trainval.txt'), 'w', encoding='utf-8')
        ftrain = open(os.path.join(save_dir, 'train.txt'), 'w', encoding='utf-8')
        fval = open(os.path.join(save_dir, 'val.txt'), 'w', encoding='utf-8')
        ftest = open(os.path.join(save_dir, 'test.txt'), 'w', encoding='utf-8')
        for i in tqdm(xml_files):
            if i in trainval_files:
                ftrainval.write(i[:-4]+'\n')
                if i in train_files:
                    ftrain.write(i[:-4]+'\n')
                else:
                    fval.write(i[:-4]+'\n')
            else:
                ftest.write(i[:-4]+'\n')
    if mode == 0 or mode == 2:
        img_sets = ['train', 'val',]
        if os.path.exists(os.path.join(data_root, 'ImageSets/Main/test.txt')):
            img_sets.append('test')
            
        for set in img_sets:
            img_ids = open(os.path.join(data_root,'ImageSets/Main/{}.txt').format(set), 'r', encoding='utf-8').read().strip().split()
            annot_txtfile = open(os.path.join(data_root, '{}.txt').format(set), 'w', encoding='utf-8')
            print('Generate {}.txt in {}'.format(set, data_root))
            for img_name in tqdm(img_ids):
                if op == 0:
                    annot_txtfile.write(os.path.join(data_root, 'JPEGImages/{}.jpg').format(img_name))
                elif op == 1:
                    annot_txtfile.write(os.path.join('/content/drive/MyDrive/VOC{}/JPEGImages/{}.jpg').format(data_year,img_name))
                
                xml_file_path = os.path.join(data_root, 'Annotations/{}.xml'.format(img_name))
                convert_annotation(data_type, xml_file_path, annot_txtfile)

def convert_annotation(data_type, xml_file_path, annot_txtfile):

    xml_file = open(xml_file_path, encoding='utf-8')
    tree = ET.parse(xml_file)
    root = tree.getroot()

    difficult = 0
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
            int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
         
        if data_type == 'yolov1':
            size = root.find('size')
            img_w, img_h = float(size.find('width').text), int(size.find('height').text)
            x = (b[0] + b[2]) / 2.0
            y = (b[1] + b[3]) / 2.0
            w = b[2] - b[0]
            h = b[3] - b[1]
            b = (float(x / img_w), float(y / img_h), float(w / img_w), float(h / img_h), )
        annot_txtfile.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    annot_txtfile.write('\n')

if __name__ == '__main__':
    # mode = 0
    # mode = 1
    # mode = 2
    split_data(mode=2, data_type='voc')

    