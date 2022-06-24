
import numpy as np
import os
from xml.etree import ElementTree as ET

from tqdm import tqdm
import matplotlib.pyplot as plt
# 1. 导入数据：读取xml文件，得到obj的width和height（归一化）
# 2. k-means算法
### a. 随机选取k个聚类中心：从box里随机取k个box的width、height作为初始坐标 
### b. 开启循环：计算每一个box到k个中心的distance，得到每一个box距离最近的center的idx
### c. 如果上一次每个box距离最近的center的idx和此次的一模一样，则算法结束
### d. 否则更新聚类中心的坐标和上一次每个box距离最近的center的idx

def load_data(xml_dir, img_set_txt):

    data = []
    xml_files = os.listdir(xml_dir)
    img_set = open(img_set_txt, 'r', encoding='utf-8').read().strip().split()
    #-------------------------------------------------------------#
    #   对于每一个xml都寻找box
    #-------------------------------------------------------------#
    for xml_file in tqdm(xml_files):
        if xml_file[:-4] in img_set:
            
            root = ET.parse(os.path.join(xml_dir, xml_file)).getroot()
            size = root.find('size')
            height = int(size.find('height').text)
            width = int(size.find('width').text)

            for obj in root.iter('object'):
                bndbox = obj.find('bndbox')
                xmin = np.float32(int(float(bndbox.find('xmin').text)) /  width)
                ymin = np.float32(int(float(bndbox.find('ymin').text)) / height)
                xmax = np.float32(int(float(bndbox.find('xmax').text)) / width)
                ymax = np.float32(int(float(bndbox.find('ymax').text)) / height)

                data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


def k_means(box, num_anchors):

    row = box.shape[0]
    # 聚类中心的坐标位置, 0-1
    cluster = box[np.random.choice(row, num_anchors)] # num_anchors, 
    # 最后每个box属于的类别的id, 0-num_anchors-1
    cluster_idx = np.zeros((row, ))
    
    iter = 0
    while True:
        distance = 1 - calc_iou(box, cluster)   # row, num_anchors

        near = np.argmin(distance, axis=1)  # row, box距离最近的center的id

        if (cluster_idx == near).all():
            break
        
        for j in range(num_anchors):
            cluster[j] = np.median(box[near == j], axis=0)  # 中位数
        cluster_idx = near
        
        if iter % 5 == 0:
            print("iter: {:d}  avg_iou:{:.2f}".format(iter, avg_iou(box, cluster)))
        iter += 1
    return cluster, cluster_idx

def calc_iou(box1, box2):
    """
        box1: a, 2
        box2: b, 2
    """
    x = np.minimum(box1[:, None, 0], box2[:, 0]) # a, b
    y = np.minimum(box1[:, None, 1], box2[:, 1]) # a, b    

    area_i = y * x
    area_1 = np.prod(box1, axis=1) 
    area_2 = np.prod(box2, axis=1)
    return area_i / (area_1[:, None] + area_2 - area_i) 

def avg_iou(box, cluster):
    ious = calc_iou(box, cluster) # row, k

    return np.mean(np.max(ious, axis=1))

if __name__ == '__main__':

    xml_dir = r'D:\keypointsProject\faster-rcnn-pytorch-master\VOCdevkit\VOC2028/Annotations'
    img_set_txt = r'D:\keypointsProject\faster-rcnn-pytorch-master\VOCdevkit\VOC2028/ImageSets\Main\train.txt'
    
    num_anchors = 9
    input_shape = [416, 416]

    print('loading data...')
    box = load_data(xml_dir, img_set_txt)
    print('loaded data size: {}'.format(box.shape[0]))

    print('K-means boxes.')
    cluster, cluster_idx   = k_means(box, num_anchors)
    print('K-means boxes done.')

    cluster = cluster * np.array([input_shape[1], input_shape[0]])
    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print(cluster)
    
    f = open('model_data/yolo_anchors.txt', 'w', encoding='utf-8')
    for i in range(num_anchors):
        if i == 0:
            f.write("{},{}".format(int(cluster[i, 0]), int(cluster[i, 1])))
        else:
            f.write(", {},{}".format(int(cluster[i, 0]), int(cluster[i, 1])))
    f.close()

    # 绘图
    box = box * np.array([input_shape[1], input_shape[0]])
    for j in range(num_anchors):
        plt.scatter(box[cluster_idx == j][:,0], box[cluster_idx == j][:,1])
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')