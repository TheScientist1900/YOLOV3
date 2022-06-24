import imp
import math
from loss.yolov3_loss import YOLOV3Loss
from nets.yolov3 import YOLOV3
from utils.config import _C as cfg
import torch
from utils.decode import Decoder
from utils.utils import get_anchors
from torch.utils.data.dataloader import DataLoader
from utils.dataloader import YoloDataset, yolo_dataset_collate
import os
import numpy as np
if __name__ == '__main__':

    x = torch.rand((1, 3, 416, 416))
    model = YOLOV3(cfg)
    y = model(x)    