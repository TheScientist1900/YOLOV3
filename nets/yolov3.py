from collections import OrderedDict
from utils.config import _C as cfg
from nets.mobilenet import MobileNetV2
import torch
import torch.nn as nn

from nets.darknet53 import darknet53


def conv2d(inplanes, planes, kernel_size, padding):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(inplanes, planes, kernel_size, padding=padding, bias=False)),
        ('bn', nn.BatchNorm2d(planes)),
        ('relu', nn.LeakyReLU(0.1)),
    ]))


def get_conv_set(filter_list, inplanes, planes):
    conv_set = nn.Sequential(
        conv2d(inplanes, filter_list[0], kernel_size=1, padding=0),
        conv2d(filter_list[0], filter_list[1], kernel_size=3, padding=1),
        conv2d(filter_list[1], filter_list[0], kernel_size=1, padding=0),
        conv2d(filter_list[0], filter_list[1], kernel_size=3, padding=1),
        conv2d(filter_list[1], filter_list[0], kernel_size=1, padding=0),
        conv2d(filter_list[0], filter_list[1], 3, padding=1),
        
        nn.Conv2d(filter_list[1], planes, kernel_size=1, bias=True)
    )
    return conv_set


class YOLOV3(nn.Module):
    def __init__(self, cfg):
        super(YOLOV3, self).__init__()
        if cfg.arch.__contains__('darknet53'):
            self.backbone = darknet53(pretrained=cfg.pretrained)

        elif cfg.arch.__contains__('mobilenet'):
            self.backbone = MobileNetV2(pretrained=cfg.pretrained)
            
        
        layers_out_filters = self.backbone.layers_out_filters
        self.last_layer0 = get_conv_set(
            [512, 1024], layers_out_filters[-1], cfg.model.num_box * (cfg.data.num_classes + 5))

        self.last_layer1_conv = conv2d(512, 256, 1, padding=0)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = get_conv_set(
            [256, 512], layers_out_filters[-2] + 256, cfg.model.num_box * (cfg.data.num_classes + 5))

        self.last_layer2_conv = conv2d(256, 128, 1, padding=0)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = get_conv_set(
            [128, 256], layers_out_filters[-3] + 128, cfg.model.num_box * (cfg.data.num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2

if __name__ == '__main__':
    input = torch.randn((1, 3, 418, 418))
    yolov3 = YOLOV3(cfg)
    y0, y1, y2 = yolov3(input)

    print('s')


