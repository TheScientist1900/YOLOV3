from torchvision.models.mobilenet import *
import torch.nn as nn

class MobileNetV2(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV2, self).__init__()
        if pretrained: 
            print('=> load weights from https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        self.model = mobilenet_v2(pretrained=pretrained)
        self.layers_out_filters = [32, 96, 320]
    def forward(self, x):
        out3 = self.model.features[:7](x)   # torch.Size([1, 32, 52, 52])
        out4 = self.model.features[7:14](out3)  # torch.Size([1, 96, 26, 26])
        out5 = self.model.features[14:18](out4) # torch.Size([1, 320, 13, 13])
        return out3, out4, out5
