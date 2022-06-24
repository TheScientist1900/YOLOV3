import torch
import torch.nn as nn
from collections import OrderedDict

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = nn.LeakyReLU(0.1)
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, layers, ):
        super(Darknet53, self).__init__()

        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1], )
        self.layer3 = self._make_layer([128, 256], layers[2], )
        self.layer4 = self._make_layer([256, 512], layers[3], )
        self.layer5 = self._make_layer([512, 1024], layers[4], )

        self.layers_out_filters = [64, 128, 256, 512, 1024]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks,):

        layers = []
        layers.append(('ds_conv', nn.Conv2d(
            self.inplanes, planes[1], 3, 2, 1, bias=False)))
        layers.append(('ds_bn', nn.BatchNorm2d(planes[1])))
        layers.append(('ds_relu', nn.LeakyReLU(0.1)))

        self.inplanes = planes[1]
        for i in range(blocks):
            layers.append(('residual_{}'.format(
                i), BasicBlock(self.inplanes, planes)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """
            b, 3, 416, 416
        """
        x = self.conv1(x)   # b, 32, 416, 416
        x = self.bn1(x)
        x = self.relu(x)

        out1 = self.layer1(x) # b, 64, 208, 208
        out2 = self.layer2(out1)    # b, 128, 104, 104
        out3 = self.layer3(out2)    # b, 256, 52, 52
        out4 = self.layer4(out3)    # b, 512, 26, 26
        out5 = self.layer5(out4)    # b, 1024, 13, 13

        return out3, out4, out5


def darknet53(pretrained=False):
    model = Darknet53([1, 2, 8, 8, 4])
    if pretrained:
        state_dict = torch.load('model_data/darknet53_backbone_weights.pth')
        model.load_state_dict(state_dict)
    return model
