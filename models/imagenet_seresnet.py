import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import partial
from models.se_module import SELayer


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)
    
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=16):
        # cfg should be a number in this case
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(cfg, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, cfg)
        self.bn2 = nn.BatchNorm2d(cfg)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.se(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
  
   
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, cfg)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.conv2 = conv3x3(cfg, cfg, stride)
        self.bn2 = nn.BatchNorm2d(cfg)
        self.se = SELayer(cfg, reduction)
        self.conv3 = conv1x1(cfg, cfg*4)
        self.bn3 = nn.BatchNorm2d(cfg*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNet(nn.Module):

    def __init__(self, block, layers, cfg=None, num_classes=1000, reduction=16):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        if cfg == None:
            cfg = [[64] * layers[0], [128]*layers[1], [256]*layers[2], [512]*layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        count = 0
        self.layer1 = self._make_layer(block, 64, layers[0], cfg[:layers[0]])
        count += layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg[count:count+layers[1]], stride=2)
        count += layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg[count:count+layers[2]], stride=2)
        count += layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg[count:count+layers[3]], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def seresnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def seresnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def seresnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def seresnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def seresnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBottleneck, [3, 8, 36, 3], **kwargs)
    return model