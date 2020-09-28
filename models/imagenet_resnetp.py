# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:09:04 2020

@author: ASUS
"""

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import partial



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
   
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, cfg)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.conv2 = conv3x3(cfg, cfg, stride)
        self.bn2 = nn.BatchNorm2d(cfg)
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

        out = self.conv3(out)
        out = self.bn3(out)

        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetp(nn.Module):

    def __init__(self, block, layers, cfg=None, num_classes=1000):
        self.inplanes = 64
        super(ResNetp, self).__init__()
        if cfg == None:
            #cfg = [[64] * layers[0], [128]*layers[1], [256]*layers[2], [512]*layers[3]]
            #cfg = [item for sub_list in cfg for item in sub_list]            
            #cfg = [30, 32, 43, 80, 102, 110, 189, 229]     #1.1
            #cfg = [57, 57, 109, 97, 185, 159, 381, 326]     #0.9
            #cfg = [49, 44, 39, 89, 90, 79, 85, 161, 139, 155, 153, 256, 256, 290, 364, 267]    #0.99
            cfg = [30, 43, 26, 38, 59, 20, 70, 96, 121, 95, 80, 100, 128, 182, 141, 182]     #1.1

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

def resnetp18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetp(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnetp34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetp(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnetp50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetp(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnetp101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetp(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnetp152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetp(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model