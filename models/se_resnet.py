# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:20:48 2019

@author: ASUS
"""

from __future__ import absolute_import
import math

import torch
import torch.nn as nn
from functools import partial
from torch.autograd import Variable
from models.cifar3.se_module import SELayer

__all__ = ['se_resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class CifarSEBasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=16):
        # cfg should be a number in this case
        super(CifarSEBasicBlock, self).__init__()
        
        
        
        self.conv1 = conv3x3(planes, cfg, stride)
        self.se = SELayer(cfg, reduction)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.se(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CifarSEBasicBlock1(nn.Module):
    expansion = 1 

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=16):
        # cfg should be a number in this case
        super(CifarSEBasicBlock1, self).__init__()
        
        
        
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.se = SELayer(cfg, reduction)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg,planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.se(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CifarSEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=16):
        super(CifarSEBottleneck, self).__init__()                       
        self.conv1 = conv1x1(inplanes, cfg)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.conv2 = conv3x3(cfg, cfg, stride)
        self.bn2 = nn.BatchNorm2d(cfg)
        self.conv3 = conv1x1(cfg, cfg*4)
        self.bn3 = nn.BatchNorm2d(cfg*4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(cfg*4, reduction)
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
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CifarSEBottleneck1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=16):
        super(CifarSEBottleneck1, self).__init__()                       
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, cfg, stride)
        self.bn2 = nn.BatchNorm2d(cfg)
        self.conv3 = conv1x1(cfg, cfg*4)
        self.bn3 = nn.BatchNorm2d(cfg*4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(cfg*4, reduction)
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
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class CifarSEResNet(nn.Module):

    def __init__(self, depth, dataset='svhn', cfg=None, reduction=16):
        super(CifarSEResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 9n+2'
        n = (depth - 2) // 6

        block = CifarSEBottleneck if depth >=165 else CifarSEBasicBlock
        
        if cfg == None:
            cfg = [[16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]      
       
        
        self.cfg = cfg

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer1(block, 16, n, cfg=cfg[0:n], stride=1, reduction=reduction)
        self.layer2 = self._make_layer2(block, 32, n, cfg=cfg[n:2*n], stride=2, reduction=reduction)
        self.layer3 = self._make_layer3(block, 64, n, cfg=cfg[2*n:3*n], stride=2, reduction=reduction)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'svhn':
            num_classes = 10
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer1(self, block, planes, blocks, cfg, stride=1, reduction=16):
        downsample = None
        if stride != 1:
            downsample = partial(downsample_basic_block, planes=16)
        layers = []
        layers.append(CifarSEBasicBlock1(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, cfg, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=32)
        layers = []
        #self.inplanes = 8        
        layers.append(CifarSEBasicBlock1(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))
        return nn.Sequential(*layers)

    def _make_layer3(self, block, planes, blocks, cfg, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=64)
        layers = []
        #self.inplanes = 16      
        layers.append(CifarSEBasicBlock1(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        
        x = self.layer2(x)  # 16x16
        
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def se_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return CifarSEResNet(**kwargs)

if __name__ == '__main__':
    net = se_resnet(depth=50)
    x=Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)
