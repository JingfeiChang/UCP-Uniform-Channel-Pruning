# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:50:32 2019

@author: ASUS
"""

from __future__ import absolute_import
import math

import torch.nn as nn



__all__ = ['preresnet']

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class Bottleneck1(nn.Module):
    expansion = 8

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck1, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 8, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class Bottleneck2(nn.Module):
    expansion = 16

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck2, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 16, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class Preresnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar100', cfg=None):
        super(Preresnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        #block1 = Bottleneck
        block2 = Bottleneck1
        block3 = Bottleneck2
        if cfg is None:
            # Construct config variable.
            # cfg = [[16, 8, 8], [64, 8, 8]*(n-1), [64, 16, 16], [128, 16, 16]*(n-1), [128, 16, 16], [256, 16, 16]*(n-1), [256]]
            #cfg = [[16, 8, 8], [64, 7, 7], [64, 9, 9], [64, 10, 10], [64, 6, 6], [64, 7, 7], [64, 11, 11], [64, 11, 11], [64, 8, 8], 
            #       [64, 8, 8], [64, 10, 10], [64, 8, 8], [64, 8, 8], [64, 6, 6], [64, 11, 11], [64, 11, 11], [64, 9, 9], [64, 9, 9],
            #       [64, 17, 17],  [128, 17, 17], [128, 19, 19], [128, 14, 14], [128, 14, 14], [128, 21, 21], [128, 16, 16], [128, 18, 18], [128, 17, 17],
            #       [128, 20, 20], [128, 19, 19], [128, 18, 18], [128, 19, 19], [128, 21, 21], [128, 17, 17], [128, 19, 19], [128, 17, 17], [128, 14, 14],
            #       [128, 35, 35], [256, 35, 35], [256, 32, 32], [256, 35, 35], [256, 31, 31], [256, 49, 49], [256, 45, 45], [256, 28, 28], [256, 30, 30], 
            #       [256, 26, 26], [256, 39, 39], [256, 19, 19], [256, 40, 40], [256, 27, 27], [256, 40, 40], [256, 23, 23], [256, 58, 58], [256, 41, 41], [256]
            #       ]   #0.999999
            #cfg = [[16, 8, 8], [64, 7, 7], [64, 9, 9], [64, 10, 10], [64, 6, 6], [64, 7, 7], [64, 11, 11], [64, 11, 11], [64, 8, 8], 
            #        [64, 8, 8], [64, 10, 10], [64, 8, 8], [64, 8, 8], [64, 6, 6], [64, 11, 11], [64, 11, 11], [64, 9, 9], [64, 9, 9],
            #        [64, 17, 17],  [128, 17, 17], [128, 19, 19], [128, 14, 14], [128, 14, 14], [128, 21, 21], [128, 16, 16], [128, 18, 18], [128, 17, 17],
            #        [128, 20, 20], [128, 19, 19], [128, 18, 18], [128, 19, 19], [128, 21, 21], [128, 17, 17], [128, 19, 19], [128, 17, 17], [128, 14, 14],
            #        [128, 35, 35], [256, 35, 35], [256, 32, 32], [256, 35, 35], [256, 31, 31], [256, 15, 15], [256, 44, 44], [256, 28, 28], [256, 30, 30], 
            #        [256, 26, 26], [256, 39, 39], [256, 19, 19], [256, 40, 40], [256, 27, 27], [256, 40, 40], [256, 23, 23], [256, 2, 2], [256, 8, 8], [256]
            #       ]    # 1.000001      
            #cfg = [[16, 8, 8], [64, 7, 7], [64, 9, 9], [64, 10, 10], [64, 6, 6], [64, 7, 7], [64, 11, 11], [64, 11, 11], [64, 8, 8], 
            #        [64, 7, 7], [64, 6, 6], [64, 7, 7], [64, 8, 8], [64, 6, 6], [64, 11, 11], [64, 6, 6], [64, 9, 9], [64, 9, 9],
            #        [64, 17, 17],  [128, 17, 17], [128, 19, 19], [128, 14, 14], [128, 14, 14], [128, 21, 21], [128, 16, 16], [128, 18, 18], [128, 17, 17],
            #        [128, 20, 20], [128, 19, 19], [128, 18, 18], [128, 19, 19], [128, 21, 21], [128, 17, 17], [128, 19, 19], [128, 17, 17], [128, 14, 14],
            #        [128, 35, 35], [256, 35, 35], [256, 32, 32], [256, 35, 35], [256, 31, 31], [256, 15, 15], [256, 35, 35], [256, 24, 24], [256, 30, 30], 
            #        [256, 23, 23], [256, 13, 13], [256, 5, 5], [256, 14, 14], [256, 27, 27], [256, 39, 39], [256, 20, 20], [256, 2, 2], [256, 8, 8], [256]
            #       ]    # 1.0001  
            
            cfg = [[16, 8, 8], [64, 4, 4], [64, 9, 9], [64, 9, 9], [64, 6, 6], [64, 4, 4], [64, 11, 11], [64, 11, 11], [64, 4, 4], 
                    [64, 4, 4], [64, 3, 3], [64, 4, 4], [64, 8, 8], [64, 3, 3], [64, 10, 10], [64, 6, 6], [64, 8, 8], [64, 9, 9],
                    [64, 17, 17],  [128, 17, 17], [128, 18, 18], [128, 14, 14], [128, 14, 14], [128, 21, 21], [128, 15, 15], [128, 18, 18], [128, 17, 17],
                    [128, 19, 19], [128, 18, 18], [128, 15, 15], [128, 19, 19], [128, 20, 20], [128, 16, 16], [128, 19, 19], [128, 16, 16], [128, 13, 13],
                    [128, 35, 35], [256, 34, 34], [256, 32, 32], [256, 35, 35], [256, 31, 31], [256, 8, 8], [256, 18, 18], [256, 12, 12], [256, 8, 8], 
                    [256, 12, 12], [256, 7, 7], [256, 5, 5], [256, 14, 14], [256, 27, 27], [256, 36, 36], [256, 10, 10], [256, 6, 6], [256, 8, 8], [256]
                   ]    # 1.01             
             
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block2, 8, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block2, 16, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block3, 16, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * 4)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)
        elif dataset == 'svhn':
            self.fc = nn.Linear(cfg[-1], 10)        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def preresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return Preresnet(**kwargs)