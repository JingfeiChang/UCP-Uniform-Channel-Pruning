# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:24:47 2019

@author: ASUS
"""

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.cifar3.se_module import SELayer


__all__ = ['se_vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    #16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    #16 : [34, 33, 'M', 62, 68, 'M', 123, 131, 157, 'M', 282, 332, 479, 'M', 313, 512, 418],    #4 
    #16 : [34, 33, 'M', 62, 67, 'M', 123, 131, 143, 'M', 265, 291, 333, 'M', 274, 274, 322],   #8
    16 : [33, 33, 'M', 62, 67, 'M', 123, 131, 110, 'M', 243, 250, 14, 'M', 241, 256, 56],   #1.0001  
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class se_vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None, reduction=16):
        super(se_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]


        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'svhn':
            num_classes = 10
        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                se = SELayer(v, reduction=16)
                if batch_norm:
                    layers += [conv2d, se, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, se, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)


if __name__ == '__main__':
    net = se_vgg()
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)