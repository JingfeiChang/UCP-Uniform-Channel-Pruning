# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 08:44:43 2019

@author: ASUS
"""

from torch import nn
import numpy as np

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if channel <= reduction:
            self.fc = nn.Sequential(
                nn.Linear(channel, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(1, channel, bias=True),
                nn.Sigmoid()            
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )        

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
                
        #z = y.clone().cpu().detach().numpy()
        #imscore = np.sum(z, axis=(0,2,3))/20000
        #imscore = imscore.tolist()
        #print(imscore)
        
        return x * y.expand_as(x)