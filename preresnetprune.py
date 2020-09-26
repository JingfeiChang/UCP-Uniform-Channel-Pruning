# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:06:47 2019

@author: ASUS
"""

import argparse
import numpy as np

import torch

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')

args = parser.parse_args()


imscore=[

]

skip = {
    #'A': [1, 4, 6, 14]
    'A':[]
}                                                    

layer_id = 0
im = 0
cfg = []

for m in range(0, 54):
    out_channels = len(imscore[im])
    if layer_id in skip[args.v]:
        cfg.append(out_channels)    
        layer_id += 1
        im += 1
    imscore_copy = imscore[im]
    thre = np.sum(imscore_copy)/out_channels
    thre = thre * 1.01
    imscore_copy = torch.Tensor(imscore[im])
    mask = imscore_copy.gt(thre).float().cuda()             
                     
    cfg.append(int(torch.sum(mask)))                  

    layer_id += 1
    im += 1
