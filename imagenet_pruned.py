# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 22:42:56 2019

@author: ASUS
"""

import argparse
import os
import shutil
import pdb, time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ptflops import get_model_complexity_info
import models
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, default='F:\soft filter pruning\imagenet', help='Path to dataset')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='seresnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', type=str, default='./best.seresnet18.pth.zip', metavar='PATH', help='path to latest checkpoint (default: none)')
# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--eval_small', dest='eval_small', action='store_true', help='whether a big or small model')
parser.add_argument('--small_model', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

imscore=[

]

skip = {
    #'A': [1, 4, 6, 14]
    'A':[]
}                                                    

layer_id = 0
im = 0
cfg = []

for m in range(16):
    out_channels = len(imscore[im])
    if layer_id in skip[args.v]:
        cfg.append(out_channels)    
        layer_id += 1
        im += 1
        continue
    imscore_copy = imscore[im]
    thre = np.sum(imscore_copy)/out_channels
    thre = thre * 0.99
    imscore_copy = torch.Tensor(imscore[im])
    mask = imscore_copy.gt(thre).float().cuda()             
                     
    cfg.append(int(torch.sum(mask)))                   

    layer_id += 1
    im += 1
    continue
