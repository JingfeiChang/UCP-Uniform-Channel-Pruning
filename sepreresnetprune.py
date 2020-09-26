# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:36:34 2019

@author: ASUS
"""

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models.cifar3 as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from ptflops import get_model_complexity_info


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the resnet')
parser.add_argument("--reduction", type=int, default=16)
parser.add_argument('--model', default='./results-se_resnet5/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--arch', default='se_preresnet', type=str, 
                    help='architecture to use')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, reduction=args.reduction)

if args.cuda:
    model.cuda()
    
print(model)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) best_acc: {:f}"
              .format(args.model, checkpoint['epoch'], best_acc))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / float(len(test_loader.dataset))

acc = test(model)
acc = acc.numpy()

num_parameters = sum([param.nelement() for param in model.parameters()])
print("number of parameters: "+str(num_parameters)+"\n")
with open(os.path.join(args.save, "sepreresnetprune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")

with torch.cuda.device(0):
  net = model
  flops, params = get_model_complexity_info(net, (3, 32,32), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)

imscore=[
        
]

skip = {
    #'A': [1, 4, 6, 14]
    'A':[]
}                                       

layer_id = 0
im = 0
cfg = []

for m in [0,54]:
    out_channels = len(imscore[im])
    if layer_id in skip[args.v]:
        cfg.append(out_channels)   
        layer_id += 1
        im += 1
        continue
    imscore_copy = imscore[im]
    thre = np.sum(imscore_copy)/out_channels
    thre = thre * 0.999
    imscore_copy = torch.Tensor(imscore[im])
    mask = imscore_copy.gt(thre).float().cuda()             
                     
    cfg.append(int(torch.sum(mask)))                 

    layer_id += 1
    im += 1
    continue
