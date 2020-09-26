# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:12:53 2019

@author: ASUS
"""

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models.cifar3 as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from ptflops import get_model_complexity_info

# Prune settings 
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the resnet')
parser.add_argument('--model', default='./results-resnet/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--arch', default='resnet', type=str, 
                    help='architecture to use')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

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
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")

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
    return 10000 * correct / float(len(test_loader.dataset))

acc = test(model)
acc=acc.numpy()

num_parameters = sum([param.nelement() for param in model.parameters()])
print("number of parameters: "+str(num_parameters)+"\n")
with open(os.path.join(args.save, "resprune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")

# 计算模型参数
with torch.cuda.device(0):
  net = model
  flops, params = get_model_complexity_info(net, (3, 32,32), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)


#******************************************************************************************************#

imscore = [

]


n = (args.depth - 2) // 6
cfg = [[16]*n, [32]*n, [32]*n]
cfg = [item for sub_list in cfg for item in sub_list]

layer_id = 1
im = 0

cfg_mask1 = []
cfg_mask2 = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if layer_id == 1:
            layer_id = layer_id + 1
            continue
        if layer_id % 2 == 0:
            out_channels = m.weight.data.shape[0]
            layer = layer_id // 2 - 1
            if out_channels == cfg[layer]:
                cfg_mask1.append(torch.ones(out_channels))
                layer_id += 1
                continue

            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:cfg[layer]]
            assert arg_max_rev.size == cfg[layer], "size of arg_max_rev not correct"
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask1.append(mask)
            layer_id += 1
            continue
            
        if layer_id % 2 == 1:
            out_channels = m.weight.data.shape[0]
            imscore_copy = imscore[im]
            arg_max = np.argsort(imscore_copy)
            layer = layer_id // 2 - 1
            m = out_channels - cfg[layer]
            arg_max_rev = arg_max[m:]
            assert arg_max_rev.size == cfg[layer], "size of arg_max_rev not correct"            
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1                      
            cfg_mask2.append(mask.clone())                      
            layer_id += 1
            im += 1
            continue
        layer_id += 1

newmodel = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset, cfg=cfg)

print(newmodel)
if args.cuda:
    newmodel.cuda()


layer_id_in_cfg = 0
conv_count = 1


for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):                  
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        
        if conv_count in [20, 38]:
            mask = cfg_mask1[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue            
        
        if conv_count % 2 == 0:
            mask = cfg_mask1[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue            
        
        if conv_count % 2 == 1:
            mask = cfg_mask2[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue

    elif isinstance(m0, nn.BatchNorm2d):
        if conv_count == 2:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()            
            continue
        if conv_count % 2 == 1:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            continue
        if conv_count % 2 == 0:
            mask = cfg_mask2[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue       

    elif isinstance(m0, nn.Linear):
        mask = cfg_mask2[layer_id_in_cfg-1]
        idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if idx.size == 1:
            idx = np.resize(idx, (1,))
        m1.weight.data = m0.weight.data[:, idx.tolist()].clone()
        m1.bias.data = m0.bias.data.clone()



torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'respruned.pth.tar'))

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
print(newmodel)
model = newmodel
acc = test(model)
acc=acc.numpy()

print("number of parameters: "+str(num_parameters))
with open(os.path.join(args.save, "respruned.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")

with torch.cuda.device(0):
  net = model
  flops, params = get_model_complexity_info(net, (3, 32,32), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)
