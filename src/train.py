import os, sys

import numpy as np
import random
import visdom

import torch
import torch.optim as optim

from torch.utils.data import DataLoader



import argparse
import time, datetime

from data import *
from model import *


vgg_types = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type = str, default = '../data/massachusettes_roads_dataset', help = 'file root')
    parser.add_argument('--workers', type = int, help = 'number of data loading workers', default = 12)
    parser.add_argument('--nEpoch', type = int, default = 100, help = 'number of epochs to train for')
    parser.add_argument('--vgg', type = str, default = 'vgg19',  help = 'pretrained vgg net (choices: vgg11, vgg11_bn, vgg13, vgg13_bn,\nvgg16, vgg16_bn, vgg19, vgg19_bn)')
    parser.add_argument('--layers', nargs='+', type = int, help = 'the extracted features from vgg')
    
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--env', type = str, default = "topo", help = 'visdom environment')

    parser.add_argument('--prelud', type = float, default = 0.2, help = 'prelu coeff for down-sampling block')
    parser.add_argument('--preluu', type = float, default = 0., help = 'prelu coeff for up-sampling block')
    parser.add_argument('--dropout', type = float, default = 0., help = 'dropout coeff for all blocks')
    parser.add_argument('--bn', type = bool, default = True, help = 'batch-normalization coeff for all blocks')
    parser.add_argument('--color', type = bool, default = True, help = 'True if input is RGB, False otherwise')

    parser.add_argument('--lamb', type = float, default = 0.0001, help = 'loss coeff for img reconstruction task')

    opt = parser.parse_args()
    print (opt)

    assert(opt.vgg in vgg_types), "The chosen VGG net does not exist!"

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    transform_train = JointTransform2D(crop = (224, 224))
    transform_test = JointTransform2D(resize = 224, crop = False, p_flip = 0)

    trainset = RoadSet(opt.dataRoot, "train", transform_train)
    valset = RoadSet(opt.dataRoot, "val", transform_test)

    trainloader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = int(opt.workers))
    valloader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = int(opt.workers))
    
    print('training set: ', len(trainset))
    print('val set     : ', len(valset))

    # Set up model 
    nf = 4 if opt.color else 2
    unet = UNet(nf, bnd = opt.bn, prelud = opt.prelud, dropoutd = opt.dropout, 
                    bnu = opt.bn, preluu = opt.preluu, dropoutu = opt.dropout)
    vggnet = VGGNet(opt.vgg, opt.layers, use_cuda)
    
    unet.to(device)
    vggnet.to(device)

    model = TopologyNet(unet, vggnet, opt.K)

    



    





