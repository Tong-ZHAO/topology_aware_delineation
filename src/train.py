#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO


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
from loss import *
from utils import *


vgg_types = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

def train_epoch(net, dataloader, optimizer, vgg_loss, pred_loss, mu, vis, epoch, cuda):

    all_train_loss = 0.
    all_pred_loss = 0.
    all_vgg_loss = 0.

    N = len(dataloader)

    for i, data in enumerate(dataloader, 0):

        optimizer.zero_grad()
        imgs, labels = data
        init_labels = torch.zeros_like(labels)
    
        if cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
            init_labels = init_labels.cuda()
            
        preds = net(imgs, init_labels)
        vgg_labels = net.vggnet(torch.cat((labels, labels, labels), dim = 1))

        curr_pred_loss, curr_vgg_loss = iterative_loss(preds, vgg_loss, pred_loss, vgg_labels, labels)
        train_loss = curr_pred_loss + mu * curr_vgg_loss
        train_loss.backward()
        optimizer.step()

        all_train_loss += train_loss.data.cpu().detach()
        all_pred_loss += curr_pred_loss.data.cpu().detach()
        all_vgg_loss += curr_vgg_loss.data.cpu().detach()

    vis.plot("loss", "train_total", "Loss Per Epoch", epoch, all_train_loss / N)
    vis.plot("loss", "train_pred", "Loss Per Epoch", epoch, all_pred_loss / N)
    vis.plot("loss", "train_vgg", "Loss Per Epoch", epoch, all_vgg_loss / N)

    print("Epoch %d (Train): total loss - %f, pred loss - %f, vgg loss - %f" % (epoch, all_train_loss / N, all_pred_loss / N, all_vgg_loss / N))


def val_epoch(net, dataloader, vgg_loss, pred_loss, mu, vis, epoch, cuda):

    print("Epoch %d (Val)" % epoch)

    all_val_loss = 0.
    all_pred_loss = 0.
    all_vgg_loss = 0.
    all_bce_loss = 0.

    N = len(dataloader)

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):

            optimizer.zero_grad()
            imgs, labels = data
            init_labels = torch.zeros_like(labels)
        
            if cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
                init_labels = init_labels.cuda()
                
            preds = net(imgs, init_labels)
            vgg_labels = net.vggnet(torch.cat((labels, labels, labels), dim = 1))

            curr_pred_loss, curr_vgg_loss = iterative_loss(preds, vgg_loss, pred_loss, vgg_labels, labels)
            curr_bce_loss = pred_loss(preds[-1][0], labels)

            all_val_loss += (curr_pred_loss + mu * curr_vgg_loss).cpu().detach()
            all_pred_loss += curr_pred_loss.data.cpu().detach()
            all_vgg_loss += curr_vgg_loss.data.cpu().detach()
            all_bce_loss += curr_bce_loss.data.cpu().detach()

            if i == 0:
                vis.show("groundtruth", "Target", labels[0])
                for i in range(len(preds)):
                    vis.show("pred_" + str(i), "Prediction in Level " + str(i), preds[i][0][0])

    vis.plot("loss", "val_total", "Loss Per Epoch", epoch, all_val_loss / N)
    vis.plot("loss", "val_pred", "Loss Per Epoch", epoch, all_pred_loss / N)
    vis.plot("loss", "val_vgg", "Loss Per Epoch", epoch, all_vgg_loss / N)

    vis.plot("reconstruction bce", "val_bce", "Reconstruction Loss - Val", epoch, all_bce_loss / N)

    print("Epoch %d (Val): total loss - %f, pred loss - %f, vgg loss - %f" % (epoch, all_val_loss / N, all_pred_loss / N, all_vgg_loss / N))

    return all_bce_loss


def update_model(net, path):
    
    state = {'net': net.unet.state_dict(),}
    torch.save(state, path)



if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type = str, default = '../data/massachusetts_roads_dataset', help = 'file root')
    parser.add_argument('--workers', type = int, help = 'number of data loading workers', default = 12)
    parser.add_argument('--nEpoch', type = int, default = 100, help = 'number of epochs to train for')
    parser.add_argument('--vgg', type = str, default = 'vgg19',  help = 'pretrained vgg net (choices: vgg11, vgg11_bn, vgg13, vgg13_bn,\nvgg16, vgg16_bn, vgg19, vgg19_bn)')
    parser.add_argument('--layers', nargs='+', type = int, help = 'the extracted features from vgg')
    parser.add_argument('--K', type = int, default = 1, help = 'number of iterative steps')
    parser.add_argument('--batchSize', type = int, default = 32, help = 'batchsize')
    
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--env', type = str, default = "topo", help = 'visdom environment')

    parser.add_argument('--prelud', type = float, default = 0.2, help = 'prelu coeff for down-sampling block')
    parser.add_argument('--preluu', type = float, default = 0., help = 'prelu coeff for up-sampling block')
    parser.add_argument('--dropout', type = float, default = 0., help = 'dropout coeff for all blocks')
    parser.add_argument('--bn', type = bool, default = True, help = 'batch-normalization coeff for all blocks')
    parser.add_argument('--color', type = bool, default = True, help = 'True if input is RGB, False otherwise')

    parser.add_argument('--mu', type = float, default = 0.1, help = 'loss coeff for vgg features')

    opt = parser.parse_args()
    print (opt)

    assert(opt.vgg in vgg_types), "The chosen VGG net does not exist!"

    # Check GPU availability
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data Loader
    transform_train = JointTransform2D(crop = (224, 224))
    transform_test = JointTransform2D(resize = 224, crop = False, p_flip = 0)

    trainset = RoadSet(opt.dataRoot, "train", transform_train)
    valset = RoadSet(opt.dataRoot, "val", transform_test)
    trainloader = DataLoader(trainset, batch_size = opt.batchSize, shuffle = True, num_workers = int(opt.workers))
    valloader = DataLoader(trainset, batch_size = opt.batchSize, shuffle = True, num_workers = int(opt.workers))
    
    print('training set: ', len(trainset))
    print('val set     : ', len(valset))

    # Visdom & log
    vis = VisdomPlotter(opt.env)
    now = datetime.datetime.now()
    save_path = now.isoformat()
    dir_name =  os.path.join('../log', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')

    # Set up model 
    nf = 4 if opt.color else 2
    unet = UNet(nf, bnd = opt.bn, prelud = opt.prelud, dropoutd = opt.dropout, 
                    bnu = opt.bn, preluu = opt.preluu, dropoutu = opt.dropout)
    vggnet = VGGNet(opt.vgg, opt.layers, use_cuda)
    unet.to(device)
    vggnet.vgg.to(device)
    model = TopologyNet(unet, vggnet, opt.K)

    # Criterion
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    best_loss = np.inf

    print("Begin training...")

    for epoch in range(opt.nEpoch):
        train_epoch(model, trainloader, optimizer, mse_loss, bce_loss, opt.mu, vis, epoch, use_cuda)

        if epoch % 5 == 0:
            print("Begin validation...")
            curr_loss = val_epoch(model, valloader, mse_loss, bce_loss, opt.mu, vis, epoch, use_cuda)
            if curr_loss < best_loss:
                best_loss = curr_loss
                update_model(model, os.path.join(dir_name, "model_%d.t7" % epoch))



