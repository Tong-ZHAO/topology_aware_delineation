#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO


import os, sys

import numpy as np
import matplotlib.pyplot as plt
import torch

import argparse
import time, datetime

from data import *
from model import *
from loss import *

def plot_image(image, title, gray = True):

    plt.figure()
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)

if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type = str, default = '../data/massachusetts_roads_dataset/test_data/20728960_15.png', help = 'image path')
    parser.add_argument('--label', type = str, default = '../data/massachusetts_roads_dataset/test_pred/20728960_15.png', help = 'label path')
    parser.add_argument('--K', type = int, default = 3, help = 'number of iterative steps')
    # Model settings
    parser.add_argument('--prelud', type = float, default = 0.2, help = 'prelu coeff for down-sampling block')
    parser.add_argument('--preluu', type = float, default = 0., help = 'prelu coeff for up-sampling block')
    parser.add_argument('--dropout', type = float, default = 0., help = 'dropout coeff for all blocks')
    parser.add_argument('--bn', type = bool, default = True, help = 'batch-normalization coeff for all blocks')
    parser.add_argument('--color', type = bool, default = True, help = 'True if input is RGB, False otherwise')
    parser.add_argument('--model', type = str, default = '../log/finetune_k3/model_70.t7', help = 'path for pretrained model')
    # Test setting
    parser.add_argument('--thresh', type = float, default = 0.3, help = 'the threshold for identifying boundary')

    opt = parser.parse_args()
    print (opt)

    # Check GPU availability
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    transform_test = JointTransform2D(resize = 224, crop = False, p_flip = 0)

    # Set up model 
    nf = 4 if opt.color else 2
    unet = UNet(nf, bnd = opt.bn, prelud = opt.prelud, dropoutd = opt.dropout,
                    bnu = opt.bn, preluu = opt.preluu, dropoutu = opt.dropout)

    if use_cuda:
        unet.load_state_dict(torch.load(opt.model)["net"])
    else:
        unet.load_state_dict(torch.load(opt.model, map_location = 'cpu')["net"])

    # Criterion
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    image, label = Image.open(opt.image), Image.open(opt.label).convert('L')
    transform_test = JointTransform2D(resize = 224, crop = False, p_flip = 0)

    input, target = transform_test(image, label)
    lpredicts = []

    with torch.no_grad():

        input = torch.unsqueeze(input, 0)
        target = torch.unsqueeze(target, 0)
        init_target = torch.zeros_like(target)

        if use_cuda:
            input = input.cuda()
            target = target.cuda()
            init_target = init_target.cuda()

        for k in range(opt.K):
            curr_input = torch.cat((input, init_target), dim = 1)
            init_target = unet(curr_input)
            lpredicts.append(torch.squeeze(init_target).data.cpu().detach().numpy())

    # plot images
    plot_image(np.asarray(image), "Input Image")
    plot_image(np.asarray(label), "Target Image", True)

    for i, img in enumerate(lpredicts):
        plot_image(img > opt.thresh, "Iteration %d" % (i + 1), True)

    plt.show()




