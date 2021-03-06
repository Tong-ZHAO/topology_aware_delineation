#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))


class blockUNet(nn.Module):

    def __init__(self, in_ch, out_ch, name, transposed = False, bn = True, prelu = 0, dropout = 0):
        super(blockUNet, self).__init__()
        block = nn.Sequential()
    
        if prelu == 0:
            block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        else:
            block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(prelu, inplace=True))
        
        if not transposed:
            block.add_module('%s_conv' % name, nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
            
        if bn:
            block.add_module('%s_bn' % name, nn.BatchNorm2d(out_ch))
            
        if dropout > 0:
            block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

        self.layer = block


    def forward(self, x):

        x = self.layer(x)

        return x


class UNet(nn.Module):

    def __init__(self, nf, bnd = True, prelud = 0.2, dropoutd = 0, bnu = True, preluu = 0, dropoutu = 0):
        # nf: 2 if gray, 4 if rgb

        super(UNet, self).__init__()
        # 224 x 224 x nf -> 112 x 112 x (2 nf)
        self.d1 = blockUNet(nf, nf * 2, "down_1", transposed = False, bn = bnd, prelu = prelud, dropout = dropoutd)
        # 112 x 112 x (2 nf) -> 56 x 56 x (4 nf)
        self.d2 = blockUNet(nf * 2, nf * 4, "down_2", transposed = False, bn = bnd, prelu = prelud, dropout = dropoutd)
        # 56 x 56 x (4 nf) -> 28 x 28 x (8 nf)
        self.d3 = blockUNet(nf * 4, nf * 8, "down_3", transposed = False, bn = bnd, prelu = prelud, dropout = dropoutd)
        # 28 x 28 x (8 nf) -> 14 x 14 x (8 nf)
        self.d4 = blockUNet(nf * 8, nf * 8, "down_4", transposed = False, bn = bnd, prelu = prelud, dropout = dropoutd)

        # 14 x 14 x (8 nf) -> 28 x 28 x (8 nf)
        self.u1 = blockUNet(nf * 8, nf * 8, "up_4", transposed = True, bn = bnu, prelu = preluu, dropout = dropoutu)
        # 28 x 28 x (16 nf) -> 56 x 56 x (4 nf)
        self.u2 = blockUNet(nf * 16, nf * 4, "up_3", transposed = True, bn = bnu, prelu = preluu, dropout = dropoutu)
        # 56 x 56 x (8 nf) -> 112 x 112 x (2 nf)
        self.u3 = blockUNet(nf * 8, nf * 2, "up_2", transposed = True, bn = bnu, prelu = preluu, dropout = dropoutu)
        # 112 x 112 x (4 nf) -> 224 x 224 x 1
        self.u4 = blockUNet(nf * 4, 1, "up_1", transposed = True, bn = bnu, prelu = preluu, dropout = dropoutu)
        self.active = nn.Sigmoid()

    def forward(self, x):

        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        x = self.u1(x4)
        x = self.u2(torch.cat([x, x3], 1))
        x = self.u3(torch.cat([x, x2], 1))
        x = self.u4(torch.cat([x, x1], 1))

        return self.active(x)


def choose_vgg(name):

    f = None

    if name == 'vgg11':
        f = models.vgg11(pretrained = True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained = True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained = True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained = True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained = True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained = True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained = True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained = True)

    for params in f.parameters():
        params.requires_grad = False

    return f


class VGGNet(nn.Module):

    def __init__(self, name, layers, cuda = True):

        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

        self.mean = pretrained_mean.cuda() if cuda else pretrained_mean
        self.std = pretrained_std.cuda() if cuda else pretrained_std

    def forward(self, x):

        x = (x - self.mean) / self.std

        results = []

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x.view(x.shape[0], -1))

        return results


class TopologyNet(nn.Module):

    def __init__(self, unet, vggnet, K = 1):

        super(TopologyNet, self).__init__()
        self.unet = unet
        self.vggnet = vggnet
        self.K = K


    def forward(self, x, y):

        results = []

        for i in range(self.K):
            input = torch.cat((x, y), dim = 1)
            y = self.unet(input)
            y_topo = self.vggnet(torch.cat((y, y, y), dim = 1))
            results.append([y, y_topo])

        return results