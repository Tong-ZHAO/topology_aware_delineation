#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable


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


    def forward(self, x):

        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        x = self.u1(x4)
        x = self.u2(torch.cat([x, x3], 1))
        x = self.u3(torch.cat([x, x2], 1))
        x = self.u4(torch.cat([x, x1], 1))

        return x