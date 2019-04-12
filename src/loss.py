#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO


import numpy as np

import torch
import torch.nn as nn


def iterative_loss(preds, vgg_loss, pred_loss, vgg_true, y_true):

    K, N = len(preds), len(vgg_true)
    loss_pred, loss_vgg = 0., 0.

    for i in range(K):
        loss_pred += pred_loss(preds[i][0], y_true)
        for j in range(N):
            loss_vgg += vgg_loss(preds[i][1][j], vgg_true[j])

    return loss_pred, loss_vgg / N