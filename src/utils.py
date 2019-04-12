#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO


import numpy as np
import pickle
import visdom
import torch


class VisdomPlotter(object):
    """Plots to Visdom
    """

    def __init__(self, env_name):
        self.viz = visdom.Visdom(port = 8097, env = env_name)
        self.env = env_name
        self.plots = {}
        self.images = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def show(self, var_name, title_name, image):
        self.viz.image(image.data.cpu().squeeze(), env = self.env, win = var_name, opts = dict(
            title = title_name,
            caption = title_name
        ))
