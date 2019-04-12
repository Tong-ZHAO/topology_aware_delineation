#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO


import numpy as np
import pickle
import os, sys

from PIL import Image

import torchvision
import torch.utils.data as data

from torchvision import transforms
from torchvision.transforms import functional


class RoadSet(data.Dataset):
    """Dataset wrapping images and target meshes for Massachusettes Roads dataset.

    Params: 
        file_root (str): the path where the dataset is stored
        file_name (str): choose among - train, val and test
        transform : the pytorch version transformer
    """

    def __init__(self, file_root, file_name, transform):

        self.file_root = file_root

        # Read file names
        self.file_name = file_name
        self.file_lists = os.listdir(os.path.join(self.file_root, self.file_name + "_data"))
        self.file_nums = len(self.file_lists)
        self.transform = transform


    def __getitem__(self, index):

        data_name = os.path.join(os.path.join(self.file_root, self.file_name + "_data"), self.file_lists[index])
        pred_name = os.path.join(os.path.join(self.file_root, self.file_name + "_pred"), self.file_lists[index])

        data = Image.open(data_name)
        pred = Image.open(pred_name)

        img, label = self.transform(data, pred)

        return img, label


    def __len__(self):
        return self.file_nums



class JointTransform2D:
    """
    From: https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/dataset.py

    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.
    Args:
        resize: tuple describing the size of the resize. If bool(resize) evaluates to False, no resize will
            be taken.
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, resize = False, crop = (224, 224), p_flip = 0.5, p_random_affine = 0, long_mask = False):
        self.resize = resize
        self.crop = crop
        self.p_flip = p_flip
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):

        if self.resize:
            image, mask = functional.resize(image, self.resize), functional.resize(mask, self.resize)

        # random crop
        if self.crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, self.crop)
            image, mask = functional.crop(image, i, j, h, w), functional.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = functional.hflip(image), functional.hflip(mask)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = transforms.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = functional.affine(image, *affine_params), functional.affine(mask, *affine_params)

        # transforming to tensor
        image = functional.to_tensor(image)

        if not self.long_mask:
            mask = functional.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask
