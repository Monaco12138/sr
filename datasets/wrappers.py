import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        #self.patch = patch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale, 训练中所使用的均为整数倍scale

        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]             # h_lr, w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]   # 
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):                 # [C, H, W]
                if hflip:                   # 相当于把图像上下颠倒
                    x = x.flip(-2)
                if vflip:                   # 相当于把图像左右颠倒
                    x = x.flip(-1)
                if dflip:                   #相当于把图像向右旋转90°
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # if self.patch != None:
        #     c, h_lr, w_lr = crop_lr.shape[:]
        #     #c, h_hr, w_hr = crop_hr.shape[:]

        #     crop_lr = crop_lr.reshape(c, h_lr//self.patch, self.patch, w_lr//self.patch, self.patch)\
        #         .permute(1, 3, 0, 2, 4).reshape(-1, 1, self.patch, self.patch)

        #     # crop_hr = crop_hr.shape( c, h_hr//(self.patch*s), self.patch*s, w_hr//(self.patch*s), self.patch*s)\
        #     #     .permute(1, 3, 0, 2, 4).reshape(-1, 1, self.patch*s, self.patch*s)
            
        #     # crop_lr: [n*c, 1, p, p], crop_hr: [c, h, w]
        return {
            'inp': crop_lr,
            'gt': crop_hr
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)
            ) 
        )
