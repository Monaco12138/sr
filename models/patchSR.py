from models import register
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import Namespace

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), padding_mode='replicate', bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class patchSR(nn.Module):
    def __init__(self, args, conv=default_conv):
                # edsr-baseline: r16,f64,x2
        super(patchSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.p = args.p
        self.scale = scale
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append( conv(n_feats, n_feats, kernel_size) )

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        
        self.out_dim = args.n_colors
        # define tail module
        if scale == 1:
            m_tail = [
                conv(n_feats, args.n_colors, kernel_size)
            ]
        else:
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
       
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//self.p, self.p, w//self.p, self.p)\
            .permute(0, 2, 4, 1, 3, 5).reshape(-1, 1, self.p, self.p)
        # x: [b * n*c, 1, p, p]

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x: [b* n*c, 1, p*s, p*s] 
        x = x.reshape(b, -1, c, self.p * self.scale, self.p * self.scale).reshape(b, h//self.p, w//self.p, c, self.p * self.scale, self.p * self.scale)\
            .permute(0, 3, 1, 4, 2, 5).reshape(b, c, h * self.scale, w * self.scale )

        return x

@register('patchSR')
def make_patchSR(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=1, p = 3):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.p = p
    args.scale = [scale]

    args.n_colors = 1
    return patchSR(args)

# make_edsr_baseline = register('edsr_baseline')(make_edsr_baseline)