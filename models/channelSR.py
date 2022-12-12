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
        padding=(kernel_size//2), bias=bias)

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

class channelSR(nn.Module):
    def __init__(self, args, conv=default_conv):
                # edsr-baseline: r16,f64,x2
        super(channelSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

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
        x = x.reshape( b*c, 1, h, w )

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        
        x = x.reshape(b, c, x.shape[-2], x.shape[-1])

        return x

@register('channelSR')
def make_channelSR(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]

    args.n_colors = 1
    return channelSR(args)

# make_edsr_baseline = register('edsr_baseline')(make_edsr_baseline)