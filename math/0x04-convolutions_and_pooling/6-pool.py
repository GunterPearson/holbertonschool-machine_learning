#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ pooling"""
    m, hm, wm, cm = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ch = int((hm - kh) / sh) + 1
    cw = int((wm - kw) / sw) + 1
    convoluted = np.zeros((m, ch, cw, cm))
    for h in range(ch):
        for w in range(cw):
            square = images[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
            if mode == 'max':
                insert = np.max(square, axis=(1, 2))
            else:
                insert = np.average(square, axis=(1, 2))
            convoluted[:, h, w, :] = insert
    return convoluted
