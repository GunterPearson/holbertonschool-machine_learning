#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale"""
    m, hm, wm = images.shape
    kh, kw = kernel.shape
    c_h = hm - kh + 1
    c_w = wm - kh + 1
    conv = np.zeros((m, c_h, c_h))
    for h in range(c_h):
        for w in range(c_w):
            square = images[:, h: h + kh, w: w + kw]
            insert = np.sum(kernel * square, axis=1).sum(axis=1)
            conv[:, h, w] = insert
    return conv
