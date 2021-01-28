#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    c_h = h - kh + 1
    c_w = w - kw + 1
    conv = np.zeros((m, c_h, c_h))
    for w in range(c_w):
        for h in range(c_h):
            square = images[:, h: h + kh, w: w + kw]
            insert = np.sum(np.sum(square * kernel, axis=1), axis=1)
            conv[:, h, w] = insert
    return conv
