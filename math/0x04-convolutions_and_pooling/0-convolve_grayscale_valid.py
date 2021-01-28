#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    c_h = int(np.floor(h - kh + 1))
    c_w = int(np.floor(w - kw + 1))
    conv = np.zeros((m, c_h, c_h))
    for w in range(c_h):
        for h in range(c_w):
            square = images[:, h: h + kh, w: w + kw]
            insert = np.sum(kernel * square, axis=(1, 2))
            conv[:, h, w] = insert
    return conv
