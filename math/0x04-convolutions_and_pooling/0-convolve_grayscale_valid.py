#!/usr/bin/env python3
""" convolutions"""
import numpy as np
import matplotlib.pyplot as plt


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    c_h = h - kh + 1
    conv = np.zeros((m, c_h, c_h))
    for h in range(c_h):
        for w in range(c_h):
            square = images[:, h: h + kh, w: w + kw]
            insert = np.sum(np.sum(square * kernel, axis=1), axis=1)
            conv[:, h, w] = insert
    return conv
