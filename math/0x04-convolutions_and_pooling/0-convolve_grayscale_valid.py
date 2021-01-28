#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale"""
    m = images.shape[0]
    hm = images.shape[1]
    wm = images.shape[2]
    hk = kernel.shape[0]
    wk = kernel.shape[1]
    ch = hm - hk + 1
    cw = wm - wk + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            matrix = images[:, h: h + hk, w: w + wk]
            v = np.sum(matrix * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = v
    return(convoluted)
