#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ convolve grayscale same"""
    hk, wk = kernel.shape
    m, hm, wm = images.shape
    P = int((hk - 1) / 2)
    padded = np.pad(images, ((0, 0), (P, P), (P, P)), 'constant')
    convoluted = np.zeros((m, hm, wm))
    for h in range(hm):
        for w in range(wm):
            square = padded[:, h: h + hk, w: w + wk]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
