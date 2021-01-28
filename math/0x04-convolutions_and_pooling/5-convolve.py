#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ convolve with multiple filters"""
    kh, kw, kc, nc = kernels.shape
    m, hm, wm, cm = images.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((hm - 1) * sh + kh - hm) / 2) + 1
        pw = int(((wm - 1) * sw + kw - wm) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    ch = int((hm + 2 * ph - kh) / sh) + 1
    cw = int((wm + 2 * pw - kw) / sw) + 1
    convoluted = np.zeros((m, ch, cw, nc))
    for c in range(nc):
        for h in range(ch):
            for w in range(cw):
                square = padded[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
                insert = np.sum(square * kernels[..., c], axis=(1, 2, 3))
                convoluted[:, h, w, c] = insert
    return convoluted
