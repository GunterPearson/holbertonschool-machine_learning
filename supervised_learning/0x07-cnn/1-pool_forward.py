#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ pool forward"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = int((h_prev - kh) / sh) + 1
    out_w = int((w_prev - kw) / sw) + 1
    conv = np.zeros((m, out_h, out_w, c_prev))
    for h in range(out_h):
        for w in range(out_w):
            square = A_prev[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
            if mode == "max":
                insert = np.max(square, axis=(1, 2))
            else:
                insert = np.average(square, axis=(1, 2))
            conv[:, h, w, :] = insert
    return conv
