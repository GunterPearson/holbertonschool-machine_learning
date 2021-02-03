#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ pool backprop"""
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = int((h_prev - kh) / sh) + 1
    out_w = int((w_prev - kw) / sw) + 1
    DA = np.zeros(A_prev.shape)
    for n in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(c_new):
                    square = A_prev[n, h * sh: h * sh + kh, w * sw:
                                    w * sw + kw, c]
                    if mode == "max":
                        mask = np.where(square == np.max(square), 1, 0)
                    else:
                        mask = np.ones(square.shape)
                        mask /= kh * kw
                    DA[n, h * sh: h * sh + kh, w * sw: w * sw + kw,
                       c] += mask * dA[n, h, w, c]
    return DA
