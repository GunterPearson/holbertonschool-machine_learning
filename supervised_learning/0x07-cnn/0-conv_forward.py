#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ conv forward"""
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph = ((h_prev * (sh - 1)) - sh + kh) // 2
        pw = ((w_prev * (sw - 1)) - sw + kw) // 2
    out_h = int((h_prev - kh + 2 * ph) / sh) + 1
    out_w = int((w_prev - kh + 2 * pw) / sw) + 1
    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    conv = np.zeros((m, out_h, out_w, c_new))
    for c in range(c_new):
        for h in range(out_h):
            for w in range(out_w):
                square = padded[:, h * sh: h * sh + kh, w * sw: w * sw + kw]
                insert = np.sum(square * W[..., c], axis=(1, 2, 3))
                conv[:, h, w, c] = insert
    result = conv + b
    return activation(result)
