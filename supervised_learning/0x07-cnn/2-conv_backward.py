#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ conv back"""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]
    sh = stride[0]
    sw = stride[1]
    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    da = np.zeros(A_prev.shape)
    dw = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for n in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    da[n, h * sh: h * sh + kh, w * sw: w * sw + kw,
                       :] += dZ[n, h, w, c] * W[..., c]
                    dw[..., c] += padded[n, h * sh: h * sh +
                                         kh, w * sw: w * sw + kw,
                                         :] * dZ[n, h, w, c]
    if padding == 'same':
        da = da[:, ph:-ph, pw:-pw, :]
    return da, dw, db
