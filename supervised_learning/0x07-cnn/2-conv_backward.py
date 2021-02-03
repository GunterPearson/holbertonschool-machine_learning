#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ conv back"""
    hk = W.shape[0]
    wk = W.shape[1]
    nc = W.shape[3]
    m = A_prev.shape[0]
    hm = A_prev.shape[1]
    wm = A_prev.shape[2]
    st1 = stride[1]
    st0 = stride[0]
    if padding == "valid":
        pad0 = 0
        pad1 = 0
    else:
        pad0 = int(((hm - 1) * st0 + hk - hm) / 2) + 1
        pad1 = int(((wm - 1) * st1 + wk - wm) / 2) + 1
    out_h = int((hm + 2 * pad0 - hk) / st0) + 1
    out_w = int((wm + 2 * pad1 - wk) / st1) + 1
    x_pad = np.pad(A_prev, ((0, 0), (pad0, pad0), (
                           pad1, pad1), (0, 0)), 'constant')
    dX = np.zeros(x_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(nc):
                    dX[i, h * st0: h * st0 + hk,
                       w * st1: w * st1 + wk, :] += dZ[
                       i, h, w, c] * W[:, :, :, c]
                    dW[:, :, :, c] += x_pad[i, h * st0: h * st0 + hk,
                                            w * st1: w * st1 + wk,
                                            :] * dZ[i, h, w, c]
    if padding == "same":
        dX = dX[:, pad0:-pad0, pad1:-pad1, :]
    return(dX, dW, db)
