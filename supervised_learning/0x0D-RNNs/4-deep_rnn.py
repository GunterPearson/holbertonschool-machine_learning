#!/usr/bin/env python3
"""RNN module"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation"""
    T, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((T + 1, l, m, h))
    h, o = rnn_cells[-1].Wy.shape
    Y = np.zeros((T, m, o))

    for i, cell in enumerate(rnn_cells):
        if i == 0:
            for t in range(1, T + 1):
                H[t, i], _ = cell.forward(H[t - 1, i], X[t - 1])
        else:
            for t in range(1, T + 1):
                H[t, i], Y[t - 1] = cell.forward(H[t - 1, i], H[t, i - 1])
    return H, Y
