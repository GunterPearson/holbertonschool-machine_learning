#!/usr/bin/env python3
"""RNN module"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation"""
    T, m, i = X.shape
    _, j = h_0.shape
    _, k = h_t.shape

    Hf = np.zeros((T, m, j))
    Hb = np.zeros((T, m, k))

    for t in range(0, T):
        Hf[t] = bi_cell.forward(h_0, X[t])
        h_0 = Hf[t]

    for x in range(0, T)[::-1]:
        Hb[x] = bi_cell.backward(h_t, X[x])
        h_t = Hb[x]

    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)
    return H, Y
