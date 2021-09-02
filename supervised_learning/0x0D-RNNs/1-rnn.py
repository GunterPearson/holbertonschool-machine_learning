#!/usr/bin/env python3
"""RNN module"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation"""
    t, m, i = X.shape
    H = []
    H.append(h_0)
    Y = []
    h_prev = h_0
    for time in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[time])
        H.append(h_next)
        Y.append(y)
        h_prev = h_next
    return np.array(H), np.array(Y)
