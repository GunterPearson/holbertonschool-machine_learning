#!/usr/bin/env python3
"""RNNs"""
import numpy as np


class RNNCell():
    """
    represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        return softmax
        """
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    def forward(self, h_prev, x_t):
        """
        feed forward
        """
        one = np.concatenate((h_prev.T, x_t.T), axis=0)
        two = np.matmul(one.T, self.Wh) + self.bh
        h_next = np.tanh(two)
        o = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(o)
        return h_next, y
