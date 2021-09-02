#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class BidirectionalCell():
    """represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """class constructor"""
        # 10, 15, 5 = (i, h, o)
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """feed forward"""
        # (8, 15) (8, 15) (8, 10)
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        h_next = np.tanh(h_x.T @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward"""
        h_x = np.concatenate((h_next.T, x_t.T), axis=0)

        h_prev = np.tanh(h_x.T @ self.Whb + self.bhb)
        return h_prev
