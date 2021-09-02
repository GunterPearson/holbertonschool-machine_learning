#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class GRUCell():
    """represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """class constructor"""
        # 10, 15, 5 = (i, h, o)
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """return softmax"""
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """feed forward"""
        # (8, 15) (8, 10)
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        z_gate = self.sigmoid((h_x.T @ self.Wz) + self.bz)
        r_gate = self.sigmoid((h_x.T @ self.Wr) + self.br)

        o_x = np.concatenate(((r_gate * h_prev).T, x_t.T), axis=0)

        t = np.tanh((o_x.T @ self.Wh) + self.bh)

        h_next = (1 - z_gate) * h_prev + z_gate * t

        y = self.softmax((h_next @ self.Wy) + self.by)
        return h_next, y
