#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class LSTMCell():
    """represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """class constructor"""
        # 10, 15, 5 = (i, h, o)
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """return softmax"""
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """feed forward"""
        # (8, 15) (8, 15) (8, 10)
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        f = self.sigmoid((h_x.T @ self.Wf) + self.bf)
        i = self.sigmoid((h_x.T @ self.Wu) + self.bu)
        o = self.sigmoid((h_x.T @ self.Wo) + self.bo)

        g = np.tanh((h_x.T @ self.Wc) + self.bc)

        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
