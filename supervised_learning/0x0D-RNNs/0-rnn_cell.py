#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class RNNCell:
    """RNNCell class
    """
    def __init__(self, i, h, o):
        """Initializer
        Arguments:
            i {int} -- Is dimensionality of the data
            h {int} -- Is dimensionality of hidden state
            o {int} -- Is dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        """Calculates the softmax
        Arguments:
            z {np.ndarray} -- Contains array input
        Returns:
            np.ndarray -- Softmaxed numpy arrat
        """
        ex = np.exp(z)
        return ex / np.sum(ex, 1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step
        Arguments:
            h_prev {np.ndarrat} -- Contains the previous hidden state
            x_t {np.ndarray} -- Contains the data input of the cell
        Returns:
            tuple(np.ndarray) -- Contains the next hidden state, the output of
            the cell
        """
        stacked = np.hstack((h_prev, x_t))
        h_next = np.tanh(stacked @ self.Wh + self.bh)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
