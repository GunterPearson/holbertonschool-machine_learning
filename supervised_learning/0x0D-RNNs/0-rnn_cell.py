#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class RNNCell():
    """
    represents a cell of a simple RNN
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

    def softmax(self, x):
        """Calculates the softmax

        Arguments:
            x {np.ndarray} -- Contains array input

        Returns:
            np.ndarray -- Softmaxed numpy arrat
        """
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step

        Arguments:
            h_prev {np.ndarrat} -- Contains the previous hidden state
            x_t {np.ndarray} -- Contains the data input of the cell

        Returns:
            tuple(np.ndarray) -- Contains the next hidden state, the output of
            the cell
        """
        one = np.concatenate((h_prev.T, x_t.T), axis=0)
        two = np.matmul(one.T, self.Wh) + self.bh
        h_next = np.tanh(two)
        o = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(o)
        return h_next, y
