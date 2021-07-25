#!/usr/bin/env python3
"""mulitvaritive probability"""
import numpy as np


class MultiNormal():
    """represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """that calculates the mean and covariance of a data set"""
        d, n = X.shape
        m = np.mean(X, axis=1, keepdims=True)
        C = np.matmul((X - m), (X - m).T) / (n - 1)
        return m, C

    def pdf(self, x):
        """ calculate a PDF"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        m = self.mean
        cov = self.cov
        bottom = np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(cov)))
        inv = np.linalg.inv(cov)
        exp = (-.5 * np.matmul(np.matmul((x - m).T, inv), (x - m)))
        result = (1 / bottom) * np.exp(exp[0][0])
        return result
