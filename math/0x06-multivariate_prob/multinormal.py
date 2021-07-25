#!/usr/bin/env python3
"""mulitvaritive probability"""
import numpy as np


class MultiNormal():
    """represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n, d = data.shape
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
