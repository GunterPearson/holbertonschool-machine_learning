#!/usr/bin/env python3
"""multivaritive cov"""
import numpy as np


class MultiNormal():
    """ mulit normal class"""
    def __init__(self, data):
        """ initializer"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """ caclulate mean covariance"""
        d, n = X.shape
        m = np.mean(X, axis=1, keepdims=True)
        cov = np.matmul((X - m), (X - m).T) / (n - 1)
        return m, cov
