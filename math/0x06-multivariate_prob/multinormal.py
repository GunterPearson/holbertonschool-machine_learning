#!/usr/bin/env python3
"""multivaritive cov"""
import numpy as np


class MultiNormal():
    """ mulit normal class"""
    def __init__(self, data):
        """ initializer"""
        d, n = data.shape
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """ caclulate mean covariance"""
        d, n = X.shape
        m = np.mean(X, axis=0, keepdims=True)
        cov = np.matmul((X - m).T, X - m) / (n - 1)
        return m, cov
