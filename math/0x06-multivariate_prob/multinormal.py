#!/usr/bin/env python3
"""multivaritive cov"""
import numpy as np


class MultiNormal():
    """ mulit normal class"""
    def __init__(self, data):
        """ initializer"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """caclulate mean covariance"""
        d, n = X.shape
        mean = np.expand_dims(np.mean(X, axis=1), axis=1)
        cov = np.matmul((X - mean), (X - mean).T) / (n - 1)
        return mean, cov
