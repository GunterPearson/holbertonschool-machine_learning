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

    def pdf(self, x):
        """ return pdf"""
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        res = np.matmul((x - self.mean).T, np.linalg.inv(self.cov))
        res = np.exp(np.matmul(res, (x - self.mean)) / -2)
        res /= np.sqrt(pow(2 * np.pi, x.shape[0]) * np.linalg.det(self.cov))
        return res[0][0]
