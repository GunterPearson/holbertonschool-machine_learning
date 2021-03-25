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
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        n = x.shape[0]
        m = self.mean
        c = self.cov
        t = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(c))
        j = np.linalg.inv(c)
        ran = (-0.5 * np.matmul(np.matmul((x - m).T, j), x - m))
        pdf = (1 / t) * np.exp(ran[0][0])
        return pdf
