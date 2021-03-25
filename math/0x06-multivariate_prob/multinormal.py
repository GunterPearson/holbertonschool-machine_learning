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
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.cov = np.matmul((data - self.mean).T,
                             data - self.mean) / (n - 1)
