#!/usr/bin/env python3
"""multivaritive cov"""
import numpy as np


def mean_cov(X):
    """ caclulate mean covariance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    m = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - m).T, X - m) / (n - 1)
    return m, cov
