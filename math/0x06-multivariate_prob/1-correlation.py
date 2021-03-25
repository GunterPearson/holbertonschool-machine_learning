#!/usr/bin/env python3
"""multivaritive cov"""
import numpy as np


def correlation(C):
    """ correlation"""
    if not isinstance(C, np.ndarray):
        raise TypeError('X must be a 2D numpy.ndarray')
    if C.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    x = np.diag(1 / np.sqrt(np.diag(C)))
    cor = x @ C @ x
    return cor
