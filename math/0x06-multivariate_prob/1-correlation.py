#!/usr/bin/env python3
"""multivaritive cov"""
import numpy as np


def correlation(C):
    """ correlation"""
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    x = np.diag(1 / np.sqrt(np.diag(C)))
    cor = x @ C @ x
    return cor
