#!/usr/bin/env python3
"""method definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """definiteness of matrix"""
    err = 'matrix must be a numpy.ndarray'
    if not isinstance(matrix, np.ndarray):
        raise TypeError(err)
    my_len = matrix.shape[0]
    if len(matrix.shape) != 2 or my_len != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None
    x, y = np.linalg.eig(matrix)

    if all(x > 0):
        return 'Positive definite'
    elif all(x >= 0):
        return 'Positive semi-definite'
    elif all(x < 0):
        return 'Negative definite'
    elif all(x <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
