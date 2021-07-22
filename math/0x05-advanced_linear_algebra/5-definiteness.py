#!/usr/bin/env python3
""" advanced linear algebra"""
import numpy as np


def definiteness(matrix):
    """ definitness"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    my_len = matrix.shape[0]
    if len(matrix.shape) != 2 or my_len != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None
    eVals, eVecs = np.linalg.eig(matrix)
    if all(eVals > 0):
        return "Positive definite"
    if all(eVals >= 0):
        return "Positive semi-definite"
    if all(eVals < 0):
        return "Negative definite"
    if all(eVals <= 0):
        return "Negative semi-definite"
    else:
        return 'Indefinite'
