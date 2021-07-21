#!/usr/bin/env python3
""" determinant """
import numpy as np


def determinant(matrix):
    """ that calculates the determinant of a matrix """
    if not all(type(row) == list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or type(matrix) != list:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if not all(len(r) == len(matrix) for r in matrix):
        raise ValueError("matrix must be a square matrix")
    return round(np.linalg.det(matrix))
