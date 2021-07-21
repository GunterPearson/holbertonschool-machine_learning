#!/usr/bin/env python3
""" determinant """
import numpy as np


def determinant(matrix):
    """ that calculates the determinant of a matrix """
    if not all(type(row) == list for row in matrix) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    return round(np.linalg.det(matrix))
