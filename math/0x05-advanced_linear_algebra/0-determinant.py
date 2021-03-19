#!/usr/bin/env python3
""" advanced linear algebra"""


def determinant(matrix):
    """ determerminant"""

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for r in matrix:
        if not isinstance(r, list):
            raise TypeError("matrix must be a list of lists")

    for r in matrix:
        if len(r) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        determ = ((matrix[0][0] * matrix[1][1])
                  - (matrix[0][1] * matrix[1][0]))
        return determ

    determ = 0
    for i, j in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        temp = []
        for r in row:
            a = []
            for c in range(len(matrix)):
                if c != i:
                    a.append(r[c])
            temp.append(a)
        determ += j * (-1) ** i * determinant(temp)
    return determ
