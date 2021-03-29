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
        x = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return x

    det = 0
    for x, num in enumerate(matrix):
        temp = []
        P = matrix[0][x]
        for row in matrix[1:]:
            l = []
            for j in range(len(matrix)):
                if j != x:
                    l.append(row[j])
            temp.append(l)
        det += P * determinant(temp) * (-1) ** x
    return det
