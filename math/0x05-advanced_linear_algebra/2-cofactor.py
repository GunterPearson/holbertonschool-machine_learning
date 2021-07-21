#!/usr/bin/env python3
""" advanced linear algebra"""


def cofactor(matrix):
    """cofactor"""
    return minor(matrix)


def determinant(matrix):
    """ determerminant"""
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


def minor(matrix):
    """ minor of matrix """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    mino = []
    for x in range(len(matrix)):
        t = []
        for y in range(len(matrix[0])):
            s = []
            for row in (matrix[:x] + matrix[x + 1:]):
                s.append(row[:y] + row[y + 1:])
            sign = (-1) ** ((x + y) % 2)
            t.append(determinant(s) * sign)
        mino.append(t)
    return mino
