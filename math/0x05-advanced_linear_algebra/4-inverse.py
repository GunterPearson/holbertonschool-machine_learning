#!/usr/bin/env python3
""" advanced linear algebra"""


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


def cofactor(matrix):
    """ cofactor of matrix"""
    if len(matrix) == 1:
        return [[1]]

    height = len(matrix)
    width = len(matrix[0])
    mat = []
    for i in range(height):
        temp = []
        for j in range(width):
            sub = []
            for row in (matrix[:i] + matrix[i + 1:]):
                sub.append(row[:j] + row[j + 1:])
            sign = (-1) ** ((i + j) % 2)
            temp.append(determinant(sub) * sign)
        mat.append(temp)
    return mat


def adjugate(matrix):
    """ adjugate matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]
    le = len(matrix)

    cof = cofactor(matrix)
    ad = []
    for i in range(le):
        ad.append([])
        for j in range(le):
            ad[i].append(cof[j][i])

    return ad


def inverse(matrix):
    """ inverse of matrix"""
    ad = adjugate(matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    return [[y / det for y in x] for x in ad]
