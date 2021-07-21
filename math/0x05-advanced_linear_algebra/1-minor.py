#!/usr/bin/env python3
""" minor """


def minor(matrix):
    """ that calculates the minor of a matrix """
    if not all(type(row) == list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or type(matrix) != list:
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 1:
        return [[1]]
    if not all(len(r) == len(matrix) for r in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    mino = []
    for x in range(len(matrix)):
        t = []
        for y in range(len(matrix[0])):
            s = []
            for row in (matrix[:x] + matrix[x + 1:]):
                s.append(row[:y] + row[y + 1:])
            t.append(determinant(s))
        mino.append(t)
    return mino


def determinant(matrix):
    """ that calculates the determinant of a matrix """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        x = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return x
    det = 0
    for x, num in enumerate(matrix):
        r = [r for r in matrix[1:]]
        temp = []
        for row in r:
            new = []
            for j in range(len(matrix)):
                if j != x:
                    new.append(row[j])
            temp.append(new)
        det += num * determinant(temp) * (-1) ** x
    return det
