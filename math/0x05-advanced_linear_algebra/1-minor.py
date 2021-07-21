#!/usr/bin/env python3
""" minor """


def minor(matrix):
    """ that calculates the minor of a matrix """
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
        temp = []
        for row in matrix[1:]:
            new = []
            for j in range(len(matrix)):
                if j != x:
                    new.append(row[j])
            temp.append(new)
        det += num * determinant(temp) * (-1) ** x
    return det
