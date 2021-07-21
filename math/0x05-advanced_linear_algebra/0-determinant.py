#!/usr/bin/env python3
""" determinant """


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
            new = []
            for j in range(len(matrix)):
                if j != x:
                    new.append(row[j])
            temp.append(new)
        det += P * determinant(temp) * (-1) ** x
    return det
