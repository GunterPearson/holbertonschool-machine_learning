#!/usr/bin/env python3
""" return size in matrix form"""


def matrix_shape(matrix):
    """Return shape of given list-matrix"""
    shape = []
    try:
        while(len(matrix) > 0):
            shape.append(len(matrix))
            matrix = matrix[0]
    except TypeError:
        pass
    return shape
