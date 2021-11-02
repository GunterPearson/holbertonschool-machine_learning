#!/usr/bin/env python3
"""Write a function that calculates the shape of a matrix"""


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
