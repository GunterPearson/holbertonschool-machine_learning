#!/usr/bin/env python3
""" return size in matrix form"""


def matrix_shape(matrix):
    """ reurn matrix shape"""
    result = []
    i = 0
    z = 0
    result.append(len(matrix))
    for list in matrix:
        for width in list:
            i += 1
            if type(width) == type(list):
                z = len(width)
    i /= 2
    result.append(int(i))
    if z is not 0:
        result.append(z)
    return result
