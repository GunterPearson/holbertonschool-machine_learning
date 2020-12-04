#!/usr/bin/env python3
""" Squashed like sardines"""


def matrix_shape(matrix):
    """ Return shape as list of integers """
    shape = []
    if type(matrix) != list:
        pass
    else:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])
    return shape


def recur(m1, m2, axis, ax):
    """ recursion """
    if axis != ax:
        return [recur(m1[i], m2[i], axis, ax + 1) for i in range(len(m1))]
    else:
        m1.extend(m2)
        return m1


def cat_matrices(mat1, mat2, axis=0):
    """ concatenates two matrices """
    from copy import deepcopy
    new = deepcopy(mat1)
    new2 = deepcopy(mat2)
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i == axis:
            continue
        if shape1[i] != shape2[i]:
            return None
    r = recur(new, new2, axis, 0)
    return r
