#!/usr/bin/env python3
""" concat matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concat matrix based on axis """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
    return [mat1[idx].copy() + mat2[idx].copy() for idx in range(len(mat1))]
