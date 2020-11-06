#!/usr/bin/env python3
""" concat matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concat matrix based on axis """
    if axis == 0:
        return mat1 + mat2
    return [mat1[idx] + mat2[idx] for idx in range(len(mat1))]
