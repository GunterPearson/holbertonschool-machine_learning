#!/usr/bin/env python3
""" concat matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concat matrix based on axis """
    if axis == 0:
        return mat1.copy() + mat2.copy()
    return [mat1[idx].copy() + mat2[idx].copy() for idx in range(len(mat1))]
