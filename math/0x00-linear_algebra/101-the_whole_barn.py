#!/usr/bin/env python3
""" the whole barn"""


def add_matrices(mat1, mat2):
    """ add two matrices"""
    try:
        if len(mat1) != len(mat2):
            return None
        result = []
        for x, y in zip(mat1, mat2):
            temp = add_matrices(x, y)
            if temp is None:
                return None
            result.append(temp)
        return result
    except TypeError:
        return mat1 + mat2
