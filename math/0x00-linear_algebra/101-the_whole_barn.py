#!/usr/bin/env python3
""" the whole barn"""


def add_matrices(mat1, mat2):
    """ add two matrices"""
    import numpy as np
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    try:
        result = m1 + m2
    except ValueError:
        return None
    return result.tolist()
