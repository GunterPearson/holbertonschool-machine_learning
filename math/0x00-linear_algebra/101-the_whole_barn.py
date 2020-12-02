#!/usr/bin/env python3
""" the whole barn"""
import numpy as np


def add_matrices(mat1, mat2):
    """ add two matrices"""
    new = np.array(mat1)
    second = np.array(mat2)
    try:
        result = new + second
    except ValueError:
        return None
    return result.tolist()
