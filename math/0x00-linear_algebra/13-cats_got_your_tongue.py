#!/usr/bin/env python3
""" cats got your tounge"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concat two matrix based on axis"""
    return np.concatenate((mat1, mat2), axis=axis)
