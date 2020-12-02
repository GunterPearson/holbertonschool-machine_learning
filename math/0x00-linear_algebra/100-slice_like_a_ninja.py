#!/usr/bin/env python3
""" slice like a ninja"""


def np_slice(matrix, axes={}):
    """ slice matrix based on axis given"""
    form = (max(axes) + 1) * [slice(None)]
    for k, v in axes.items():
        form[k] = slice(*v)
    result = matrix[tuple(form)]
    return result
