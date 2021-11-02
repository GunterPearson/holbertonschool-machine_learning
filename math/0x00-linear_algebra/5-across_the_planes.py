#!/usr/bin/env python3
""" add 2d matrix """


def add_matrices2D(mat1, mat2):
    """add 2 matrices or return None"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    try:
        return [[mat1[idx][i] + mat2[idx][i] for i in range(len(mat1[0]))]
                for idx in range(len(mat1))]
    except IndexError:
        return None
