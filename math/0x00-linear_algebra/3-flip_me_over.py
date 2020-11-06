#!/usr/bin/env python3
""" transpose matrix """


def matrix_transpose(matrix):
    """ transpose matrix"""
    new = []
    for idx in range(len(matrix[0])):
        mid = []
        for row in matrix:
            mid.append(row[idx])
        new.append(mid)
    return new
