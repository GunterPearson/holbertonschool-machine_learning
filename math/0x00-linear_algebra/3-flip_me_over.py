#!/usr/bin/env python3
""" Transpose a matrix """


def matrix_transpose(matrix):
    """ Given matrix perform transpose"""
    new = []
    for idx in range(len(matrix[0])):
        mid = []
        for row in matrix:
            mid.append(row[idx])
        new.append(mid)
    return new
