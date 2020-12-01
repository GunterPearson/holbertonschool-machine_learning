#!/usr/bin/env python3
""" ridin bareback"""

def mat_mul(mat1, mat2):
    """ multiply two matrix"""
    if len(mat1[0]) == len(mat2):
        res = [[0 for x in range(len(mat2[0]))] for y in range(len(mat1))]
        for x in range(len(mat1)):
            for y in range(len(mat2[0])):
                for z in range(len(mat2)):
                    res[x][y] += mat1[x][z] * mat2[z][y]
        return res
    return None
