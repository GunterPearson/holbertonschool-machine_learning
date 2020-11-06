#!/usr/bin/env python3
""" matrix addition"""


def add_arrays(arr1, arr2):
    """ adds two arrays and returns them"""
    try:
        return [arr1[idx] + arr2[idx] for idx in range(len(arr1))]
    except:
        return None
