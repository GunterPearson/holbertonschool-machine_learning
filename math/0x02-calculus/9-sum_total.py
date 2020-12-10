#!/usr/bin/env python3
""" summs"""


def summation_i_squared(n):
    """ squared sum"""
    if n < 1 or type(n) != int:
        return None
    return sum(map(lambda n: pow(n, 2), range(n + 1)))
