#!/usr/bin/env python3
"""sum squared"""


def summation_i_squared(n):
    """returns sum of n squared"""
    if n < 1 or type(n) != int:
        return None
    return sum(map(lambda n: pow(n, 2), range(n + 1)))
