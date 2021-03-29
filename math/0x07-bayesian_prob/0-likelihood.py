#!/usr/bin/env python3
""" Probability"""
import numpy as np


def likelihood(x, n, P):
    """ likelihood"""
    if n < 1:
        raise ValueError("n must be a positive integer")
    if x < 0:
        y = "x must be an integer that is greater than or equal to 0"
        raise ValueError(y)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if any(j >= 1 and j <= 0 for j in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    new = np.math.factorial(n)
    f = np.math.factorial(x)
    nf = np.math.factorial(n - x)
    result = new / (f * nf)
    suc = pow(P, x)
    fail = pow(1 - P, n - x)
    likelihood = result * suc * fail
    return likelihood
