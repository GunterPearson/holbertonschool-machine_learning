#!/usr/bin/env python3
""" bayesian probability"""
import numpy as np


def likelihood(x, n, P):
    """calculates the likelihood"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that "
                         "is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.where(P > 1, 1, 0).any() or np.where(P < 0, 1, 0).any():
        raise ValueError("All values in P must be in the range [0, 1]")
    new = np.math.factorial(n)
    f = np.math.factorial(x)
    nf = np.math.factorial(n - x)
    result = new / (f * nf)
    suc = pow(P, x)
    fail = pow(1 - P, n - x)
    likelihood = result * suc * fail
    return likelihood
