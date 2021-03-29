#!/usr/bin/env python3
""" Probability"""
import numpy as np


def likelihood(x, n, P):
    """ likelihood"""
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


def marginal(x, n, P, Pr):
    """ marginal"""
    like = likelihood(x, n, P)
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        z = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(z)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any([h < 0 or h > 1 for h in P]):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any([h < 0 or h > 1 for h in Pr]):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    return (like * Pr).sum()


def posterior(x, n, P, Pr):
    """ posterior"""
    m = marginal(x, n, P, Pr)
    return likelihood(x, n, P) * Pr / m
