#!/usr/bin/env python3
""" clustering """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None

    gaus_sum = 0
    gauss = np.zeros((k, n))
    for i in range(k):
        aux = pi[i] * pdf(X, m[i], S[i])
        gauss[i] = aux
        gaus_sum += aux
    gauss = gauss / gaus_sum
    likehood = np.sum(np.log(gaus_sum))
    return gauss, likehood
