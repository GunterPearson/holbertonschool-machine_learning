#!/usr/bin/env python3
"""hidden markov chain"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    if len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if (n1 != n2) or type(P) is not np.ndarray:
        return None
    diag = np.diagonal(P)
    if (diag == 1).all():
        return True
    if not (diag == 1).any():
        return False
    test = np.where(diag == 1)
    if len(test[0]) == 1:
        for x in range(n1):
            new = P[:, [x]]
            count = np.where(new > 0)[0]
            if len(count) < 2 and x + 1 != n1:
                return False
            if len(count) >= 1 and x + 1 == n1:
                return True
    if len(test[0]) == 2:
        return True
    return False
