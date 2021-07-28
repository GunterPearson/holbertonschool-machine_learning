#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """calculates the gradients of Y"""
    n, d = Y.shape
    Q, num = Q_affinities(Y)
    z = np.zeros((n, d))
    PQ = P - Q
    for i in range(n):
        z[i, :] = np.sum(
            np.tile(PQ[:, i] * num[:, i], (d, 1)).T * (Y[i, :] - Y), 0)
    return z, Q
