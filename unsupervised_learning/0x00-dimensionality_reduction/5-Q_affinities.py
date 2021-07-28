#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np


def Q_affinities(Y):
    """calculates the Q affinities"""
    sum = np.sum(np.square(Y), 1)
    D = np.add(np.add(-2 * np.matmul(Y, Y.T), sum).T, sum)
    top = (1 + D) ** (-1)
    np.fill_diagonal(top, 0)
    bottom = np.sum(top)
    Q = top / bottom
    return Q, top
