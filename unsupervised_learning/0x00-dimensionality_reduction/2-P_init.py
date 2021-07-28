#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np


def P_init(X, perplexity):
    """initializes all variables"""
    # x.shape and perp = (2500, 50), 30.0
    n = X.shape[0]
    X1 = X[:, :, None]
    D = ((X1 - X1.T) ** 2).sum(1)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, beta, H
