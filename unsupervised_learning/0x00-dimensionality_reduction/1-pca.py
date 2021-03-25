#!/usr/bin/env python3
""" dimension reduction"""
import numpy as np


def pca(X, ndim):
    """ PCA """
    new = X - np.mean(X, axis=0)
    U, s, V = np.linalg.svd(new)
    W = V[:ndim].T
    return np.matmul(new, W)
