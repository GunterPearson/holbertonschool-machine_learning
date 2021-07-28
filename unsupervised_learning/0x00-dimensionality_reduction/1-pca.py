#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    # X is a (n x d) matrix (2500, 784)
    n, d = X.shape
    X1 = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X1, full_matrices=True)
    return np.matmul(X1, V[:ndim].T)
