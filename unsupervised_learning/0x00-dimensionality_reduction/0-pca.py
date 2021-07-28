#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    # X is a (n x d) matrix (50, 6)
    n, d = X.shape
    K = 1
    U, S, V = np.linalg.svd(X, full_matrices=True)
    for x in range(d):
        varian = (np.sum(S[:K]) / np.sum(S))
        if varian >= var:
            return V[:K].T
        K += 1
