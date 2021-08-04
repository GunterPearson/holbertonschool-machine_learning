#!/usr/bin/env python3
""" clustering """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance"""
    # X.shape = (250, 2)
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) != int or kmin < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None
    if kmax is None:
        kmax = X.shape[1]
    k_result = []
    min_var = []
    for i in range(kmin, kmax + 1):
        centroids, classes = kmeans(X, i, iterations)
        k_result.append((centroids, classes))
        if i == kmin:
            min = variance(X, centroids)
        var_at_i = variance(X, centroids)
        min_var.append(min - var_at_i)
    return k_result, min_var
