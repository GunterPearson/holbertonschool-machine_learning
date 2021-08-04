#!/usr/bin/env python3
""" clustering """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model"""
    # X.shape = (12500, 2)
    # --- Select random points for centroids
    # plt.scatter(X[:, 0], X[:, 1], s=10)
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=13)
    # plt.show()
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) != int or k < 1:
        return None, None, None
    n, d = X.shape
    pi = np.ones(k) / k
    cen, classes = kmeans(X, k)
    cov_mat = np.tile(np.identity(d), (k, 1, 1))
    return pi, cen, cov_mat
