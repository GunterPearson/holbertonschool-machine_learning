#!/usr/bin/env python3
""" clustering """
import numpy as np


def initialize(X, k):
    """ initialize """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k < 1:
        return None
    # X.shape is (250, 2)
    n, d = X.shape
    small = np.min(X, axis=0)
    large = np.max(X, axis=0)
    new = np.random.uniform(
        low=small,
        high=large,
        size=(k, d)
    )
    # --- Select random points for centroids
    # centroids = X.copy()
    # np.random.shuffle(centroids)
    # cent = centroids[:k]
    # -- We can use 2 scatter plot
    # plt.scatter(X[:, 0], X[:, 1], s=10)
    # plt.scatter(cent[:, 0], cent[:, 1], c='r', s=8)
    # plt.show()
    # -- OR WE CAN USE ONE plot
    # plt.plot(X[:, 0], X[:, 1], 'ro', cent[:, 0], cent[:, 1], 'bs')
    # plt.show()
    return new
