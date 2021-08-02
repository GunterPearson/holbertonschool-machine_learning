#!/usr/bin/env python3
""" clustering """
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if type(k) is not int or k <= 0:
        return None, None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    # X.shape is (250, 2), k = 5
    n, d = X.shape
    centroids = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, d)
    )
    classes = np.zeros(X.shape[0])
    for i in range(iterations):
        # temp to test against new centroid for change
        temp = centroids.copy()
        # find the distance using squared distance
        distances = np.sum(np.square(X - centroids[:, np.newaxis]), axis=2)
        # assign each value to class depending on min distance
        classes = np.argmin(distances, axis=0)
        for j in range(k):
            # arrange X by classes and pull out where class equal to j
            if len(X[classes == j]) == 0:
                centroids[j] = np.random.uniform(
                                low=np.min(X, axis=0),
                                high=np.max(X, axis=0),
                                size=(1, d))
            else:
                centroids[j] = np.mean(X[classes == j], axis=0)
        distances = np.sum(np.square(X - centroids[:, np.newaxis]), axis=2)
        classes = np.argmin(distances, axis=0)
        # --- Select random points for centroids
        # plt.scatter(X[:, 0], X[:, 1], s=10)
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=13)
        # plt.show()
        if np.all(centroids == temp):
            return centroids, classes
    return centroids, classes
