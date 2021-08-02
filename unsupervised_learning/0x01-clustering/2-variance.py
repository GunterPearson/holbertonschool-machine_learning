#!/usr/bin/env python3
""" clustering """
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance"""
    distances = np.sum(np.square(X - C[:, np.newaxis]), axis=2)
    min_distances = np.min(distances, axis=0)
    intra = np.sum(min_distances)
    return intra
