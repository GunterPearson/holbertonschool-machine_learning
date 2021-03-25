#!/usr/bin/env python3
""" dimension reduction"""
import numpy as np


def pca(X, var=0.95):
    """ pca """
    u, s, vh = np.linalg.svd(X)
    c = np.cumsum(s)
    variance = c / np.sum(s)
    r = np.argwhere(variance >= var)[0, 0]
    return vh[:r + 1].T
