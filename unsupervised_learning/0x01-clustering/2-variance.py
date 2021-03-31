#!/usr/bin/env python3
""" clustering """
import numpy as np


def variance(X, C):
    """ variance """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    try:
        L = np.sqrt(np.sum((X - C[:, np.newaxis])**2, axis=2))
        return np.sum(np.min(L, axis=0)**2)
    except Exception:
        return None
