#!/usr/bin/env python3
""" hot encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encode"""
    if type(Y) is not np.ndarray:
        return None
    try:
        x = np.eye(classes)[Y].T
        return x
    except Exception:
        return None
