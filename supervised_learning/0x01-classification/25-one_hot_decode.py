#!/usr/bin/env python3
""" hot encode """
import numpy as np


def one_hot_decode(one_hot):
    """ decode hot"""
    if type(one_hot) is not np.ndarray or one_hot.shape < (2, 2):
        return None
    try:
        return np.argmax(one_hot.T, axis=1)
    except Exception:
        return None
