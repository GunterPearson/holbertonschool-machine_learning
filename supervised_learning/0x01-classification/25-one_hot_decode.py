#!/usr/bin/env python3
""" hot encode """
import numpy as np


def one_hot_decode(one_hot):
    """ decode hot"""
    try:
        return np.argmax(one_hot.T, axis=1)
    except Exception:
        return None
