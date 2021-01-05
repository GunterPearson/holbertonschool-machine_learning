#!/usr/bin/env python3
""" hot encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encode"""
    try:
        x = np.eye(classes)[Y].T
        return x
    except Exception:
        return None
