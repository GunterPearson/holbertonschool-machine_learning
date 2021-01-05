#!/usr/bin/env python3
""" hot encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encode"""
    if classes <= 0:
        return None
    x = np.eye(classes)[Y].T
    return x
