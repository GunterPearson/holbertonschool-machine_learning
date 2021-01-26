#!/usr/bin/env python3
""" one hot"""
import numpy as np


def one_hot(labels, classes=None):
    """ one hot matrix"""
    m = labels.shape[0]
    result = np.eye(m)[labels]
    return result
