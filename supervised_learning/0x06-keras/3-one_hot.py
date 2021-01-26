#!/usr/bin/env python3
""" one hot"""
import numpy as np


def one_hot(labels, classes=None):
    """ one hot matrix"""
    m = labels.shape[0]
    if classes is None:
        classes = len(np.unique(labels, axis=0))
    return np.eye(classes)[labels]
