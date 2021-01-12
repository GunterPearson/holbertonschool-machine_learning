#!/usr/bin/env python3
""" normalize"""
import numpy as np


def shuffle_data(X, Y):
    """ shuffle data"""
    s = np.random.permutation(X.shape[0])
    return X[s], Y[s]
