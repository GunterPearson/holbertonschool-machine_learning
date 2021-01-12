#!/usr/bin/env python3
""" normalize"""
import numpy as np


def normalization_constants(X):
    """ normalization function"""
    return np.mean(X, axis=0), np.std(X, axis=0)
