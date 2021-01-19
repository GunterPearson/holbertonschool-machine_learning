#!/usr/bin/env python3
""" create confusion"""
import numpy as np


def precision(confusion):
    """ calculate precision on confusion matrix"""
    x = np.sum(confusion, axis=0)
    y = np.max(confusion, axis=1)
    return y / x
