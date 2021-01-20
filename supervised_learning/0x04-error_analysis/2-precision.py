#!/usr/bin/env python3
""" create confusion"""
import numpy as np


def precision(confusion):
    """ calculate precision on confusion matrix"""
    py = np.sum(confusion, axis=0)
    tp = np.diagonal(confusion)
    return tp / py
