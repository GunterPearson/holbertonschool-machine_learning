#!/usr/bin/env python3
""" create confusion"""
import numpy as np


def sensitivity(confusion):
    """ calculate sensitivity"""
    ay = np.sum(confusion, axis=1)
    tp = np.diagonal(confusion)
    return tp / ay
