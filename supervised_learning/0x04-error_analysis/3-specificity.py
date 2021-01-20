#!/usr/bin/env python3
""" create confusion"""
import numpy as np


def specificity(confusion):
    """ calculate specificity"""
    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    fp = np.sum(confusion, axis=0) - tp
    tn = np.sum(confusion) - tp - fp - fn
    return tn / (fp + tn)
