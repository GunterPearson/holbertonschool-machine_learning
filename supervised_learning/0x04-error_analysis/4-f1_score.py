#!/usr/bin/env python3
""" create confusion"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """return f1 score"""
    prec = precision(confusion)
    recal = sensitivity(confusion)
    return 2 / ((1 / recal) + (1 / prec))
