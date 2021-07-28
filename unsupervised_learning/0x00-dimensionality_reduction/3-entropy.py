#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np


def HP(Di, beta):
    """calculates the Shannon entropy"""
    # Di shape and beta shape = (2499,),  (1,)
    top = np.exp(-Di * beta)
    bottom = np.sum(top)
    pi = top / bottom
    hi = -1 * np.sum(pi * np.log2(pi))
    return hi, pi
