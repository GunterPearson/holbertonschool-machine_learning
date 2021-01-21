#!/usr/bin/env python3
""" regularize"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ regularize cost using L2"""
    norm = 0
    for layer in range(1, L + 1):
        norm += np.linalg.norm(weights["W" + str(layer)])
    return cost + (lambtha / (2 * m)) * norm
