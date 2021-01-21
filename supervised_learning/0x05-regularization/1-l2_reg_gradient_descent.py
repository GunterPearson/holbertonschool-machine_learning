#!/usr/bin/env python3
""" regularize"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ gradient descent with l2 norm"""
    m = Y.shape[1]
    for x in reversed(range(1, L + 1)):
        AN1 = cache["A" + str(x - 1)]
        A0 = cache["A" + str(x)]
        W0 = weights["W" + str(x)]
        if x == L:
            dz = A0 - Y
        else:
            dz = da * (A0 * (1 - A0))
        db = dz.mean(axis=1, keepdims=True)
        dw = np.matmul(dz, AN1.T) / m + ((lambtha / m) * weights['W' + str(x)])
        da = np.matmul(W0.T, dz)
        weights['W' + str(x)] -= (alpha * dw)
        weights['b' + str(x)] -= (alpha * db)
