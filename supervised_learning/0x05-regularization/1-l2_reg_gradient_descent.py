#!/usr/bin/env python3
""" regularize"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ gradient descent with l2 norm"""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for x in range(L, 0, -1):
        A_prev = cache["A" + str(x - 1)]
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        inv = 1 - np.square(A_prev)
        dz = np.matmul(weights["W" + str(x)].T, dz) * inv
        reg = (1 - lambtha * alpha / m)
        weights["W" + str(x)] = reg * weights["W" + str(x)] - alpha * dw
        weights["b" + str(x)] -= alpha * db
