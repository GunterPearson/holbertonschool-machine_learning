#!/usr/bin/env python3
""" regularize"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ gradient descent with dropout"""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for x in range(L, 0, -1):
        A_prev = cache["A" + str(x - 1)]
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        inv = 1 - np.square(A_prev)
        da = np.matmul(weights["W" + str(x)].T, dz)
        if x > 1:
            da = da * cache["D" + str(x - 1)]
            da = da / keep_prob
        dz = da * inv
        weights["W" + str(x)] -= alpha * dw
        weights["b" + str(x)] -= alpha * db
