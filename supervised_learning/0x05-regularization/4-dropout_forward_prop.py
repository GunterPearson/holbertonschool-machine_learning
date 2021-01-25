#!/usr/bin/env python3
""" regularize"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ dropout with forward prop"""
    cache = {}
    cache["A0"] = X
    for x in range(1, L + 1):
        z = np.matmul(weights["W" + str(x)],
                      cache["A" + str(x - 1)]) + weights["b" + str(x)]
        if x == L:
            tmp = np.exp(z)
            A = tmp / np.sum(tmp, axis=0, keepdims=True)
            cache["A" + str(x)] = A
        else:
            A = np.tanh(z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            cache["D" + str(x)] = D
            cache["A" + str(x)] = A / keep_prob
    return cache
