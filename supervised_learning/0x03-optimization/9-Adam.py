#!/usr/bin/env python3
""" mini batch"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """ update using adam"""
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * (grad ** 2)
    vdw_c = vdw / (1 - beta1 ** t)
    sdw_c = sdw / (1 - beta2 ** t)
    new_var = var - alpha * vdw_c / (epsilon + np.sqrt(sdw_c))
    return new_var, vdw, sdw
