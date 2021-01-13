#!/usr/bin/env python3
""" mini batch"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ update using rms"""
    sdw = beta2 * s + (1 - beta2) * (grad ** 2)
    new_var = var - alpha * (grad / (epsilon + np.sqrt(sdw)))
    return new_var, sdw
