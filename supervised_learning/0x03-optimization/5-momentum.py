#!/usr/bin/env python3
""" mini batch"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ update using momentum"""
    dw_prev = beta1 * v + (1 - beta1) * grad
    new_w = var - (alpha * dw_prev)
    return new_w, dw_prev
