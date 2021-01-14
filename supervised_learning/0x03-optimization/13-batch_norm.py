#!/usr/bin/env python3
""" batch normilization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ batch normilization"""
    m = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    z_norm = (Z - m) / (np.sqrt(var + epsilon))
    z_tild = gamma * z_norm + beta
    return z_tild
