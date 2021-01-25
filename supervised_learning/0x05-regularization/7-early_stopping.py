#!/usr/bin/env python3
""" regularization"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ early stopping"""
    if opt_cost - cost > threshold:
        return (False, 0)
    else:
        count += 1
        if count != patience:
            return(False, count)
    return (True, count)
