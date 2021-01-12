#!/usr/bin/env python3
""" normalize"""
import numpy as np


def normalize(X, m, s):
    """ standardize matrix"""
    return (X - m) / s
