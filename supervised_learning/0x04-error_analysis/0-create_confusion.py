#!/usr/bin/env python3
""" create confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates confusion matrix"""
    x = np.matmul(labels.T, logits)
    return x
