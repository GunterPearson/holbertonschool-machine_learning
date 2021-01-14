#!/usr/bin/env python3
""" mini batch"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ learning rate decay"""
    a = alpha / (1 + decay_rate * (global_step // decay_step))
    return a
