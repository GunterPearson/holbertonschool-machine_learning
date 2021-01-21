#!/usr/bin/env python3
""" regularize"""
import tensorflow as tf


def l2_reg_cost(cost):
    """ calculate cost for l2-reg"""
    return cost + tf.losses.get_regularization_losses()
