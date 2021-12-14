#!/usr/bin/env python3
"""changes hue of image"""
import tensorflow as tf


def change_hue(image, delta):
    """that changes the hue of an image"""
    return tf.image.adjust_hue(image, delta)
