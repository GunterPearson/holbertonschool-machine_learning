#!/usr/bin/env python3
"""randomly brighten image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """that randomly changes the brightness of an image"""
    return tf.image.random_brightness(image, max_delta)
