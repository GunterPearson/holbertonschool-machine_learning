#!/usr/bin/env python3
""" convolutions"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = int(np.floor(h - kh + 1))
    output_w = int(np.floor(w - kw + 1))
    output = np.zeros((m, output_h, output_w))
    for w in range(output_w):
        for h in range(output_h):
            output[:, h, w] = np.sum(
                kernel * images[:, h:h + kh, w:w + kw],
                axis=(1, 2)
            )
    return output
