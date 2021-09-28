#!/usr/bin/env python3
"""Positional Encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer"""
    def get_angles(pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dm))
        return pos * angle_rates

    position = np.arange(max_seq_len)
    pos_emb = get_angles(position[:, np.newaxis], np.arange(dm)[np.newaxis, :])
    pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])
    pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])
    return pos_emb
