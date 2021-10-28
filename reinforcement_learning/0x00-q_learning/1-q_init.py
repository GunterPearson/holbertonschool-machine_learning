#!/usr/bin/env python3
"""reinforcement learning"""
import numpy as np


def q_init(env):
    """Initializes the Q-table"""
    action = env.action_space.n
    state = env.observation_space.n
    qtable = np.zeros((state, action))
    return qtable
