#!/usr/bin/env python3
"""reinforcement learning"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the next action"""
    explore_rate = np.random.uniform()
    if explore_rate > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(Q.shape[1])
    return action
