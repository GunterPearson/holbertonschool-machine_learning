#!/usr/bin/env python3
"""reinforcement learning"""
import numpy as np


def play(env, Q, max_steps=100):
    """has the trained agent play an episode"""
    state = env.reset()
    done = False
    for x in range(max_steps):
        env.render()
        action = np.argmax(Q[state, :])
        new, reward, done, y = env.step(action)
        if done:
            env.render()
            break
        state = new
    env.close()
    return reward
