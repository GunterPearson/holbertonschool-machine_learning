#!/usr/bin/env python3
"""reinforcement learning"""
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads a pre-made FrozenLakeEnv environment"""
    env = FrozenLakeEnv(desc, map_name, is_slippery)
    return env
