#!/usr/bin/env python3
"""train an agent that can play Atari’s Breakout"""
import argparse
T = __import__('train')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'play'], default='play')
args = parser.parse_args()


def play(mode="play"):
    """Plays or train"""
    weights_filename = 'policy.h5'
    model = T.model
    dqn = T.build_agent(model, T.nb_actions)
    if mode == "train":
        dqn = T.train_agent(dqn, T.env, weights_filename)
    else:
        dqn.load_weights(weights_filename)
        dqn.test(T.env, nb_episodes=10, visualize=True)


play(args.mode)
