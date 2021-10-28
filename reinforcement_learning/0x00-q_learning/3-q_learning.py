#!/usr/bin/env python3
"""reinforcement learning"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs Q-learning"""
    all_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        current_reward = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _ = env.step(action)
            Q[state, action] = Q[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(Q[new_state, :]))
            state = new_state
            current_reward += reward
            if done:
                break
        epsilon = min_epsilon + (epsilon - min_epsilon) \
            * np.exp(-epsilon_decay * episode)
        all_rewards.append(current_reward)
    return Q, all_rewards
