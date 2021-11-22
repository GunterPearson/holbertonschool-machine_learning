#!/usr/bin/env python3
"""Policy Gradient module"""
import numpy as np
from policy_gradient import policy
from policy_gradient import policy_gradient


def single_episode(env, weight, episode, show_result):
    """play one episode"""
    state = env.reset()[None, :]
    return_grad = []

    while True:
        if show_result and (episode % 1000 == 0):
            env.render()
        action, grad = policy_gradient(state, weight)
        state, reward, done, _ = env.step(action)
        state = state[None, :]
        return_grad.append((state, action, reward, grad))
        if done:
            break
    env.close()
    return return_grad


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """a function that implements a full training"""
    weight = np.random.rand(4, 2)
    episodes = []

    for episode in range(nb_episodes):
        single = single_episode(env, weight, episode, show_result)
        T = len(single) - 1

        sum_rewards = 0
        for t in range(0, T):
            _, _, reward, grad = single[t]
            sum_rewards += reward
            G = np.sum([
                gamma**single[k][2] *
                single[k][2] for k in range(t + 1, T + 1)])
            weight += alpha * G * grad
        episodes.append(sum_rewards)
        print("{}: {}".format(episode, sum_rewards), end="\r", flush=False)
    return episodes
