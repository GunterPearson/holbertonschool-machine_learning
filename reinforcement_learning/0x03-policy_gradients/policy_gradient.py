#!/usr/bin/env python3
"""Policy gradients"""
import numpy as np


def policy(matrix, weight):
    """ function that computes to policy with a weight of a matrix

    Args:
        matrix (numpy)
        weight (numpy)
    """
    comb = matrix @ weight
    # compute softmax
    ex = np.exp(comb - np.max(comb))
    soft = ex / np.sum(ex)
    return soft


def policy_gradient(state, weight):
    """function that computes the Monte-Carlo policy gradient

    Args:
        matrix (numpy)
        weight (numpy)
    """
    state_mat = policy(state, weight)
    action = np.random.choice(len(state_mat[0]), p=state_mat[0])
    # compute gradient softmax
    softmax = state_mat.reshape(-1, 1)
    grad_soft = np.diagflat(softmax) - softmax @ softmax.T
    d_state_mat = grad_soft[action, :]
    d_log = d_state_mat / state_mat[0, action]
    grad = state.T @ d_log[None, :]
    return action, grad
