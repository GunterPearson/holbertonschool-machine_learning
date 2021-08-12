#!/usr/bin/env python3
"""hidden markov chain"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    try:
        N, M = Emission.shape
        T = Observation.shape[0]
        B = np.zeros((N, T))
        B[:, T - 1] = np.ones(N)
        for j in range(T - 2, -1, -1):
            for i in range(N):
                aux = Emission[:, Observation[j + 1]] * Transition[i, :]
                B[i, j] = np.dot(B[:, j + 1], aux)
        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
        return P, B
    except Exception as e:
        return None, None
