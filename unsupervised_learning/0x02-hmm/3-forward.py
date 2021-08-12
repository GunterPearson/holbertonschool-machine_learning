#!/usr/bin/env python3
"""hidden markov chain"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm"""
    try:
        T = Observation.shape[0]
        N = Transition.shape[0]

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for x in range(1, T):
            for n in range(N):
                tran = Transition[:, n]
                E = Emission[n, Observation[x]]
                F[n, x] = np.sum(tran * F[:, x - 1] * E)
        P = np.sum(F[:, -1])
        return P, F
    except Exception as e:
        None, None
