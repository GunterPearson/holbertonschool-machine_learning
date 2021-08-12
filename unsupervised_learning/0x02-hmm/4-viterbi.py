#!/usr/bin/env python3
"""hidden markov chain"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden states"""
    try:
        N, M = Emission.shape
        T = Observation.shape[0]
        viter = np.zeros((N, T))
        back = np.zeros((N, T))
        aux = (Initial * Emission[:, Observation[0]].reshape(-1, 1))
        viter[:, 0] = aux.reshape(-1)

        back[:, 0] = 0
        for t in range(1, T):
            for n in range(N):
                temp = viter[:, t - 1]
                trans = Transition[:, n]
                em = Emission[n, Observation[t]]
                result = temp * trans * em
                viter[n, t] = np.amax(result)
                back[n, t - 1] = np.argmax(result)
        path = []
        last_state = np.argmax(viter[:, T - 1])
        path.append(int(last_state))
        for i in range(T - 2, -1, -1):
            path.append(int(back[int(last_state), i]))
            last_state = back[int(last_state), i]
        path.reverse()
        least = np.amax(viter, axis=0)
        least = np.amin(least)

        return path, least
    except Exception as e:
        return None, None
