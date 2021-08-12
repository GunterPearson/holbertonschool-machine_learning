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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model"""
    try:
        if iterations > 454:
            iterations = 454
        N, M = Emission.shape
        T = Observations.shape[0]
        a = Transition.copy()
        b = Emission.copy()
        for n in range(iterations):
            _, al = forward(Observations, b, a, Initial.reshape((-1, 1)))
            _, be = backward(Observations, b, a, Initial.reshape((-1, 1)))
            xi = np.zeros((N, N, T - 1))
            for col in range(T - 1):
                denominator = np.dot(np.dot(al[:, col].T, a) *
                                     b[:, Observations[col + 1]].T,
                                     be[:, col + 1])
                for row in range(N):
                    numerator = al[row, col] * a[row, :] * \
                                b[:, Observations[col + 1]].T * \
                                be[:, col + 1].T
                    xi[row, :, col] = numerator / denominator
            g = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(g, axis=1).reshape((-1, 1))
            g = np.hstack(
                (g, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
            denominator = np.sum(g, axis=1)
            for k in range(M):
                b[:, k] = np.sum(g[:, Observations == k], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)))
        return a, b
    except Exception as e:
        return None, None
