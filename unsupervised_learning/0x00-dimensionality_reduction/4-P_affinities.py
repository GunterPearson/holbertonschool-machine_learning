#!/usr/bin/env python3
""" dimensionality reduction"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """calculates the symmetric P affinities of a data set"""
    # (2500, 2500), (2500, 2500), (2500, 1), 4.906890595608519
    Dnorm, P, betas, H = P_init(X, perplexity)
    n, d = X.shape
    for i in range(n):
        copy = Dnorm[i]
        copy = np.delete(copy, i, axis=0)
        hi, pi = HP(copy, betas[i])
        betamin = None
        betamax = None
        Hdiff = hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                betamin = betas[i, 0]
                if betamax is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + betamax) / 2
            else:
                betamax = betas[i, 0]
                if betamin is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + betamin) / 2
            hi, pi = HP(copy, betas[i])
            Hdiff = hi - H
        aux = np.insert(pi, i, 0)
        P[i] = aux
    P = (P.T + P) / (2*n)
    return P
