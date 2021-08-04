#!/usr/bin/env python3
""" clustering """
import numpy as np


def pdf(X, m, S):
    """calculates the probability density function"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    # formula
    # p(x∣ μ,Σ) = (1 / √(2π)d|Σ|)exp(−1/2(x−μ)T Σ−1(x−μ))
    n, d = X.shape
    # |Σ| = area(Σ) = det(Σ), |Σ| = S
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    # Formula one: (1 / √(2π)d|Σ|)
    front = 1 / np.sqrt((2 * np.pi) ** d * det)
    # Formula two: −1/2(x−μ)T Σ−1
    part1 = np.matmul((-(X - m) / 2), inv)
    # Formula three: formula 2 * (x−μ) used diagonal to fix alloc err
    part2 = np.matmul(part1, (X - m).T).diagonal()
    # Formula four: exp(−1/2(x−μ)T Σ−1(x−μ))
    expo = np.exp(part2)
    # final formula: (1 / √(2π)d|Σ|) * exp(−1/2(x−μ)T Σ−1(x−μ))
    pdf = expo * front
    P = np.where(pdf < 1e-300, 1e-300, pdf)
    return P
