#!/usr/bin/env python3
""" clustering """
import scipy.cluster.hierarchy as sh
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """performs agglomerative clustering on a dataset"""
    linkage = sh.linkage(X, method='ward')
    clss = sh.fcluster(linkage, t=dist, criterion='distance')
    plt.figure()
    sh.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
