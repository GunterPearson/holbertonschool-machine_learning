#!/usr/bin/env python3
"""Pca Scatter plot"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmplasma = plt.get_cmap("plasma")

ax.scatter(pca_data[:, 0], pca_data[:, 1],
           pca_data[:, 2], label=labels, c=labels,
           cmap=cmplasma)

ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title('PCA of Iris Dataset')
plt.show()
