#!/usr/bin/env python3
"""hyperparameter tuning"""
import GPy
import GPyOpt
import numpy as np
from GPyOpt.methods import BayesianOptimization


def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)


X_init = np.array([[-0.9], [1.1]])
Y_init = f(X_init)
bounds = np.array([[-1.0, 2.0]])
noise = 0.2
kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
bds = [{'name': 'X', 'type': 'continuous', 'domain': bounds.ravel()}]

optimizer = BayesianOptimization(f=f,
                                 domain=bds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type='EI',
                                 acquisition_jitter=0.01,
                                 X=X_init,
                                 Y=-Y_init,
                                 noise_var=noise**2,
                                 exact_feval=False,
                                 normalize_Y=False,
                                 maximize=True)

optimizer.run_optimization(max_iter=10)
optimizer.plot_acquisition()
