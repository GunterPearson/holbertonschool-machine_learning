#!/usr/bin/env python3
"""hyperparameter tuning"""
import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization

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
