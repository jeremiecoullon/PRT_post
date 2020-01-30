# -*- coding: utf-8 -*-

import unittest
import numpy as np
from MCMC.samplers.mhsampler import MHSampler
from MCMC.moves.gaussian import GaussianMove

class TestMHSampler(unittest.TestCase):

    def test_gaussian_sampler_no_cov(self):
        mu = 0
        sd = 3.8
        log_post = lambda x: -0.5*(1/sd**2)*(x-mu)**2
        ICs = {'x': 0}
        mcmc = MHSampler(log_post=log_post, ICs=ICs, verbose=0)
        mcmc.move = GaussianMove(mcmc.backend.param_info)
        n_iter = 100
        mcmc.run(n_iter=n_iter, print_rate=n_iter)
        assert mcmc.all_samples.shape == (n_iter+1, 2)

    def test_gaussian_sampler_with_cov_1float(self):
        mu = 0
        sd = 3.8
        log_post = lambda x: -0.5*(1/sd**2)*(x-mu)**2
        ICs = {'x': 0}
        cov = 1.3
        mcmc = MHSampler(log_post=log_post, ICs=ICs, cov=cov, verbose=0)
        mcmc.move = GaussianMove(mcmc.backend.param_info, cov=cov)
        n_iter = 100
        mcmc.run(n_iter=n_iter, print_rate=n_iter)
        assert mcmc.all_samples.shape == (n_iter+1, 2)

    def test_gaussian_sampler_with_cov_1int(self):
        mu = 0
        sd = 3.8
        log_post = lambda x: -0.5*(1/sd**2)*(x-mu)**2
        ICs = {'x': 0}
        cov = 1
        mcmc = MHSampler(log_post=log_post, ICs=ICs, cov=cov, verbose=0)
        mcmc.move = GaussianMove(mcmc.backend.param_info, cov=cov)
        n_iter = 100
        mcmc.run(n_iter=n_iter, print_rate=n_iter)
        assert mcmc.all_samples.shape == (n_iter+1, 2)

    def test_gaussian_sampler_with_cov_2dmi(self):
        mu = 0
        sd = 3.8
        log_post = lambda x, y: -0.5* np.linalg.multi_dot([[x,y],np.eye(2), [x,y]])
        ICs = {'x': 0, 'y': 4}
        cov = np.array([[2,-1], [-1, 2]])
        mcmc = MHSampler(log_post=log_post, ICs=ICs, cov=cov, verbose=0)
        mcmc.move = GaussianMove(mcmc.backend.param_info, cov=cov)
        n_iter = 100
        mcmc.run(n_iter=n_iter, print_rate=n_iter)
        assert mcmc.all_samples.shape == (n_iter+1, 3)
