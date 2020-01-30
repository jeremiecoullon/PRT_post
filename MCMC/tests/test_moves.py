# -*- coding: utf-8 -*-

import unittest
import numpy as np

from MCMC.moves import gaussian

class TestMHMove(unittest.TestCase):

    def test_gaussian_cov_1(self):
        param_info = {'x': 2, 'y': -1}
        move = gaussian.GaussianMove(param_info)
        assert np.array_equal(move.cov, np.eye(len(param_info)))

    def test_gaussian_cov_2(self):
        param_info = {'x': 2, 'y': -1}
        x_rand = np.random.random(size=(2,2))
        cov = np.dot(x_rand.T, x_rand)
        move = gaussian.GaussianMove(param_info, cov)
        assert np.array_equal(move.cov, cov)

    def test_gaussian_move_cov_wrong_dim(self):
        """
        Instantiating a gaussian with a covariance shape that doesn't match the parameters
        should raise ValueError
        """
        ICs = {'x': 0}
        cov = np.array([[2,-1], [-1, 2]])
        self.assertRaises(ValueError, gaussian.GaussianMove, param_info=ICs, cov=cov)

    def test_get_proposal(self):
        param_info = {'x': 2, 'y': -1}
        x_rand = np.random.random(size=(2,2))
        cov = np.dot(x_rand.T, x_rand)
        move = gaussian.GaussianMove(param_info, cov)
        new_samples, log_hastings = move.get_proposal(current_samples={'x': 2, 'y': -1.4})
        assert log_hastings == 0
        assert type(new_samples) == dict
        assert len(new_samples) == 2
