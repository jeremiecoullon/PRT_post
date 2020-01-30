# -*- coding: utf-8 -*-

import unittest
import numpy as np

from MCMC.tools.util import scale_cov_directions

class TestTools(unittest.TestCase):

    def test_reconstruct_cov(self):
        """
        Scaling a covariance matrix with factor 1 should leave it unchanged.
        """
        rand_array = np.random.uniform(size=(4,4))
        cov = np.dot(rand_array, rand_array.T)
        new_cov = scale_cov_directions(cov=cov, alpha_list=[1,1,1,1])
        assert np.allclose(new_cov, cov)
        # now check that scaling it makes a difference
        new_cov = scale_cov_directions(cov=cov, alpha_list=[1.5,1,1,1])
        assert not np.allclose(new_cov, cov)
