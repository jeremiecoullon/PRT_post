# -*- coding: utf-8 -*-

import numpy as np
from .mh import MHMove


class GaussianMove(MHMove):
    """

    Parameters
    ----------
    param_info: dict
        Dictionary of paramters and initial conditions. Format: {'x': 3, 'y': 10}
    cov: ndarray, int, or float
        Covariance matrix of RW proposal
    """

    def __init__(self, param_info, cov=None):
        self.len_data = len(param_info.keys())
        if cov is None:
            self.cov = np.eye(len(param_info.keys()))
        elif (isinstance(cov, float)) or (isinstance(cov, int)):
            self.cov = np.array([[cov]])
        elif self.len_data != len(cov):
            raise ValueError("Covariance matrix must be the same dimension as the parameters")
        else:
            self.cov = cov
        self.chol = np.linalg.cholesky(self.cov)

        super(GaussianMove, self).__init__(self.get_proposal)

    def get_proposal(self, current_samples):
        """
        Proposal function

        Parameters
        ----------
        current_samples: dict
            Dictionary of current samples

        Returns
        -------
        new_samples: dict
            Dictionary of updated samples using a Gaussian move
        log_hastings: float
            Value of log-hastings for the proposal
        """
        new_samples = {}
        new_sample_list = np.dot(self.chol, np.random.normal(size=self.len_data)) + list(current_samples.values())
        for idx, param in enumerate(list(current_samples.keys())):
            new_samples[param] = new_sample_list[idx]
        return new_samples, 0

    def __str__(self):
        move_description = "Random walk for all parameters using a Gaussian proposal"
        return move_description
