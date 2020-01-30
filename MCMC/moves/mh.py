# -*- coding: utf-8 -*-

import numpy as np

class MHMove:
    """
    Generic Metropolis Hastings move.
    Subclass or add a proposal function directly

    Parameters
    ----------

    get_proposal:
        It should take as input the parameters (current_samples) and return the proposed parameters and the log-hastings correction
    """

    def __init__(self, get_proposal):
        self.get_proposal = get_proposal

    def propose(self, current_samples, loss_current, log_posterior):
        """
        Propose parameter, do the accept/reject step, and return return the new samples, loss
        and whether the parameters were accepted or not.

        Parameters
        ----------
        current_samples: dict
            current samples

        loss_current: float

        log_posterior: function
            log-posterior

        Returns
        -------
        new_samples: dict
            Dictionary of samples (so either the newly accepted ones or previously kept ones)
        loss_new: float
            Loss corresponding to the samples returned
        accepted: Bool
            Whether or not the sample was accepted
        """
        new_samples, log_hastings = self.get_proposal(current_samples=current_samples)

        loss_new = log_posterior(**new_samples)

        alpha = loss_new - loss_current + log_hastings
        exp_sample = - np.random.exponential()
        # print("loss_new: {:.1f}".format(loss_new))
        # print("accept parameter alpha: {:.1f}\n".format(alpha))

        if alpha > exp_sample:
            return new_samples, loss_new, True
        else:
            return current_samples, loss_current, False
