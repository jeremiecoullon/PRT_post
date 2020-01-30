# -*- coding: utf-8 -*-
#
import numpy as np
import copy
import time

from MCMC.tools import util
from MCMC.backends.backend import Backend
from MCMC.moves.gaussian import GaussianMove
from MCMC.config import ConfigHolder

class MHSampler:
    """
    Metropolis-Hastings sampler

    Parameters
    ----------
    log_post: function
        log-posterior
    ICs: dict
        format: {'x': IC}
    cov: ndarray, int/float, or None
        Covariance matrix of proposal. If None: uses an identity matrix.
    verbose: int
        0: print nothing
        1: print basic info when running.
        2: print basic info when running along with acceptance rate
    """

    def __init__(self, log_post, ICs, cov=None, verbose=1, save_chain=False):
        self.verbose = verbose
        self.log_posterior = log_post
        self.backend = Backend(ICs)
        self.config = ConfigHolder(save_chain=save_chain)
        # whether or not to run the save_to_file() function at the end of the chain
        self.save_at_end = True
        # self.move = GaussianMove(self.backend.param_info, cov=cov)

    def stats(self):
        if self.verbose==2:
            print("\nAcceptance rate: {}".format(self.backend.acceptance_rate))
            print("\nMSEJD: {:.3f}".format(self.backend.MSEJD()))
        else:
            pass

    def progress_bar(self, iter_num, n_iter, print_rate, init=False):
        if init==True:
            if self.verbose>=1:
                print("Running MCMC for {} iterations...".format(n_iter))
            else:
                pass
        if self.verbose>=1:
            if iter_num >= print_rate:
                if iter_num%print_rate==0:
                    print("Iteration {0}/{1}".format(iter_num, n_iter))
        else:
            pass

    def step(self):
        """
        Run a step of MCMC

        1. Propose samples and accept/reject
        2. Save parameters to backend
        """
        # propose and accept/reject
        new_samples, loss_new, accepted = self.move.propose(current_samples=self.backend.current_samples,
            loss_current=self.backend.loss_current, log_posterior=self.log_posterior)

        # save step to backend
        self.backend.save_step(new_samples=new_samples, loss_new=loss_new, accepted=accepted)


    # @util.time_it
    def run(self, n_iter, print_rate=1000):
        """
        Parameters:
        -----------
        n_iter : int
            Number of samples to do
        print_rate : int
            Print the iteration number at every multiple of this integer.
        """
        #self.progress_bar(0, n_iter, print_rate, True)

        if len(self.backend.log_post_list)==0:
            # If the chain has never been run append self.loss_current to list_loss_current
            self.backend.loss_current = self.log_posterior(**self.backend.current_samples)
            self.backend.log_post_list.append(self.backend.loss_current)

        # run MCMC
        for iter_num in range(1, n_iter+1):
            self.step()
            self.progress_bar(iter_num, n_iter, print_rate)
            # save chain to hdf5 (by default this is off)
            if self.config.save_chain == True:
                self.save_to_file(iter_num=len(self.backend.log_post_list), iter_step=200)

        if self.config.save_chain == True:
            # don't save at the end: save every 100 iterations (above)
            # this is because it'll slow down PTSampler
            if self.save_at_end == True:
                self.save_to_file(iter_num=iter_num, iter_step=1)
            else:
                pass
        # some stats at the end of the sampling
        self.stats()

    def save_to_file(self, *args, **kwargs):
        "Override this in subclass to pass parameters to backend.to_file()"
        self.backend.to_file(*args, **kwargs)

    @property
    def all_samples(self):
        return self.backend.all_samples

    @property
    def acceptance_rate(self):
        return self.backend.acceptance_rate

    def trace_plots(self, *args, **kwargs):
        self.backend.trace_plots(*args, **kwargs)

    def kde_plots(self, *args, **kwargs):
        self.backend.kde_plots(*args, **kwargs)

    def acf(self, *args, **kwargs):
        self.backend.acf(*args, **kwargs)

    def pacf(self, *args, **kwargs):
        self.backend.pacf(*args, **kwargs)

    def MSEJD(self, *args, **kwargs):
        """ Return the Mean Square Euclidiean Jump Distance """
        return self.backend.MSEJD(*args, **kwargs)
