#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from MCMC.samplers.mhsampler import MHSampler
from MCMC.moves.gaussian import GaussianMove
from PRT import run_PRT
from PRT import G, build_log_posterior, build_log_likelihood, sigma_data

# ============================
# Run MCMC

# generate 20 data points
data_array = G(5) + np.random.normal(loc=0, scale=sigma_data, size=10)
# print(data_array)

# build posterior
log_likelihood = build_log_likelihood(data_array=data_array)
log_posterior = build_log_posterior(log_likelihood)

sd_proposal = 20
ICs = {'theta': 1}
mcmc_sampler = MHSampler(log_post=log_posterior, ICs=ICs)
mcmc_sampler.move = GaussianMove(ICs, cov=sd_proposal)

# print(log_prior(2))

mcmc_sampler.run(2000)
mcmc_sampler.trace_plots(burnin=0)
mcmc_sampler.kde_plots(burnin=0)
print(f"Acceptance rate: {mcmc_sampler.backend.acceptance_rate:.2f}%")
plt.show()
