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
print(data_array)

# build posterior
log_likelihood = build_log_likelihood(data_array=data_array)
log_posterior = build_log_posterior(log_likelihood)

sd_proposal = 20
ICs = {'theta': 1}
mcmc_sampler = MHSampler(log_post=log_posterior, ICs=ICs)
mcmc_sampler.move = GaussianMove(ICs, cov=sd_proposal)

# print(log_prior(2))

mcmc_sampler.run(300)
mcmc_sampler.trace_plots(burnin=0)
mcmc_sampler.kde_plots(burnin=0)
print(f"Acceptance rate: {mcmc_sampler.backend.acceptance_rate:.2f}%")
plt.show()





# ============
# PRT: 2 plots
# empirical CDF and prior vs posterior samples
# ============
# array_samples = np.array([elem['posterior'] for elem in results])
# array_samples.sort()
# # plt.scatter(array_samples, np.arange(B)/B, s=4)
# # plt.plot(range(0, 11), np.arange(0, 1.1,0.1) , lw=1, c='r')

# fig, ax = plt.subplots(2, figsize=(7, 7))
# ax[0].scatter(array_samples, np.arange(B)/B, s=4)
# ax[0].plot(range(0, 11), np.arange(0, 1.1,0.1) , lw=1, c='r')
# ax[0].set_xlabel(r"$\theta$")
# ax[0].set_ylabel("empirical CDF")
# ax[0].set_title("Emprirical CDF", size=18)

# posterior_samples = [elem['posterior'] for elem in results]
# prior_samples = [elem['prior'] for elem in results]
# ax[1].scatter(posterior_samples, prior_samples, s=4)
# ax[1].set_xlabel("Posterior samples")
# ax[1].set_ylabel("Prior samples")
# ax[1].set_title("Posterior and prior samples", size=18)



# # True uniform samples
# # unisamples = np.random.uniform(0,10, size=300)
# # unisamples.sort()
# # plt.scatter(unisamples, list(range(300)), s=4)

plt.tight_layout()
plt.show()
