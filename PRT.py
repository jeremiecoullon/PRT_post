import matplotlib.pyplot as plt
import numpy as np
from MCMC.samplers.mhsampler import MHSampler
from MCMC.moves.gaussian import GaussianMove

# ============================
# Helper functions
# logpost:
# prior: uniform
# likelihood: linear model + gaussian noise
def G(theta):
	"G(theta): observation operator. Here it's just the identity function"
	return theta


# data noise:
sigma_data = 3

def build_log_likelihood(data_array):
	"Builds the log_likelihood function given some data"
	def log_likelihood(theta):
		"Data model: y = G(theta) + eps"
		return - (0.5)/(sigma_data**2) * np.sum([(elem - G(theta))**2 for elem in data_array])
	return log_likelihood

def log_prior(theta):
	# uniform prior on [0, 10]
	if not (0 < theta < 10):
		return -9999999
	else:
		return np.log(0.1)

def build_log_posterior(log_likelihood):
	def log_posterior(theta):
		return log_prior(theta) + log_likelihood(theta)
	return log_posterior


# ============
# PRT:
def run_PRT(num_data, B):
	results = []
	for elem in range(B):
		# sample from prior
		sam_prior = np.random.uniform(0,10)

		# generate data points
		data_array = G(sam_prior) + np.random.normal(loc=0, scale=sigma_data, size=num_data)

		# build posterior
		log_likelihood = build_log_likelihood(data_array=data_array)
		log_posterior = build_log_posterior(log_likelihood)

		# define sampler:
		ICs = {'theta': 1}
		sd_proposal = 20
		mcmc_sampler = MHSampler(log_post=log_posterior, ICs=ICs, verbose=0)
		mcmc_sampler.move = GaussianMove(ICs, cov=sd_proposal)

		# Get a posterior sample:
		mcmc_sampler.run(n_iter=200, print_rate=300)
		last_sample = mcmc_sampler.all_samples.iloc[-1].theta

		results.append({'posterior': last_sample, 'prior': sam_prior})
	return results
