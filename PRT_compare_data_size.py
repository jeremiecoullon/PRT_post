#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from PRT import run_PRT


"""
Run PRT and compare the effect of number of data points generated

B: number of iterations of the PRT
num_data: number of data points to generate. Try 2, 10, and 100
"""


B = 200
results_2 = run_PRT(num_data=1, B=B)
results_10 = run_PRT(num_data=10, B=B)
results_100 = run_PRT(num_data=100, B=B)


fig, ax = plt.subplots(3, figsize=(7,7))

posterior_samples_2 = [elem['posterior'] for elem in results_2]
prior_samples_2 = [elem['prior'] for elem in results_2]
ax[0].scatter(posterior_samples_2, prior_samples_2, s=4)
ax[0].set_xlabel("Posterior samples")
ax[0].set_ylabel("Prior samples")
ax[0].set_title("N=1", size=15)

posterior_samples_10 = [elem['posterior'] for elem in results_10]
prior_samples_10 = [elem['prior'] for elem in results_10]
ax[1].scatter(posterior_samples_10, prior_samples_10, s=4)
ax[1].set_xlabel("Posterior samples")
ax[1].set_ylabel("Prior samples")
ax[1].set_title("N=10", size=15)

posterior_samples_100 = [elem['posterior'] for elem in results_100]
prior_samples_100 = [elem['prior'] for elem in results_100]
ax[2].scatter(posterior_samples_100, prior_samples_100, s=4)
ax[2].set_xlabel("Posterior samples")
ax[2].set_ylabel("Prior samples")
ax[2].set_title("N=100", size=15)


plt.tight_layout()
plt.show()
