#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from PRT import run_PRT


"""
Run PRT and plot the empirical CDF 

B: number of iterations of the PRT
num_data: number of data points to generate
"""

B = 200
results_10 = run_PRT(num_data=10, B=B)
array_samples = np.array([elem['posterior'] for elem in results_10])
array_samples.sort()
# plt.scatter(array_samples, np.arange(B)/B, s=4)
# plt.plot(range(0, 11), np.arange(0, 1.1,0.1) , lw=1, c='r')


plt.scatter(array_samples, np.arange(B)/B, s=4)
plt.plot(range(0, 11), np.arange(0, 1.1,0.1) , lw=1, c='r')
plt.xlabel(r"$\theta$")
plt.ylabel("empirical CDF")
plt.title("Emprirical CDF", size=18)


plt.tight_layout()
plt.show()
