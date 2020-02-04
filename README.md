# code for Prior Reproduction Test blog post

*blog post: https://www.jeremiecoullon.com/2020/02/04/priorreproductiontest/*

Code to reproduce the figures in the PRT blog post. Includes a simple MCMC framework to test (in the `MCMC` module) using PRT.

## Installation

- `virtualenv venv; source venv/bin/activate`
- `pip install -r requirements.txt`
- `py.test`: run some unit tests


In `PRT.py`, we generated data from a Gaussian with mean 5 and standard deviation 3. We assume the standard deviation is known and use MCMC to estimate the mean parameter.


## Scripts:

- `run_MCMC.py`: generates data and runs MCMC to estimate the mean

- `empirical_CDF.py`: Runs PRT to test the MCMC software and checks that the samples follow a uniform distribution (ie: the prior). Creates the figure with the empirical CDF of the PRT samples.

- `PRT_compare_data_size.py`: Run PRT for different dataset sizes to compare the effect on the test. Creates the figure with 3 plots of posterior vs prior samples from PRT.
