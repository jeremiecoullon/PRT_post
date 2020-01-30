# code of Prior Reproduction Test blog post

blog: http://jeremiecoullon.com/


Data is generated from a Gaussian with mean 5 and standard deviation 3. We assume the standard deviation is known and use MCMC to estimate the mean parameter.

This reproduces the figures in the blog post.


- `run_MCMC.py`: generates data and runs MCMC to estimate the mean

- `empirical_CDF.py`: Runs PRT to test the MCMC software and checks that the samples follow a uniform distribution (ie: the prior). Creates the figure with the empirical CDF

- `PRT_compare_data_size.py`: Run PRT for different dataset sizes to compare the effect on the test. Creates the figures with 3 plots of posterior vs prior samples.
