# -*- coding: utf-8 -*-

import numpy as np
from functools import wraps
import time

# from statsmodels.tsa.stattools import acf

# def int_ACT(x, M=50):
#     """
#     Integrated autocorrelation time: T_int = 1 + sum \rho_j
#     with \rho_j normalised autocorrelation function (uses statsmodels' function)

#     Parameters
#     ----------
#     x: ndarray
#         MCMC samples
#     M: int
#         Cuttoff to estimate ACF. According to Sokal, M should be smallest value such that  M >= C*T_int.
#         Choose C between 5 and 10.
#     """
#     # remove the first value of the ACF as this is 1
#     acf_arr = acf(x, nlags=M)[1:]
    # return 1 + 2*np.sum(acf_arr)


def time_it(fun):
    """
    Decorator that returns execution time of function
    if time > 60s: returns the time in minutes as well as seconds
    """
    @wraps(fun)
    def _wrapper(*args, **kwargs):
        start = time.time()
        result = fun(*args, **kwargs)
        end = time.time()
        time_in_sec = end-start
        time_in_min = np.floor((time_in_sec)/60).astype('int')
        num_sec = (time_in_sec) % 60
        if time_in_sec > 60:
            min_str = "({0} min {1} sec)".format(time_in_min, int(num_sec))
            time_in_sec = int(time_in_sec)
        else:
            min_str = ''
            time_in_sec = round(time_in_sec,3)
        print("Running time: {0} sec {1}".format(time_in_sec, min_str))
        return result
    return _wrapper

