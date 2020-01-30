 # -*- coding: utf-8 -*-

import numpy as np
from statsmodels.graphics.tsaplots import _prepare_data_corr_plot, _plot_corr
from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf

# from https://www.statsmodels.org/dev/_modules/statsmodels/graphics/tsaplots.html#plot_acf
# `lag_list` argument allows plotting of thinned samples

def my_plot_acf(x, lag_list, ax=None, lags=None, alpha=.05, use_vlines=True, unbiased=False,
             fft=False, title='Autocorrelation', zero=True,
             vlines_kwargs=None, **kwargs):
    fig, ax = utils.create_mpl_ax(ax)

    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs

    confint = None
    # acf has different return type based on alpha
    if alpha is None:
        acf_x = acf(x, nlags=nlags, alpha=alpha, fft=fft,
                    unbiased=unbiased)
    else:
        acf_x, confint = acf(x, nlags=nlags, alpha=alpha, fft=fft,
                             unbiased=unbiased)

    _plot_corr(ax, title, acf_x, confint, lag_list, irregular, use_vlines, vlines_kwargs=vlines_kwargs, **kwargs)

    return fig
