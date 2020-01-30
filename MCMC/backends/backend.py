# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
# import statsmodels.tsa.api as smt
from collections import OrderedDict


class BaseBackend:
    """
    Base Backend to store parameters

    Parameters
    ----------
    ICs: dict
        format: {'x': 0}
    """
    def __init__(self, ICs):
        self.param_info = OrderedDict([(k, {'sample_list': [v]}) for k,v in ICs.items()])

    @property
    def current_samples(self):
        return OrderedDict([(k, self.param_info[k]['sample_list'][-1]) for k,v in self.param_info.items()])

    @property
    def initial_conditions(self):
        return OrderedDict([(k, self.param_info[k]['sample_list'][0]) for k,v in self.param_info.items()])

    def append_sample(self, new_samples):
        for k,v in self.param_info.items():
            self.param_info[k]['sample_list'].append(new_samples[k])

    def chain_to_df(self):
        """
        Create a pandas DataFrame from the dictionaries of parameters
        """
        samples_dict = OrderedDict([("{}".format(param), self.param_info[param]['sample_list']) for param, v in self.param_info.items()])
        df_samples = pd.DataFrame(samples_dict)
        return df_samples

    def to_file(self):
        """
        Save chain to hdf5: implement in subclasses
        """
        pass

    @property
    def all_samples(self):
        "Return all samples and log_posterior in a DataFrame"
        return self.chain_to_df()



class Backend(BaseBackend):
    """
    Backend to store parameters in MCMC

    Parameters
    ----------
    ICs: dict
        format: {'x': 0}
    """
    def __init__(self, ICs):
        self.log_post_list = []
        # Initialise it with a dummy varianble
        self.loss_current = 1
        self.loss_new = 1
        self.new_samples = {}
        self.reset()
        super(Backend, self).__init__(ICs)

    def reset(self):
        "Reset iteration and acceptance counters to zero"
        self.counter_params_total = 0
        self.counter_params_accept = 0

    def save_step(self, new_samples, loss_new, accepted):
        """
        Save an MCMC step to the chain of parameters and log-posterior

        Parameters
        ----------
        new_samples: dict
            Dictionary of samples to append to chain
        loss_new: float
            Loss (log-posterior) to append to chain
        accepted: Bool
            Boolean: whether or not the sample was accepted.
        """
        # append samples
        for k,v in self.param_info.items():
            self.param_info[k]['sample_list'].append(new_samples[k])

        self.loss_current = loss_new

        if accepted == True:
            self.counter_params_accept += 1
        else:
            pass

        self.counter_params_total += 1
        self.log_post_list.append(self.loss_current)

    @property
    def acceptance_rate(self):
        "Acceptance rate (in percentage)"
        if self.counter_params_total > 0:
            return 100 * self.counter_params_accept / self.counter_params_total
        else:
            raise ValueError("No parameters have been sampled yet!")

    def best_samples(self):
        return self.all_samples.loc[self.all_samples.log_post == max(self.all_samples.log_post)]

    def chain_to_df(self):
        """
        Create a pandas DataFrame from the dictionaries of parameters
        """
        samples_dict = OrderedDict([("{}".format(param), self.param_info[param]['sample_list']) for param, v in self.param_info.items()])
        samples_dict['log_post'] = self.log_post_list
        df_samples = pd.DataFrame(samples_dict)
        return df_samples

    # Diagnostics stuff
    def trace_plots(self, burnin=0, log_post=True, step=1, figsize=(8,8)):
        x = np.arange(len(self.log_post_list[burnin::step]))
        pt_size = 5
        le_columns = [cols for cols in self.all_samples.columns]
        if log_post == False:
            le_columns.remove('log_post')
        number_of_subplots = len(le_columns)
        plt.figure(figsize=figsize)
        for i,col_name in enumerate(le_columns,1):
            ax1 = plt.subplot(number_of_subplots,1,i)
            ax1.scatter(x, self.all_samples[col_name][burnin::step], s=pt_size, alpha=0.4)
            ax1.set_title(col_name)
        plt.tight_layout()

    def kde_plots(self, burnin=0):
        if burnin > len(self.all_samples):
            raise ValueError("burnin must be less than number of samples")
        number_of_subplots = len(self.all_samples.columns)
        col_params = [elem for elem in self.all_samples.columns if elem!='log_post']
        plt.figure(figsize=(8,8))
        for i,col_name in enumerate(col_params,1):
            ax1 = plt.subplot(number_of_subplots,1,i)
            sns.kdeplot(self.all_samples.iloc[burnin:, :][col_name].values, ax=ax1)
            ax1.set_title(col_name)
        plt.tight_layout()

    # def acf(self, lags=100, burnin=0):
    #     # use the first parameter. They're all sampled in 1 step so the autocorrelations will be the same.
    #     param_column_name = filter(lambda x:x !='log_post',self.all_samples.columns.values)[0]
    #     fig = plt.figure(figsize=(10, 8))
    #     layout = (2, 2)
    #     acf_ax = plt.subplot2grid(layout, (1, 0))
    #     smt.graphics.plot_acf(self.all_samples[param_column_name].values[burnin:], lags=lags, ax=acf_ax)
    #     acf_ax.set_xlim(1.5)
    #     sns.despine()
    #     plt.tight_layout()

    # def pacf(self, lags=500, burnin=0):
    #     param_column_name = filter(lambda x:x !='log_post',self.all_samples.columns.values)[0]
    #     fig = plt.figure(figsize=(10, 8))
    #     layout = (2, 2)
    #     pacf_ax = plt.subplot2grid(layout, (1, 1))
    #     smt.graphics.plot_pacf(self.all_samples[param_column_name].values[burnin:], lags=lags, ax=pacf_ax)
    #     pacf_ax.set_xlim(1.5)
    #     sns.despine()
    #     plt.tight_layout()

    def MSEJD(self, burnin=0, params=None):
        """
        Return the Mean Square Euclidiean Jump Distance

        Parameters
        ----------
        burnin: int
            Burn in
        params: None or list
            If None (default) use all the parameters
            Else pass a list of parameters to use to calculate MSEJD
        """
        if params==None:
            df = self.all_samples.drop('log_post', axis=1)
        else:
            df = self.all_samples[params]
        df_diff = df.diff(axis=0).dropna(axis=0)
        return np.sum([np.square(elem).sum() for elem in df_diff.values]) * (1/(df_diff.shape[0]))
