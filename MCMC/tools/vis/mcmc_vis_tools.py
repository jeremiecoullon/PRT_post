 # -*- coding: utf-8 -*-

import numpy as np
import os.path
import os
import pandas as pd
import time
import glob
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import itertools
import statsmodels.tsa.api as smt
import h5py
from MCMC.tools import util
# commented out because jupyter notebook couldn't find the fortran src of del_Cast
# from . import vis_util
import boto3

plot_params = {'c':'royalblue', 'alpha':0.5, 's':30}



class MCMC_vis:
    """
    Class to visualise MCMC chains loaded from hdf5 files
    """

    def __init__(self, data_directory):
        """
        Parameters
        ----------
        data_directory : str
            Path to directory with hdf5 files (output of list_paths() )
        """
        self.d_samples = self.load_MCMC(data_directory=data_directory)
        self.set_chain_attributes()

    @property
    def num_chains(self):
        return len(list(self.d_samples.keys()))

    @property
    def num_samples(self):
        d_samples, le_count = self.keep_samples()
        return le_count

    @property
    def MAP(self, params=['z','rho_j','u','w']):
        """
        Returns a Series with the FD parameters and log posterior
        """
        df_all = self.concat_chains(params=params+['log_post'])
        return df_all.loc[df_all['log_post']==max(df_all['log_post'])].iloc[0]

    def load_MCMC(self, data_directory):

        """
        Load up MCMC chains from hdf5 in requested folder. Puts them in a dictionary of dataframess

        Parameters
        ----------
        data_directory : str
            Path to directory with hdf5 files (output of list_paths() )
        Returns
        -------
        d_samples : dict
            Dictionary with MCMC chains as dataframes
        """
        data_files = glob.glob(os.path.join(data_directory,"*.h5"))
        d_samples = {}
        self.attributes = {}

        for idx, data in enumerate(data_files,1):
            d_samples['MCMC_{}'.format(idx)] = pd.read_hdf(data)
            with h5py.File(data, 'r') as f:
                try:
                    attr_dict = {}
                    # numerical metadata is stored as datasets
                    numerical_attrs = {key: np.array(f['attributes'][key]) for key in f['attributes']}
                    attr_dict.update(numerical_attrs)
                    # string metadata is stored as attributes in the hdf5 file
                    for key in ['move', 'date', 'folder_and_run', 'data_array_dict', 'upload_to_S3', 'comments', 'step_save']:
                        if key in f.attrs.keys():
                            attr_dict.update({key: f.attrs[key]})
                        else:
                            pass
                    self.attributes['MCMC_{}'.format(idx)] = attr_dict
                except KeyError:
                    pass

        print("Number of MCMC chains: {}".format(len(d_samples)))
        if len(d_samples) == 0:
            print("There are no MCMC chains here...")
        if len(d_samples) > 0:
            num_samples = np.sum([len(d_samples[elem]['log_post']) for elem in d_samples])
            print("Total number of samples: {}".format(num_samples))
            dict_keys = [elem for elem in list(d_samples["MCMC_1"].keys()) if elem != 'log_post']
            print("Samples are taken from {}".format(dict_keys))

        print("\nData directory: {}".format(data_directory))
        return d_samples

    def set_chain_attributes(self):
        """
        Set all string attributes in hdf5 file as attributes to MCMC_vis
        """
        attr_list = [k for k in self.attributes['MCMC_1'].keys() if k not in ['cov', 'cov_joint', 'move']]
        for k in attr_list:
            setattr(self, k, self.attributes['MCMC_1'][k])

    def print_chain_info(self, attribute_list=['folder_and_run', 'data_array_dict', 'upload_to_S3', 'comments', 'step_save']):
        """
        Print information about the chain stored as attributes
        """
        for elem in attribute_list:
            if hasattr(self, elem):
                pre_string = ""
                if elem =='upload_to_S3':
                    pre_string = "Upload to S3: "
                elif elem == "comments":
                    pre_string = "Comments: "
                elif elem == 'step_save':
                    pre_string = "Step save: "
                print(pre_string + "{}".format(getattr(self, elem))+"\n------")
        print("\n")

    @property
    def move(self):
        print(self.attributes['MCMC_1']['move'])

    @property
    def cov(self):
        np.set_printoptions(suppress=True)
        return self.attributes['MCMC_1']['cov']

    def PT_accepts(self, chain_num):
        """Get number of PT accepts for chain `chain_num`"""
        return self.attributes['MCMC_{}'.format(chain_num)]['PT_accept_rejects'][0]

    def PT_rejects(self, chain_num):
        """Get number of PT reject for chain `chain_num`"""
        return self.attributes['MCMC_{}'.format(chain_num)]['PT_accept_rejects'][1]

    @property
    def cov_joint(self):
        np.set_printoptions(suppress=True)
        return self.attributes['MCMC_1']['cov_joint']

    def keep_samples(self, chains=None, log_post_lim=-999999, burnin=0):
        """
        Only keep samples from requested chains and if the log_posterior is above a threshold.
        Also remove the first sample of the chain which has log-post=1 (the dummy value to initialise MCMC)

        Parameters
        ----------
        d_samples : dict
            Dictionary of dataframes generated by load_MCMC()
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        log_post_lim : int
            lower limit for log_posterior. By default is -999999

        Returns
        -------
        d_samples : dict
            Dictionary of dataframes with only requested chains and above log_post_lim
        le_count : int
            Number of samples satisfying the constraints
        """
        if chains == None:
            chains = list(range(1,len(self.d_samples)+1))
        d_sam_keep = {k:v.loc[(v.log_post>log_post_lim) & (v.log_post<0) & (v.index>=burnin)] for k,v in self.d_samples.items() if int(k[5:]) in chains}
        le_count = []
        for k,v in d_sam_keep.items():
            le_count.append(len(v))
        return d_sam_keep, np.sum(le_count)


    def max_log_post(self):
        """
        Finds the parameters with the maximum value of log_post amongst all chains

        Parameters
        ----------
        d_samples : dict
            Dictionary of dataframes generated by load_MCMC()

        Returns
        -------
        df_min_log_post : df
            Dataframe with 1 row: this is the FD sample with maximum log-posterior
        """
        d_samples, le_count = self.keep_samples()
        df_min_log_post = pd.DataFrame()
        for k,v in d_samples.items():
            df_min_log_post=df_min_log_post.append(v.loc[v.log_post == max(v.log_post)])
        return df_min_log_post.loc[df_min_log_post.log_post == max(df_min_log_post.log_post)]

    def _calculate_acceptance(self, param_type, param_accept_list):
        """
        Calculates the acceptance ratio of a list/array of type ['FD_r', 'BC_a', 'BC_r', 'FD_a', 'global_a', 'global_r'] when using a RSGS

        Parameters
        ----------
        param_accept_list: list/ndarray
            Array in the format ['FD_r', 'BC_a', 'BC_r', 'FD_a'] that says which parameter you sampled and whether you accepted or rejected it
        param_type: str
            Either 'FD', 'BC', or 'global'

        Returns
        -------
            Ratio of accepted parameters to total number sampled (as a percentage)

        Note: only matches the first 2 characters is the string 'param_type' with the items in 'param_accept_list'
        """
        num_params = list(filter(lambda x: x if x[0:2]==param_type[0:2] else None, param_accept_list))
        accepted_params = list(filter(lambda x: x if x[-1]=='a' else None, num_params))
        # 2to3 output
        # num_params = [x for x in param_accept_list if x if x[0:2]==param_type else None]
        # accepted_params = [x for x in num_params if x if x[-1]=='a' else None]
        if len(num_params) > 0:
            return len(accepted_params)*100 / len(num_params)
        else:
            raise ValueError("No parameters have been sampled yet!")

    def RSGS_ratio(self, param_type="FD", chains=None, burnin=0):
        """
        Returns the ratio of sampled parameters (FD or BCs) to the total number of MCMC iterations
        """
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=-1e10)
        if 'param_accept' in self.d_samples['MCMC_1'].columns:
            samples_list = [le_dict['param_accept'][burnin:] for le_dict in list(d_samples.values())]
            param_accept_list = np.concatenate(samples_list,axis=0)
            param_accept_list = filter(lambda x: x if x != 'IC' else None, param_accept_list)
            num_params = len(filter(lambda x: x if x[0:2]==param_type else None, param_accept_list))
            # 2to3 output
            # param_accept_list = [x for x in param_accept_list if x if x != 'IC' else None]
            # num_params = len([x for x in param_accept_list if x if x[0:2]==param_type else None])
            print("Sampled {0} {1} parameters out of {2} iterations".format(num_params, param_type, len(param_accept_list)))
        else:
            print("These MCMC chains didn't use a random scan Gibbs sampler")

    def acceptance_rate(self, chains=None, burnin=0, param_type='FD', section_num=None):
        """
        Prints acceptance rate (as a ratio) for MCMC chains.
        If the chain used a random scan Gibbs sampler, you get the acceptance rate for FD or BC parameters
        separately.

        Parameters
        ----------
        d_samples : dict
            Dictionary of dataframes generated by load_MCMC()
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        burnin : int
            Le burn-in
        param_type: str
            'FD', 'BC', or 'global'. This only makes a difference if a random scan Gibbs sampler was used,
            or if a global move is used

        Returns
        -------
        None : doesn't return anythings; just prints the acceptance rate
        """
        if param_type == "PT_beta":
            if len(chains)!=1:
                raise ValueError("For the 'PT_beta', must calculate the acceptance rate one chain at a time")
            ac_rate = 100*self.PT_accepts(chains[0]) / (self.PT_accepts(chains[0]) + self.PT_rejects(chains[0]))
            print("Acceptance rate for PT_beta parameter: {:.3f}%".format(ac_rate))
        else:
            d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=-1e10)
            if 'param_accept' in self.d_samples['MCMC_1'].columns:
                samples_list = [le_dict['param_accept'][burnin:] for le_dict in list(d_samples.values())]
                param_accept_list = np.concatenate(samples_list,axis=0)
                accept_rate = self._calculate_acceptance(param_type=param_type, param_accept_list=param_accept_list)
                print("Acceptance rate for {0} parameter: {1:.3f}%".format(param_type, accept_rate))
            else:
                samples_list = [le_dict['log_post'][burnin:] for le_dict in list(d_samples.values())]
                param_chain = np.concatenate(samples_list,axis=0)
                sample_diff = np.diff(param_chain)
                accept_rate = 100 * np.count_nonzero(sample_diff)/len(sample_diff)
                print("Acceptance rate: {0:.3f}%".format(accept_rate))

    def accept_Gibbs(self, chain_num, section_num):
        """
        Calculate acceptance rate for each section in BC Gibbs sampling

        Parameters
        ----------

        chain_num: int
            Chain number to calculate acceptance rate for
        section_num: int
            Section number to calculate acceptance rate for
        """
        le_section = "section_{}".format(section_num)# .encode('utf-8') # .encode('utf-8') works only with outputs from python2
        df = self.d_samples["MCMC_{}".format(chain_num)]
        if le_section not in df.BC_Gibbs.unique():
            raise ValueError("{} wasn't sampled".format(le_section))
        param_accept_list = list(df.loc[df.BC_Gibbs==le_section].param_accept)
        return self._calculate_acceptance(param_type="BC", param_accept_list=param_accept_list)


    def trace_plot(self, params=[], burnin=0, step=1, chains=None, log_post_lim=-9999999, pt_size=10, title_save=None, figsize=(10,8), ylim_dict={}):
        """
        Plots trace plots for MCMC chain

        Parameters
        ----------
        param : list
            List of parameters to do trace plots for
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        burnin : int
            Le burn-in
        step: int
            Step size to thin samples
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        log_post_lim : int
            lower limit for log_posterior. By default is -999999
        title_save: None (default) or str
            If not None: save figure with given name
        figsize: tuple
            Figure size
        ylim_dict: dict
            Dictionary of parameter with bounds on the y axis. Used to manually
            bound the y axis for a given parameters. Default is the empty dictionary {}
            Format: {'beta': (0.0224, 0.0229)}

        Returns
        -------
        fig : matplotlib figure object
        """
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=log_post_lim)
        colors = cm.rainbow(np.linspace(0, 1, len(d_samples)))
        if params == []:
            params = [elem for elem in self.d_samples['MCMC_1'].columns if elem not in ['BC_inlet', 'BC_outlet', 'param_accept', 'BC_Gibbs']]
        else:
            pass
        number_of_subplots = len(params)
        f, ax = plt.subplots(number_of_subplots, sharex=True, sharey=False, figsize=figsize)
        if number_of_subplots==1:
            ax = [ax]
        for idx, (k,df_val) in enumerate(d_samples.items()):
            le_dict = df_val[params].loc[df_val.index > burnin].iloc[::step]
            x = le_dict.index.values
            for le_ax, (i,col_name) in zip(ax, enumerate(params,1)):
                # le_ax.scatter(x, le_dict[col_name], s=pt_size, alpha=0.6, label='chain {}'.format(k[-1]))
                le_ax.plot(x, le_dict[col_name], lw=pt_size, alpha=0.6, label='chain {}'.format(k[-1]))
                le_ax.set_title(col_name)
                # le_ax.legend()
                if col_name in ylim_dict.keys():
                    le_ax.set_ylim(ylim_dict[col_name])
            plt.tight_layout()
        ax[-1].set_xlabel("Iteration", size=30)
        f.set_size_inches(10,10)
        if title_save is not None:
            plt.savefig(title_save)
        # plt.show()
        return f,ax

    def BC_trace_plots(self, BC_type, cell_num_list, step=1, chains=None, burnin=0, figsize=None, title_size=20):
        """
        Plot trace plots of BC times

        Parameters
        ----------
        BC_type: str
            Either 'BC_outlet' or 'BC_inlet'
        cell_num_list: list
            List of BC times to do trace plots for
        step: int
            Step to use to thin samples
        chains: list or None
            List of chains to plot trace_plots for. If None: use all chains.
        figsize: None or tuple
            If none, use (10,len(cell_num_list*2))
        title_size: int
            Size for titles
        """
        start = time.time()
        if figsize is None:
            figsize = (16, len(cell_num_list)*2)
        else:
            pass
        fig, ax = plt.subplots(len(cell_num_list), sharex=True, figsize=figsize)
        if chains is None:
            key_list = self.d_samples.keys()
        elif type(chains)==list:
            key_list = ['MCMC_{}'.format(elem) for elem in chains]
        else:
            raise ValueError("'chains' must be a list or 'None'")
        for cell_num_idx, cell_num in enumerate(cell_num_list):
            for MCMC_key in key_list:
                le_dict = self.d_samples[MCMC_key].loc[self.d_samples[MCMC_key].index > burnin].iloc[::step]
                BC_outlet_samples = np.array([elem[cell_num] for elem in le_dict[BC_type]])
                ax[cell_num_idx].plot(le_dict.index.values, BC_outlet_samples, linewidth=1)
            ax[cell_num_idx].set_ylabel('Density', size=22)
            ax[cell_num_idx].set_title("{} BC chains at time {} ({} min)".format(len(key_list), cell_num, int(cell_num/40)), size=title_size)
        ax[0].set_ylabel('Density (veh/km)', size=22)
        ax[-1].set_xlabel('Iteration', size=28)
        plt.tight_layout()

        # plt.plot()

        end = time.time()
        print("Running time: {:.2f}s".format(end-start))

    def R_hat(self, param, burnin=0):
        """
        R_hat for a FD parameter.
        Cut all chains to the same length as the shortest run

        Parameters
        ----------
        param: str
            FD parameter
        burnin: int
            burnin (default is 0)
        """
        # get index of shortest run
        shortest_run_idx = min([max(v.index) for k,v in self.d_samples.items()])
        # get list of arrays
        list_sams = [v.loc[(v.index>=burnin) & (v.index<=shortest_run_idx)][param].values for k,v in self.d_samples.items()]
        # variance of all chains mixed together
        mixed_var = np.var(np.concatenate(list_sams))
        mean_vars = np.mean([np.var(elem) for elem in list_sams])
        R_hat = np.sqrt(mixed_var/mean_vars)
        return R_hat

    def BC_R_hat(self, BC_type, BC_t, burnin=0):
        """
        R_hat statistic: mixed_var / mean_vars
        Cut all chains to the same length as the shortest run

        Parameters
        ----------
        BC_type: str
            "BC_outlet" or "BC_inlet"
        BC_t: int
            BC time point to get R_hat for
        burnin: int
            burnin
        """

        shortest_run_idx = min([max(v.index) for k,v in self.d_samples.items()])
        # get list of arrays
        list_sams = [v.loc[(v.index>=burnin) & (v.index<=shortest_run_idx)][BC_type].values for k,v in self.d_samples.items()]

        # mixed variance
        all_BCs = np.concatenate(list_sams)
        sams_timet = np.array([elem[BC_t] for elem in all_BCs])
        mixed_var = np.var(sams_timet)

        # variance for each chain
        list_vars = []
        for elem in list_sams:
            list_vars.append(np.var(np.array([el[BC_t] for el in elem])))

        mean_vars = np.mean(list_vars)
        R_hat = np.sqrt(mixed_var/mean_vars)
        return R_hat

    def plot_BC_R_hat(self, BC_type, t_list=None, burnin=0, figsize=(14, 8)):
        """
        Plot R_hat for 39 BC time points

        Parameters
        ----------
        BC_type: str
            "BC_outlet" or "BC_inlet"
        burnin: int
            burnin
        figsize: tuple
            Default is (14, 8)
        """
        N_BCs = len(self.d_samples['MCMC_1'].iloc[0].BC_outlet)
        R_hat_list = []
        x_range = np.arange(0, N_BCs, 39)
        for BC_t in x_range:
            R_hat_list.append(self.BC_R_hat(BC_type=BC_type, BC_t=BC_t, burnin=burnin))

        plt.figure(figsize=figsize)

        if t_list is None:
            plt.plot(x_range, R_hat_list, marker="+", lw=0.8)
        else:
            plt.plot(t_list, R_hat_list, marker="+", lw=0.8)
        plt.axhline(1.1, c='r', alpha=0.5, label="Appropriate limit")
        # plt.title("R_hat diagnostic for {}".format(BC_type), size=23)
        plt.xlabel("Time (min)", size=30)
        plt.ylabel("R.hat", size=30)
        plt.legend(prop={'size': 23})

    def BC_kdeplots(self, BC_type, cell_num_list, burnin=0, step=1):
        """
        Plot kdeplots of BC times: concatenates all chains
        and plots the kdeplot for each requested time

        Parameters
        ----------
        BC_type: str
            Either 'BC_outlet' or 'BC_inlet'
        cell_num_list: list
            List of BC times to do trace plots for
        burnin: int
            Burnin to apply
        step: int
            Step size to thin samples by
        """
        start = time.time()

        fig, ax = plt.subplots(len(cell_num_list), sharex=True, figsize=(10,len(cell_num_list)*2))

        for cell_num_idx, cell_num in enumerate(cell_num_list):
            all_BCs = self.concat_chains(params=[BC_type], burnin=burnin).values[::step]
            BC_outlet_samples = np.array([elem[0][cell_num] for elem in all_BCs])
            sns.kdeplot(BC_outlet_samples, ax=ax[cell_num_idx])
            ax[cell_num_idx].set_ylabel('density')
            ax[cell_num_idx].set_title("{} BC chains at time {}".format(len(self.d_samples), cell_num), size=12)

        plt.xlabel('density', size=19)
        plt.legend()
        # plt.plot()

        end = time.time()
        print("Running time: {:.2f}s".format(end-start))

    def kdeplot(self, params=[], chains=None, burnin=0, figsize=(10,10), title_save=None):
        """

        Parameters
        ----------
        params : str
            Name of parameter from MCMC to plot kdeplot for
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        burnin : int
            Le burn-in
        title_save: None (default) or str
            If not None: save figure with given name

        Returns
        -------
        f, ax : matplotlib figure and axes
        """
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=-9999999)
        if params == []:
            params = [elem for elem in self.d_samples['MCMC_1'].columns if elem not in ['BC_inlet', 'BC_outlet', 'param_accept', 'BC_Gibbs']]
        else: pass
        number_of_subplots = len(params)
        f, ax = plt.subplots(number_of_subplots, sharex=False, sharey=False, figsize=figsize)
        if number_of_subplots==1:
            ax = [ax]
        # Loop for each parameter in param
        for le_ax, par in zip(ax, params):
            samples_list = [le_dict[par][burnin:] for le_dict in list(d_samples.values())[:]]
            param_chain = np.concatenate(samples_list,axis=0)
            if chains==None:
                chains = list(range(1,len(d_samples)+1))
            sns.kdeplot(param_chain, ax=le_ax)
            le_ax.set_title("{0} samples from chains: {1}".format(par, chains),size = 16)
            le_ax.set_xlabel(par,size = 17)
            plt.tight_layout()
        if title_save is not None:
            plt.savefig(title_save)
        # return f,ax

    def acf(self, chain, lags=300, burnin=0, title_save=None):
        """
        Plot acf of requested chain

        Parameters
        ----------
        chain: int
            Chain number
        lags: int (optional)
            Number of lags to calculate acf for. Defaults to 300
        burnin: int (optional)
            Burnin after which to calculate acf. Defaults to 0
        title_save: None (default) or str
            If not None: save figure with given name
        """
        df = self.d_samples['MCMC_{}'.format(chain)]
        # get any FD parameter
        param_name = [elem for elem in df.columns.values if elem not in ['log_post', 'BC_inlet', 'BC_outlet', 'param_accept']][0]
        fig = plt.figure(figsize=(10, 8))
        layout = (2, 2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        smt.graphics.plot_acf(df[param_name].values[burnin:], lags=lags, ax=acf_ax)
        acf_ax.set_xlim(1.5)
        sns.despine()
        plt.tight_layout()
        if title_save is not None:
            plt.savefig(title_save)
        plt.show()

    def pacf(self, chain, lags=300, burnin=0, title_save=None):
        """
        Plot pacf of requested chain

        Parameters
        ----------
        chain: int
            Chain number
        lags: int (optional)
            Number of lags to calculate pacf for. Defaults to 300
        burnin: int (optional)
            Burnin after which to calculate pacf. Defaults to 0
        title_save: None (default) or str
            If not None: save figure with given name
        """
        df = self.d_samples['MCMC_{}'.format(chain)]
        # get any FD parameter
        param_name = [elem for elem in df.columns.values if elem not in ['log_post', 'BC_inlet', 'BC_outlet', 'param_accept']][0]
        fig = plt.figure(figsize=(10, 8))
        layout = (2, 2)
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        smt.graphics.plot_pacf(df[param_name].values[burnin:], lags=lags, ax=pacf_ax)
        pacf_ax.set_xlim(1.5)
        sns.despine()
        plt.tight_layout()
        if title_save is not None:
            plt.savefig(title_save)
        plt.show()


    def fit_cov(self, params=['z', 'rho_j', 'u', 'w'], burnin=0, chains=None, log_post_lim=-999999):
        """
        Fit covariance matrix on MCMC chains for chosen parameters to use in proposal matrix
        Parameters
        ----------
        params : list
            List of FD parameters to fit covariance matrix
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        log_post_lim : int
            lower limit for log_posterior. By default is -999999

        Returns
        -------
        cov : ndarray
            Covariance matrix
        """
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=log_post_lim)
        d_df = {k: v[params][burnin::] for k,v in d_samples.items()}
        df_all_chains = pd.concat(list(d_df.values()))
        return np.cov(np.transpose(df_all_chains.values))

    def concat_chains(self, chains=None, burnin=0, params=['z', 'rho_j', 'u', 'w']):
        """
        Returns a dataframe of all chains concatenated together (discarding burnin)

        Parameters
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        burnin: int
            Burnin to discard for each chain
        """
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=-999999999)
        d_df = {k: v[params].loc[v.index>=burnin] for k,v in d_samples.items()}
        df_all_chains = pd.concat(list(d_df.values()))
        return df_all_chains

    def plot_FD_jumps(self, chains=None, burnin=0, move_type='global', step_size=40):
        """
        Plot the FD for jumps.

        Parameters
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        burnin: int
            Burnin to discard for each chain
        move_type: str
            Either global or FD
        step_size: int
            Step to jump when plotting moves
        """
        df = self.concat_chains(chains=chains, burnin=burnin, params=['z', 'rho_j', 'u', 'w', 'param_accept'])

        # get list of indexes where there is a global move
        global_list = list(df.loc[df.param_accept=='{}_a'.format(move_type)].index)
        print("Number of {} moves: {}".format(move_type, len(global_list)))
        print("plotting {} FD jumps".format(int(len(global_list)/step_size)+1))

        x_range = np.arange(0,500, 0.01)

        for idx_num in range(1, len(global_list), step_size):
            df.iloc[global_list[idx_num]]

            FD_1 = {k:v for k,v in df.iloc[global_list[idx_num]-1].to_dict().items() if k in ['z', 'rho_j', 'u', 'w']}
            FD_1['w'] = 1/FD_1.pop('w')

            FD_2 = {k:v for k,v in df.iloc[global_list[idx_num]].to_dict().items() if k in ['z', 'rho_j', 'u', 'w']}
            FD_2['w'] = 1/FD_2.pop('w')

            plt.plot(x_range, util.FD_neg_power(rho=x_range, **FD_1), label="previous FD")
            plt.plot(x_range, util.FD_neg_power(rho=x_range, **FD_2), label="accepted FD")
            plt.legend()
            plt.xlabel("Density")
            plt.ylabel("Flow")
            plt.title("FD jumps during (accepted) {2} move {0} (ie: iter {1})".format(idx_num+1, global_list[idx_num], move_type), size=18)
            plt.show()

    def plot_cov_diagnostic(self, cov, num_props, thin, FD_loc, end=6, burnin=0, chains=None):
        """
        Plot 6 figures for each combination of FD parameters
        Plot a single FD along with proposals based on a covariance matrix

        Parameters
        ---------
        cov: ndarray
            Covariance matrix used to generate samples from
        num_props: int
            Number of proposal to include
        thin: int
            Amount to thin dataframe with joint samples by
        FD_loc: int
            Index number of FD to center the proposals around
        burnin,
        chains=[1,2,3,4...]
        """
        df_all = self.concat_chains(chains=chains, burnin=burnin)
        FD_point_d = df_all[['z','rho_j','u','w']].iloc[FD_loc].to_dict()
        df_all = df_all[::thin]
        FD_proposal = np.random.multivariate_normal(mean=list(FD_point_d.values()), cov=cov, size=num_props)
        ax_idx_combs = itertools.product([0,1,2], [0,1])
        FD_combs = itertools.combinations(['z','rho_j','u','w'], 2)

        fig, ax = plt.subplots(3,2, figsize=(18,13))
        for ax_idx, (x,y) in zip(list(ax_idx_combs)[:end], list(FD_combs)[:end]):
            ax[ax_idx].scatter(FD_point_d[x], FD_point_d[y], c='r', s=140)
            vis_util.plot_FD_proposals(x=x, y=y, ax=ax[ax_idx], FD_proposal=FD_proposal)
            vis_util.plot_2_FD_params(x=x, y=y, ax=ax[ax_idx], df=df_all)

        plt.tight_layout()


    def plot_FD_jump_scatter(self, move_list, move_type='global', chains=None, burnin=0):
        """
        Plot FD accepted jumps in scatter plots

        Parameters
        ---------
        move_list: list
            List of indexes (based off the number of moves) to plot FD jumps
        move_type: str
            Either 'global' or 'FD'
        chains: None or list
            List of chain numbers
        burnin: int
            Burnin to discard for each chain
        """
        df = self.concat_chains(chains=chains, burnin=burnin, params=['z', 'rho_j', 'u', 'w', 'param_accept'])
        # get list of indexes where there is a global move
        global_list = list(df.loc[df.param_accept=='{}_a'.format(move_type)].index)
        print("Number of {} moves: {}".format(move_type, len(global_list)))

        ax_idx_combs = itertools.product([0,1,2], [0,1])
        FD_combs = itertools.combinations(['z','rho_j','u','w'], 2)
        end = 6
        alpha = 0.6
        fig, ax = plt.subplots(3,2, figsize=(18,13))
        for ax_idx, (x,y) in zip(list(ax_idx_combs)[:end], list(FD_combs)[:end]):
            for move_num in move_list:
                FD_1, FD_2 = vis_util.get_FD_jump(self=self, chains=chains,  move_num=move_num, move_type=move_type)
                ax[ax_idx].scatter(FD_1[x], FD_1[y], c='r', s=70, alpha=alpha)
                ax[ax_idx].scatter(FD_2[x], FD_2[y], c='r', s=70, alpha=alpha)
                ax[ax_idx].plot([FD_1[x], FD_2[x]], [FD_1[y], FD_2[y]], c='r', linewidth=2, alpha=alpha)
            vis_util.plot_2_FD_params(x=x, y=y, ax=ax[ax_idx], df=df)

        plt.tight_layout()
        plt.legend()
        plt.show()


    def plot_sample_FD(self, df_data, step=20, burnin=0, w_transf_type='inv', chains=None, log_post_lim=-99999999, title_save=None):
        """
        Plot sampled FDs from MCMC chains
        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame of Midas data to plot alongside FDs. columns = ['density', 'flow']
        step : int
            Step size to keep samples from MCMC chains
        burnin : int
            Where to start plotting samples
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        log_post_lim : int
            lower limit for log_posterior. By default is -999999
        title_save: None (default) or str
            If not None: save figure with given name

        Returns
        -------
        fig : matplotb figure
            Le figure
        """
        # 1) keep only rows in dataframes with log_post above limit, and 2) keep only requested chain numbers
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=log_post_lim)
        fig, ax = plt.subplots(figsize=(12,8))
        x_range = np.arange(0,600,1)

        print("Number of FDs under the selected value of log posterior: {}".format(le_count))


        df_concat = pd.concat([v.iloc[burnin::step] for k,v in d_samples.items()])
        print("Number of plotted FDs: {}".format(len(df_concat)))
        # df_concat = df_concat
        if 'w' in df_concat.columns:
            rho_j_max = max(df_concat.rho_j)
            if w_transf_type == 'log_inv':
                df_concat.w = 1/np.exp(df_concat.w.values)
            elif w_transf_type == 'nat':
                pass
            elif w_transf_type == 'inv':
                df_concat.w = 1/df_concat.w.values
            else:
                raise ValueError("w_transf_type must be either inv, 'nat', or 'log_inv'")
        for idx,row in df_concat.iterrows():
            if (row.log_post > log_post_lim):
                if 'z' in df_concat.columns:
                    # Del Castillo
                    x_range = np.arange(0, rho_j_max,1)
                    le_FD = util.FD_neg_power(rho=x_range, z=row.z, rho_j=row.rho_j, w=row.w, u=row.u)
                    ax.plot(x_range, le_FD , linewidth=1, alpha=0.5)#, label="Z={0:.2f}, rho_j={1:.2f}, w={2:.2f}".format(row.z, row.rho_j, row.w))
                elif 'alpha' in df_concat.columns:
                    le_FD = util.FD_exp(rho=x_range, alpha=row.alpha, beta=row.beta)
                    ax.plot(x_range, le_FD , linewidth=1, alpha=0.5)#, label="alpha={0:.2f}, beta={1:.2f}".format(row.alpha, row.beta))

        ax.scatter(df_data.density, df_data.flow, label='M25 data', **plot_params)

        # ax.set_title("Del Castillo FD samples", size = 18)
        ax.set_xlabel('Density (veh/km)', size=30)
        ax.set_ylabel('Flow (veh/min)', size=30)
        if title_save is not None:
            plt.savefig(title_save)
        # plt.show()
        return fig,ax

    def plot_sample_BC(self, chain, BC_type, step=100, burnin=0, title_save=None):
        """
        Plot sampled BCs from selected MCMC chain

        Parameters
        ----------
        chains : int
            Chain number
        BC_type : str
            Either 'BC_inlet' or 'BC_outlet'
        step : int
            Step size to keep samples from MCMC chains
        burnin : int
            le burnin
        title_save: None (default) or str
            If not None: save figure with given name

        """
        df = self.d_samples["MCMC_{}".format(chain)]
        if BC_type not in df.columns:
            raise ValueError("That boundary condition wasn't sampled!")
        print("Number of plotted BCs: {}".format(int((len(df)-burnin)/step)))
        for idx, row in df.iterrows():
            if idx%step==0:
                if idx>=burnin:
                    plt.plot(row[BC_type], label=idx)
                else:
                    pass
        plt.legend()
        plt.title("Sampled {} BC ".format(BC_type[3:]), size=18)
        if title_save is not None:
            plt.savefig(title_save)
        plt.show()


    def _MAP_to_dict(self, w_transf_type='inv'):
        """
        Do some processing on MAP parameters and return them as a dictionary
        """
        FD = self.max_log_post().iloc[0].to_dict()
        FD.pop('log_post')
        if 'z' in FD:
            FD['solver'] = 'lwr_del_Cast'
            if w_transf_type == 'log_inv':
                FD['w'] = 1/np.exp(FD.pop('w'))
            elif w_transf_type == 'nat':
                pass
            elif w_transf_type == 'inv':
                FD['w'] = 1/FD.pop('w')
            else:
                raise ValueError("w_transf_type must be either 'nat' or 'log_inv'")
        elif 'alpha' in FD:
            FD['solver'] = 'lwr_exp'
        # remove 'param_accept' if it exists
        FD.pop('param_accept', None)
        return FD

    def plot_BC_mean_std(self, chain, BC_type, burnin=0):
        """
        Plot pointwise boundary condition mean along with pointwise standard deviation

        Parameters
        ----------
        chains : int
            Chain number
        BC_type: str
            Either 'BC_outlet' or 'BC_inlet'
        burnin : int
            Le burn-in
        """
        df = self.d_samples["MCMC_{}".format(chain)].iloc[burnin:, :]
        if BC_type not in df.columns:
            raise ValueError("That boundary condition wasn't sampled!")
        vis_util.function_mean_sd(fun_series=df[BC_type])

    def plot_BC_MAP(self, BC):
        """
        Plot FD for MAP parameters

        Parameters
        ----------
        BC: str
            Either 'BC_outlet' or 'BC_inlet'
        """
        FD = self._MAP_to_dict()
        if BC not in FD:
            raise ValueError("That boundary condition wasn't sampled!")
        plt.plot(FD[BC])
        plt.title("MAP {} boundary condition".format(BC[3:]), size=18)
        plt.xlabel("time")
        plt.ylabel("density")

    def plot_FD_MAP(self, df_data, title_save=None, w_transf_type='inv'):
        """
        Plot FD for MAP parameters

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame of Midas data to plot alongside FDs
        title_save: None (default) or str
            If not None: save figure with given name
        """
        FD = self._MAP_to_dict(w_transf_type=w_transf_type)
        x_range = np.arange(0,600,1)

        if FD.get('solver')=='lwr_del_Cast':
            le_FD = util.FD_neg_power(rho=x_range, z=FD.get('z'), rho_j=FD.get('rho_j'), w=FD.get('w'), u=FD.get('u'))
            plt.plot(x_range, le_FD , linewidth=2) #, label="Z={0:.2f}, rho_j={1:.2f}, w={2:.2f}, u={3:.2f},".format(FD.get('Z'), FD.get('rho_j'), FD.get('w'), FD.get('u')))
            plt.legend()
        elif FD.get('solver')=='lwr_exp':
            le_FD = util.FD_exp(rho=x_range, alpha=FD.get('alpha'), beta=FD.get('beta'))
            plt.plot(x_range, le_FD , linewidth=2, label="alpha={0:.2f}, beta={1:.2f}".format(FD.get('alpha'), FD.get('beta')))
            plt.legend()
        plt.scatter(df_data.density, df_data.flow, c ='r', alpha=0.6, s=2)
        plt.title("MAP {} FD".format(FD.get('solver')[4:]), size = 18)
        plt.xlabel('density')
        plt.ylabel('flow')
        if title_save is not None:
            plt.savefig(title_save)
        plt.show()

    def plot_LWR_MAP(self, data_variable, title_save=None, w_transf_type='inv'):
        """
        Solve LWR with MAP parameters and plot in the x-t plane

        Parameters
        ----------
        data_variable: str
            Variable to use to plot output: either 'flow' or 'density'
        title_save: None (default) or str
            If not None: save figure with given name
        """
        FD = self._MAP_to_dict(w_transf_type=w_transf_type)
        vis_util.plot_LWR_xt(FD=FD, data_variable=data_variable, title_save=title_save)


    def MSEJD(self, chains=None, burnin=0, params=['z', 'rho_j', 'u', 'w']):
        """
        MSEJD for FD parameters

        Parameters
        ----------
        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        burnin: int
            Burnin to discard
        params: list
            List of FD parameters
        """
        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=-1e10)
        d_df = {k: v[params][burnin::] for k,v in d_samples.items()}
        df_all_chains = pd.concat(list(d_df.values()))
        df_diff = df_all_chains.diff(axis=0).dropna(axis=0)
        return np.sum([np.square(elem).sum() for elem in df_diff.values]) * (1/(df_diff.shape[0]))

    def BC_MSEJD(self, BC_type, chains=None, burnin=0):
        """
        MSEJD for BCs

        Parameters
        ----------
        BC_type: str
            Either 'BC_outlet' or 'BC_inlet'

        chains : list
            list of requested chains. Set by default to None (which keeps all chains)
        """

        d_samples, le_count = self.keep_samples(chains=chains, log_post_lim=-1e10)
        array_BCs = np.concatenate([v[BC_type][burnin:] for k,v in d_samples.items()], axis=0)
        BC_diff = np.diff(array_BCs, axis=0)
        return np.sum([np.square(elem).sum() for elem in BC_diff]) * (1/BC_diff.shape[0])
