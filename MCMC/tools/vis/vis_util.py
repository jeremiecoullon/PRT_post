# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from MCMC.lwr.lwr_solver import LWR_Solver
from MCMC.tools.util import FD_neg_power



plot_params = {'alpha': 0.05, 'c': 'royalblue', 's': 6}

def plot_2_FD_params(x, y, ax, df):
    """
    Plot all samples for FD parameters x and y in a dataframe in a specified ax
    """
    ax.scatter(df[x], df[y], **plot_params)
    ax.set_xlabel(x, size=18)
    ax.set_ylabel(y, size=18, rotation='horizontal')
    ax.set_title("Samples for {} and {}".format(x,y), size=25)

def plot_FD_chain(x, y, ax, loc_list):
    """
    Plot several FD samples for parameters x and y
    """
    for elem in loc_list:
        FD_dict = df_all[['z', 'rho_j', 'u', 'w']].iloc[elem].to_dict()
        ax.scatter(FD_dict[x], FD_dict[y], c='r', s=60)

def plot_FD_proposals(x, y, ax, FD_proposal):
    for elem in FD_proposal:
        FD_dict = {k:v for k,v in zip(['z','rho_j','u', 'w'], elem)}
        ax.scatter(FD_dict[x], FD_dict[y], c='g', s=30, alpha=1)


def get_FD_jump(self, move_num, chains=None, burnin=0, move_type='global'):
    df = self.concat_chains(chains=[1,2,3], burnin=burnin, params=['z', 'rho_j', 'u', 'w', 'param_accept'])
    # get list of indexes where there is a global move
    global_list = list(df.loc[df.param_accept=='{}_a'.format(move_type)].index)
    # print("Number of {} moves: {}".format(move_type, len(global_list)))

    x_range = np.arange(0,500, 0.01)
    idx_num = global_list[move_num]
    df.iloc[global_list[move_num]]

    FD_1 = {k:v for k,v in df.iloc[global_list[move_num]-1].to_dict().items() if k in ['z', 'rho_j', 'u', 'w']}

    FD_2 = {k:v for k,v in df.iloc[global_list[move_num]].to_dict().items() if k in ['z', 'rho_j', 'u', 'w']}
    return FD_1, FD_2



def function_mean_sd(fun_series):
    """
    Plots the pointwise mean of a bunch of time series, along with the pointwise standard devitation

    Parameters
    ----------
    fun_series: pd.Series
        This should be a Series of arrays/lists. So each element is a function
    """
    fun_mean = fun_series.values.mean()
    fun_std = fun_series.values.std()
    x_range = np.arange(len(fun_mean))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.fill_between(x_range, fun_mean-fun_std, fun_mean+fun_std, alpha=0.5)
    ax.plot(fun_mean, c='r')
    plt.title("Pointwise mean curve along with standard deviation", size=17)
    plt.show()



def plot_LWR_xt(FD, data_variable, title_save=None, config_dict=None, out_times=None, figsize=(12,9)):
    """
    Run LWR using the given parameters and plot in the x-t plane

    Parameters
    ----------
    FD: dict
        Dictionary of FD parameter values. Must include the FD parameters along with the key 'solver',
        whos values can either be 'lwr_exp' or 'lwr_del_Cast'
    data_variable: str
        Variable to use to plot output: either 'flow' or 'density'
    title_save: None (default) or str
        If not None: save figure with given name
    config_dict: dict
        Configuration dictionary to pass to LWR_Solver. It should have 'data_array_dict' with the data CSVs
        Default is None, which uses the default config_dict
    figsize: tuple
        Default is (10, 10)
    """
    if config_dict is None:
        LWR = LWR_Solver()
    else:
        LWR = LWR_Solver(config_dict=config_dict)
    if out_times is None:
        pass
    else:
        LWR.out_times = out_times

    if 'z' in FD:
        FD["z"] = FD.pop('z')
    for elem in ['BC_Gibbs']:
        if elem in FD:
            FD.pop(elem)
    claw = LWR.lwr(**FD)
    x_min_solver = 0

    rho_claw = np.array([elem.q[0,:] for elem in claw.frames])

    # select time range
    t_min = 0
    t_max = LWR.final_time

    t_grid = np.linspace(t_min, t_max, len(LWR.out_times))
    x_grid = np.linspace(LWR.x_min_solver,LWR.x_max_solver,LWR.PDE_num_cells)
    # x_grid = x
    xx_LWR, tt_LWR = np.meshgrid(x_grid, t_grid)

    if data_variable=='flow':
        params = {k:v for k,v in FD.items() if k not in ['solver', 'BC_inlet', 'BC_outlet']}
        if FD.get('solver')=='lwr_exp':
            my_FD = lambda x: FD_exp(rho=x, **params)
        elif FD.get('solver')=='lwr_del_Cast':
            params["z"] = params.pop('z')
            my_FD = lambda x: FD_neg_power(rho=x, **params)
        rho_claw = list(map(my_FD, rho_claw))

    fig_lwr_xt = plt.figure(figsize=figsize)
    ax = fig_lwr_xt.add_subplot(1,1,1)
    CS = ax.contourf(tt_LWR, xx_LWR,rho_claw, cmap=cm.coolwarm)

    cbar = fig_lwr_xt.colorbar(CS, shrink=0.8, extend='both')
    cbar.set_label('Density (veh/km)', size=25)
    # ax.set_title("{0} from LWR with {1} FD".format(data_variable, FD.get('solver')[4:]), size = 15)
    ax.set_ylabel('Distance on the road (km)', size=30)
    ax.set_xlabel('Time (min)', size=30)
    plt.tight_layout()
    if title_save is not None:
        plt.savefig(title_save)
    # plt.show()


def plot_LWR_xt_on_ax(FD, data_variable, ax, fig, title_save=None, config_dict=None, out_times=None):
    """
    Run LWR using the given parameters and plot in the x-t plane

    Parameters
    ----------
    FD: dict
        Dictionary of FD parameter values. Must include the FD parameters along with the key 'solver',
        whos values can either be 'lwr_exp' or 'lwr_del_Cast'
    data_variable: str
        Variable to use to plot output: either 'flow' or 'density'
    title_save: None (default) or str
        If not None: save figure with given name
    config_dict: dict
        Configuration dictionary to pass to LWR_Solver. It should have 'data_array_dict' with the data CSVs
        Default is None, which uses the default config_dict
    figsize: tuple
        Default is (10, 10)
    """
    if config_dict is None:
        LWR = LWR_Solver()
    else:
        LWR = LWR_Solver(config_dict=config_dict)
    if out_times is None:
        pass
    else:
        LWR.out_times = out_times

    if 'z' in FD:
        FD["z"] = FD.pop('z')
    for elem in ['BC_Gibbs']:
        if elem in FD:
            FD.pop(elem)
    claw = LWR.lwr(**FD)
    x_min_solver = 0

    rho_claw = np.array([elem.q[0,:] for elem in claw.frames])

    # select time range
    t_min = 0
    t_max = LWR.final_time * LWR.config.ratio_times_BCs

    t_grid = np.linspace(t_min, t_max, len(LWR.out_times))
    x_grid = np.linspace(LWR.x_min_solver,LWR.x_max_solver,LWR.PDE_num_cells)
    # x_grid = x
    xx_LWR, tt_LWR = np.meshgrid(x_grid, t_grid)

    if data_variable=='flow':
        params = {k:v for k,v in FD.items() if k not in ['solver', 'BC_inlet', 'BC_outlet']}
        if FD.get('solver')=='lwr_del_Cast':
            params["z"] = params.pop('z')
            my_FD = lambda x: FD_neg_power(rho=x, **params)
        rho_claw = list(map(my_FD, rho_claw))
    CS = ax.contourf(tt_LWR, xx_LWR,rho_claw, cmap=cm.coolwarm)
    # CS = ax.contourf(tt_LWR, tt_LWR,rho_claw, cmap=cm.coolwarm)

    cbar = fig.colorbar(CS, shrink=0.8, extend='both')
    cbar.set_label('Density (veh/km)')#, size=25)
    # ax.set_title("{0} from LWR with {1} FD".format(data_variable, FD.get('solver')[4:]), size = 15)
    ax.set_ylabel('Distance on the road (km)')#, size=30)
    ax.set_xlabel('Time (min)')#, size=30)
    plt.tight_layout()
    if title_save is not None:
        plt.savefig(title_save)
    return ax
    # plt.show()
