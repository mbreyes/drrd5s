#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ailacamara
"""


import math
from unittest import result
import pandas as pd
import seaborn as sns
import os
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
import pingouin as pg 
import drrdTools as dr

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")


plt.style.use('ggplot')
sns.set(style='ticks')

DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/')+'/'
PREFIX = 'BD'


# ------------------- MODEL FITTING: DOUBLE GAUSSIAN ------------------

# ------ BINNING ------

bin_width = 0.05

# Initial guess
init_params = [
    0.5,  # gamma
    5.0,  # mu1
    0.3,  # sigma1
    5.0,  # mu2
    0.5   # sigma2
]




def data_histogram(data, bin_width=bin_width, n_bins = 50, log = False):

    """Calculate the histogram of the data with specified bin width and return bin centers and density."""
   
    data = np.asarray(data, dtype=float).flatten()

    if log:

        data = data[data > 0]  # Filter out non-positive values for log transformation  

        log_data = np.log(data)
        #bins_log = np.logspace(np.min(log_data), np.max(log_data), 50)
        bins_log = np.linspace(np.min(log_data), np.max(log_data), n_bins)

        counts, bin_edges = np.histogram(log_data, bins=bins_log, density=False)
        hist_density = counts / (np.sum(counts) * np.diff(bins_log)[0])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
        #bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        hist_density = counts / (np.sum(counts) * bin_width)
        bin_centers = bin_edges[:-1] + bin_width/ 2
        
    return bin_centers, hist_density


# ----- GAUSSIANS -----


def gaussian(x, mu, sigma):
    """Standard unnormalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_pdf(x, mu, sigma):
    """Standard normalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)/ (sigma * np.sqrt(2 * np.pi))


def mixture_pdf(x, gamma, mu1, sigma1, mu2, sigma2):
    """The normalized Double Gaussian Mixture."""
    return (1 - gamma) * gaussian_pdf(x, mu1, sigma1) + (gamma) * gaussian_pdf(x, mu2, sigma2)


def mu_order_constraint(params):
# Constraint function: mu2 - mu1 >= 0
    return params[3] - params[1]  # mu2 - mu1



# ----- LEAST SQUARES FITTING -----

def mixture_gaussian(x, params):

#params: [gamma, mu1, sigma1, mu2, sigma2]
#gamma: mixing proportion (0 to 1)
#mu1, sigma1: mean and std of Gaussian 1
#mu2, sigma2: mean and std of Gaussian 2

# This function calculates the value of the double Gaussian at each point x
# xi is the normalization constant to ensure the area under the curve is 1

    gamma, mu1, sigma1, mu2, sigma2 = params
    g1 = gaussian(x, mu1, sigma1)
    g2 = gaussian(x, mu2, sigma2)
    f_t = (1 - gamma)*g1 + (gamma)*g2
    xi = np.sum(f_t) * bin_width
    return f_t / xi


def objective(data, params):
    # This is the function we want to minimize. It calculates the sum of squared differences
    gamma, mu1, sigma1, mu2, sigma2 = params
    bin_centers, hist_density = data_histogram(data, bin_width=bin_width)
    if not (0 <= gamma <= 1) or sigma1 <=0 or sigma2 <=0:
        return np.inf
    model = mixture_gaussian(bin_centers, params)
    return np.sum((hist_density - model)**2)


def LSF(data, init_params, log = False):
    
    bounds = [
    (0,1),       # gamma
    (None,None), # mu1
    (1e-6,None), # sigma1
    (None,None), # mu2
    (1e-6,None)  # sigma2
    ]

    # NonlinearConstraint object
    mu_constraint = NonlinearConstraint(mu_order_constraint, 0, np.inf)

    if log==True:

        data = data[data > 0]  # Filter out non-positive values for log transformation  
        log_data = np.log(data)
    
        result_nll = minimize(
        objective,
        init_params,
        args=(log_data,),
        bounds=bounds,
        constraints=[mu_constraint],
        options={'disp': True}
        )

        bin_centers, hist_density = data_histogram(data, bin_width=bin_width, log = True)

    else:
            
        # Fit with constraint
        result = minimize(
        objective,
        init_params,
        args=(data,),
        bounds=bounds,
        constraints=[mu_constraint],
        options={'disp': True}
        )

        bin_centers, hist_density = data_histogram(data, bin_width=bin_width)

    # Extract fitted parameters
    gamma, mu1, sigma1, mu2, sigma2 = result.x


    g1 = gaussian(bin_centers, mu1, sigma1)
    g2 = gaussian(bin_centers, mu2, sigma2)
    f_t = (1 - gamma)*g1 + gamma*g2
    xi = np.sum(f_t)*bin_width
    P = f_t/xi
    P1 = ((1 - gamma)*g1)/xi
    P2 = (gamma*g2)/xi

    #Parameter text
    param_text = (
    f"$\\gamma$ = {gamma:.3f}\n"
    f"$\\mu_1$ = {mu1:.3f}\n"
    f"$\\sigma_1$ = {sigma1:.3f}\n"
    f"$\\mu_2$ = {mu2:.3f}\n"
    f"$\\sigma_2$ = {sigma2:.3f}")

    df = pd.DataFrame({'sigma1': sigma1, 
                       'mu1': mu1, 
                       'sigma2': sigma2, 
                       'mu2': mu2, 
                       'gamma': gamma}, index=[0])
    
    # --- Print Results --
    print("--- LSF Optimization Results ---")
    print(f"Optimization Success: {result.success}")
    print(f"Gamma (Trial Proportion): {gamma:.3f} ({(gamma *100):.1f}% in Gaussian 2)")
    print(f"Gaussian 1: Mean = {mu1:.3f}, StdDev = {sigma1:.3f}")
    print(f"Gaussian 2: Mean = {mu2:.3f}, StdDev = {sigma2:.3f}")

    return {'bin_centers': bin_centers, 'hist_density': hist_density, 'P_t': P, 'P1': P1, 'P2': P2, 'param_text': param_text, 'df': df}



# ----- MAXIMUM LIKELIHOOD ESTIMATION (MLE) -----


def negative_log_likelihood(params, data):
    gamma, mu1, sigma1, mu2, sigma2 = params
    
    # Calculate the probability of every raw data point
    probabilities = mixture_pdf(data, gamma, mu1, sigma1, mu2, sigma2)
    
    # Safety check: prevent log(0) by adding a tiny number (epsilon)
    epsilon = 1e-9
    probabilities = np.maximum(probabilities, epsilon)
    
    # Sum the natural logs and make it negative
    nll = -np.sum(np.log(probabilities))
    return nll



def MLE(data, init_params, log = False):

    bounds = [
    (0,1),       # gamma
    (None,None), # mu1
    (1e-6,None), # sigma1
    (None,None), # mu2
    (1e-6,None)  # sigma2
    ]

    mu_constraint = NonlinearConstraint(mu_order_constraint, 0, np.inf)
    
    if log==True:

        data = data[data > 0]  # Filter out non-positive values for log transformation  
        log_data = np.log(data)
    
        result_nll = minimize(
        negative_log_likelihood,
        init_params,
        args=(log_data,),
        bounds=bounds,
        constraints=[mu_constraint],
        method = 'L-BFGS-B'
        )

        bin_centers, hist_density = data_histogram(data, bin_width=bin_width, log = True)

    else:

        result_nll = minimize(
        negative_log_likelihood,
        init_params,
        args=(data,),
        bounds=bounds,
        constraints=[mu_constraint],
        method = 'L-BFGS-B'
        )
        
        bin_centers, hist_density = data_histogram(data, bin_width=bin_width)

    
    fit_gamma, fit_mu1, fit_sigma1, fit_mu2, fit_sigma2 = result_nll.x

    P1 = (1- fit_gamma) * gaussian_pdf(bin_centers, fit_mu1, fit_sigma1)
    P2 = fit_gamma * gaussian_pdf(bin_centers, fit_mu2, fit_sigma2)
    P_t = P1 + P2


    # Parameter text
    param_text = (
        f"$\\gamma$ = {fit_gamma:.3f}\n"
        f"$\\mu_1$ = {fit_mu1:.3f}\n"
        f"$\\sigma_1$ = {fit_sigma1:.3f}\n"
        f"$\\mu_2$ = {fit_mu2:.3f}\n"
        f"$\\sigma_2$ = {fit_sigma2:.3f}"
    )

    df = pd.DataFrame({'sigma1': fit_sigma1, 
                       'mu1': fit_mu1, 
                       'sigma2': fit_sigma2, 
                       'mu2': fit_mu2, 
                       'gamma': fit_gamma}, index=[0])
    
    # --- Print Results --
    print("--- MLE Optimization Results ---")
    #print(f"Optimization Success: {result_nll.success}")
    print(f"Gamma (Trial Proportion): {fit_gamma:.3f} ({(fit_gamma*100):.1f}% in Gaussian 2)")
    print(f"Gaussian 1: Mean = {fit_mu1:.3f}, StdDev = {fit_sigma1:.3f}")
    print(f"Gaussian 2: Mean = {fit_mu2:.3f}, StdDev = {fit_sigma2:.3f}")

    return {'bin_centers': bin_centers, 'hist_density': hist_density, 'P_t': P_t, 'P1': P1, 'P2': P2, 'df': df, 'param_text': param_text}
  



# ----- MEAN RESPONSE DURATION -----

def mean_response_duration(data, phase= 'DRRD 2s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    df_mean = data.groupby(['rat', 'session']).duration.mean().reset_index()
    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='session', y='duration', hue='rat', data=df_mean)
    plt.xlabel('Session')
    plt.ylabel('Mean Response Duration')
    plt.title(f'{phase}: Mean Response Duration per Session')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_Response_Duration_{phase}.png'))
    plt.show()
    return

# ------ % OF CORRECT RESPONSES -----

def proportion_correct_responses(data, phase = 'DRRD 2s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    df_correct = data.groupby(['rat', 'session']).reinforced.sum().reset_index()
    df_tot = data.groupby(['rat', 'session']).reinforced.count().reset_index()
    df_correct['correct'] = df_correct['reinforced'] / df_tot['reinforced']

    plt.figure()    
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='session', y='correct', hue='rat', data=df_correct)
    plt.xlabel('Session')
    plt.ylabel('Proportion of Correct Responses')
    plt.title(f'{phase}: Proportion of Correct Responses per Session')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_{phase}.png'))
    plt.show()
    return


#-------- DISTRIBUTION OF RESPONSE DURATIONS ---------


def plot_fitting_mle_rats(data, session = 'late', init_params = init_params, log = False, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    
    df_parameters = pd.DataFrame()
    rats = data.rat.unique()
    n_rats = len(rats)
    n_cols = 3
    n_rows = math.ceil(n_rats / n_cols)

    subtitle = 'Double Gaussian (MLE Fit)'


    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,5*n_rows))
    axes = axes.flatten()

    

    for i, rat in enumerate(data.rat.unique()):

        ax = axes[i]  

        hist = data.query(f'rat == {rat}').duration.values
        
        res = MLE(hist, init_params, log=log)

        res['df']['rat'] = rat

        res['df']['session'] = session

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)
        
        bin_width = np.diff(res['bin_centers'])[0]  # Calculate bin width from bin centers
        
        if log:
            label = 'Log(Time (s))'
        else:
            label = 'Time (s)'

        ax.bar(res['bin_centers'], res['hist_density'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers'], res['P_t'], 'r-', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers'], res['P1'], 'g--', lw=2, label='Component 1')
        ax.plot(res['bin_centers'], res['P2'], 'b--', lw=2, label='Component 2')
        ax.text(
            0.45, 0.95, res['param_text'],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.set_xlabel(label)
        ax.set_ylabel('Probability Density')
        ax.set_title(f' Rat {rat}')
        ax.legend()


    plt.suptitle(f'{subtitle} - Session {session}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit__{session}.png'))  
    plt.show()

    return df_parameters


def plot_fitting_mle_groups(data, individual=False, session = 'late', init_params = init_params, log = False, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    
    df_parameters = pd.DataFrame()

    if individual:
        groups = data.group.unique()
        n_groups = len(groups)
        n_cols = 2
        n_rows = math.ceil(n_groups / n_cols)

    else: 
        n_cols = 1
        n_rows = 1

    subtitle = 'Double Gaussian (MLE Fit)'


    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,5*n_rows))
    axes = axes.flatten()

    

    for i, group in enumerate(data.group.unique()):

        ax = axes[i]  

        hist = data.query('group == @group').duration.values
        
        res = MLE(hist, init_params, log=log)

        res['df']['group'] = group

        res['df']['session'] = session

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)
        
        bin_width = np.diff(res['bin_centers'])[0]  # Calculate bin width from bin centers
        
        if log:
            label = 'Log(Time (s))'
        else:
            label = 'Time (s)'

        ax.bar(res['bin_centers'], res['hist_density'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers'], res['P_t'], 'r-', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers'], res['P1'], 'g--', lw=2, label='Component 1')
        ax.plot(res['bin_centers'], res['P2'], 'b--', lw=2, label='Component 2')
        ax.text(
            0.45, 0.95, res['param_text'],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.set_xlabel(label)
        ax.set_ylabel('Probability Density')
        ax.set_title(f' Group {group}')
        ax.legend()

    plt.suptitle(f'{subtitle} - Session {session}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit__{session}.png'))  
    plt.show()

    return df_parameters


def plot_fitting_lsf_rats(data, init_params = init_params, session = 'early'):
    
    rats = data.rat.unique()
    n_rats = len(rats)
    n_cols = 3
    n_rows = math.ceil(n_rats / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, rat in enumerate(data.rat.unique()):

        ax = axes[i]  
    
        hist = data.query(f'rat == {rat}').duration.values
        
        res = LSF(hist, init_params)

        res['df']['rat'] = rat

        res['df']['session'] = session

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)
        
        bin_width = np.diff(res['bin_centers'])[0]  # Calculate bin width from bin centers
        
        if log:
            label = 'Log(Time (s))'
        else:
            label = 'Time (s)'


        ax.bar(res['bin_centers'], res['hist_density'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers'], res['P_t'], 'r-', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers'], res['P1'], 'g--', lw=2, label='Component 1')
        ax.plot(res['bin_centers'], res['P2'], 'b--', lw=2, label='Component 2')
        ax.set_xlabel(label)
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Rat {rat}')
        ax.legend()

        ax.text(
            0.65, 0.95, res['param_text'],
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.suptitle(f'Double Gaussian Fit (LSF) - Session {session}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_LSF_Fit_{session}.png'))  
    plt.show()

    return df_parameters




def plot_fitting_lsf_groups(data, individual=False, session = 'late', init_params = init_params, log = False, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    
    df_parameters = pd.DataFrame()

    if individual:
        groups = data.group.unique()
        n_groups = len(groups)
        n_cols = 2
        n_rows = math.ceil(n_groups / n_cols)

    else: 
        n_cols = 1
        n_rows = 1

    subtitle = 'Double Gaussian (MLE Fit)'


    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,5*n_rows))
    axes = axes.flatten()

    

    for i, group in enumerate(data.group.unique()):

        ax = axes[i]  

        hist = data.query(f'group == {group}').duration.values
        
        res = MLE(hist, init_params, log=log)

        res['df']['group'] = group

        res['df']['session'] = session

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)
        
        bin_width = np.diff(res['bin_centers'])[0]  # Calculate bin width from bin centers
        
        if log:
            label = 'Log(Time (s))'
        else:
            label = 'Time (s)'

        ax.bar(res['bin_centers'], res['hist_density'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers'], res['P_t'], 'r-', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers'], res['P1'], 'g--', lw=2, label='Component 1')
        ax.plot(res['bin_centers'], res['P2'], 'b--', lw=2, label='Component 2')
        ax.text(
            0.45, 0.95, res['param_text'],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.set_xlabel(label)
        ax.set_ylabel('Probability Density')
        ax.set_title(f' Group {group}')
        ax.legend()

    plt.suptitle(f'{subtitle} - Session {session}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit__{session}.png'))  
    plt.show()

    return df_parameters


def mask_last(data, number_sessions = 2):

    last_sessions = data.groupby('rat')['session'].unique().apply(lambda x: x[-number_sessions:]).explode().reset_index()

    valid_keys = list(zip(last_sessions['rat'], last_sessions['session']))

    data['rat_session'] = list(zip(data['rat'], data['session']))

    data_final = data[data['rat_session'].isin(valid_keys)].drop(columns=['rat_session'])
    return data_final


def plot_parameters(df_parameters, hue = 'group', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, experiment = 'DRRD 2s retract', fit_type = 'MLE'):

    order = ['early', 'late']
    params = ['mu1', 'sigma1', 'mu2', 'sigma2', 'gamma']

    for param in params:
        plt.figure()
        sns.pointplot(x='session', y=param, data=df_parameters, hue = hue, palette = 'Set2', order = order)
        sns.boxplot(x='session', y=param, data=df_parameters, color='lightgray', fliersize=0, width=0.5, showfliers=False, boxprops=dict(alpha=0.3), order = order)
        plt.title(f'Boxplot of {param} - {fit_type} Fit - {experiment}')
        plt.xlabel('Session')
        plt.ylabel(param)
        plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_{param}_{fit_type}_{experiment}.png'))
        plt.show()
   


def compare_parameters(df_parameters, hue = 'group', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, experiment = 'DRRD 2s retract', fit_type = 'MLE'):

    order = ['early', 'late']
    params = ['mu1', 'sigma1', 'mu2', 'sigma2', 'gamma']

    for param in params:
        plt.figure()
        sns.pointplot(x='session', y=param, data=df_parameters, hue = hue, palette = 'Set2', order = order)
        plt.title(f'Comparison of {param} - {fit_type} Fit - {experiment}')
        plt.xlabel('Session')
        plt.ylabel(param)
        plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Comparison_{param}_{fit_type}_{experiment}.png'))
        plt.show()