#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ailacamara
"""


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


def n_bins(data, bin_width):

    """Calculate the number of bins for a histogram based on the data range and desired bin width."""
    exact_bins = np.arange(np.min(data), np.max(data)+bin_width, bin_width)
    return exact_bins


def data_histogram(data, bin_width=0.05):

    bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    hist_density = counts / (np.sum(counts) * bin_width)
    bin_centers = bin_edges[:-1] + bin_width/ 2
    return bin_centers, hist_density


# ----- GAUSSIANS -----


def gaussian(x, mu, sigma):
    """Standard normalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_pdf(x, mu, sigma):
    """Standard normalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)/ (sigma * np.sqrt(2 * np.pi))


def mixture_pdf(x, gamma, mu1, sigma1, mu2, sigma2):
    """The normalized Double Gaussian Mixture."""
    return gamma * gaussian_pdf(x, mu1, sigma1) + (1 - gamma) * gaussian_pdf(x, mu2, sigma2)


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
    f_t = (1 - gamma)*g1 + gamma*g2
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



def LSF(data, rat, init_params):
    
    bounds = [
    (0,1),       # gamma
    (None,None), # mu1
    (1e-6,None), # sigma1
    (None,None), # mu2
    (1e-6,None)  # sigma2
    ]

    # NonlinearConstraint object
    mu_constraint = NonlinearConstraint(mu_order_constraint, 0, np.inf)

    # Fit with constraint
    result = minimize(
    objective,
    init_params,
    args=(data,),
    bounds=bounds,
    constraints=[mu_constraint],
    options={'disp': True}
    )

    # Extract fitted parameters
    gamma, mu1, sigma1, mu2, sigma2 = result.x

    bin_centers, hist_density = data_histogram(data, bin_width=bin_width)

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

    return bin_centers, hist_density, P, P1, P2, param_text


def MLE(data, init_params):

    bounds = [
    (0,1),       # gamma
    (None,None), # mu1
    (1e-6,None), # sigma1
    (None,None), # mu2
    (1e-6,None)  # sigma2
    ]

    mu_constraint = NonlinearConstraint(mu_order_constraint, 0, np.inf)

    result_nll = minimize(
        negative_log_likelihood,
        init_params,
        args=(data,),
        bounds=bounds,
        constraints=[mu_constraint],
        method = 'L-BFGS-B'
    )

    fit_gamma, fit_mu1, fit_sigma1, fit_mu2, fit_sigma2 = result_nll.x

    bin_centers, hist_density = data_histogram(data, bin_width=bin_width)
    
    P1 = fit_gamma * gaussian_pdf(bin_centers, fit_mu1, fit_sigma1)
    P2 = (1 - fit_gamma) * gaussian_pdf(bin_centers, fit_mu2, fit_sigma2)
    P_t = P1 + P2


    # Parameter text
    param_text = (
        f"$\\gamma$ = {fit_gamma:.3f}\n"
        f"$\\mu_1$ = {fit_mu1:.3f}\n"
        f"$\\sigma_1$ = {fit_sigma1:.3f}\n"
        f"$\\mu_2$ = {fit_mu2:.3f}\n"
        f"$\\sigma_2$ = {fit_sigma2:.3f}"
    )

    # --- Print Results ---
    print("--- MLE Optimization Results ---")
    print(f"Optimization Success: {result.success}")
    print(f"Gamma (Trial Proportion): {fit_gamma:.3f} ({(fit_gamma*100):.1f}% in Gaussian 1)")
    print(f"Gaussian 1: Mean = {fit_mu1:.3f}, StdDev = {fit_sigma1:.3f}")
    print(f"Gaussian 2: Mean = {fit_mu2:.3f}, StdDev = {fit_sigma2:.3f}")

    return bin_centers, hist_density, P_t, P1, P2, param_text
  



# ----- MEAN RESPONSE DURATION -----

def mean_response_duration(data, phase= 'DRRD 2s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    df_mean = data.groupby(['rat', 'session']).duration.mean().reset_index()
    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='session', y='duration', hue='rat', data=df_mean)
    plt.xlabel('Session')
    plt.ylabel('Mean Response Duration')
    plt.title(f'{phase}: Mean Response Duration per Session')
    plt.show()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_Response_Duration_{phase}.png'))

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
    plt.show()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_{phase}.png'))

    return


#-------- DISTRIBUTION OF RESPONSE DURATIONS ---------


def plot_fitting_mle(data, session = 'last', init_params = init_params, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    axes = axes.flatten()

    for i, rat in enumerate(data.rat.unique()):

        ax = axes[i]  

        if session == 'last':
            last_session = data.query(f'rat == {rat}').session.max()
            data_histogram = data.query(f'rat == {rat} and session == {last_session}').duration
        if session == 'first':
            data_histogram = data.query(f'rat == {rat} and session == 1').duration 
        
        bin_centers, hist_density, P_t, P1, P2, param_text = MLE(data, init_params)

        ax.bar(bin_centers, hist_density, width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(bin_centers, P_t, 'r-', lw=2, label='Fitted P(t)')
        ax.plot(bin_centers, P1, 'g--', lw=2, label='Component 1')
        ax.plot(bin_centers, P2, 'b--', lw=2, label='Component 2')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Double Gaussian (MLE Fit) - Rat {rat}')
        ax.legend()

        ax.text(
            0.65, 0.95, param_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.suptitle(f'Double Gaussian Fit (MLE) - Session {session}', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit__{session}.png'))  




def plot_fitting_lsf(data, session = 'last', init_params = init_params):
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    axes = axes.flatten()

    for i, rat in enumerate(data.rat.unique()):

        ax = axes[i]  

        if session == 'last':
            last_session = data.query(f'rat == {rat}').session.max()
            data_histogram = data.query(f'rat == {rat} and session == {last_session}').duration
        if session == 'first':
            data_histogram = data.query(f'rat == {rat} and session == 1').duration 
        
        bin_centers, hist_density, P_t, P1, P2, param_text = LSF(data, init_params)

        ax.bar(bin_centers, hist_density, width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(bin_centers, P_t, 'r-', lw=2, label='Fitted P(t)')
        ax.plot(bin_centers, P1, 'g--', lw=2, label='Component 1')
        ax.plot(bin_centers, P2, 'b--', lw=2, label='Component 2')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Double Gaussian (LSF Fit) - Rat {rat}')
        ax.legend()

        ax.text(
            0.65, 0.95, param_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.suptitle(f'Double Gaussian Fit (LSE) - Session {session}', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_LSF_Fit_{session}.png'))  



