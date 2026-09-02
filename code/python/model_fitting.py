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
from scipy.optimize import minimize, NonlinearConstraint, minimize_scalar, curve_fit
import pingouin as pg 
from sklearn.mixture import GaussianMixture
import drrdTools as dr
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator

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

bin_width = 0.1

# Initial guess
init_params = [
    0.5,  # gamma
    5.0,  # mu1
    0.3,  # sigma1
    5.0,  # delta_mu
    0.5   # sigma2
]



def data_histogram(data, bin_width=bin_width, n_bins = 60, log = False, calc_curve = False):

    """Calculate the histogram of the data with specified bin width and return bin centers and density."""
   
    data = np.asarray(data, dtype=float).flatten()


    if log:

        data = data[data > 0]  # Filter out non-positive values for log transformation  

        log_data = np.log(data)
        
        # calc_curve: fit a smooth curve in the functions defined through the given parameters values
        if calc_curve == True:

            n_bins = 700
            bins_log = np.linspace(np.log(0.02), np.log(30), n_bins)   
            counts, bin_edges = np.histogram(log_data, bins=bins_log, density=False)
            hist_density = counts / (np.sum(counts) * np.diff(bins_log)[0])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

        else:

            bins_log = np.arange(np.log(0.02), np.log(30) + bin_width, bin_width)
            #bins_log = np.arange(np.min(log_data), np.max(log_data) + bin_width, bin_width)
            counts, bin_edges = np.histogram(log_data, bins = bins_log, density=False)
            hist_density = counts / (np.sum(counts) * bin_width)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

    else:

        if calc_curve == True:

            dt = 0.01
            bins = np.arange(0.02, 30 + dt, dt)  
            counts, bin_edges = np.histogram(data, bins = bins, density=False)
            hist_density = counts / (np.sum(counts) * dt)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
        
        # calculates the histogram to be plotted in the graph
        else:

            bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
            counts, bin_edges = np.histogram(data, bins=bins, density=False)
            hist_density = counts / (np.sum(counts) * bin_width)
            bin_centers = bin_edges[:-1] + bin_width/ 2


        
    return bin_centers, hist_density


def lognormal_mean_sd(mu_log, sigma_log):
    mean_original = np.exp(mu_log + sigma_log**2 / 2)
    sd_original = mean_original * np.sqrt(np.exp(sigma_log**2) - 1)
    return mean_original, sd_original


# ----- GAUSSIANS -----

def estimate_init_params(data):
    # crude estimate: split data by median, take mean/std of each half
    median = np.median(data)
    lower = data[data <= median]
    upper = data[data > median]
    
    mu1_init = np.mean(lower)
    mu2_init = np.mean(upper)
    delta_mu_init = mu2_init - mu1_init
    sigma1_init = np.std(lower) if len(lower) > 1 else 0.5
    sigma2_init = np.std(upper) if len(upper) > 1 else 0.5
    gamma_init = 0.5

    return [gamma_init, mu1_init, sigma1_init, delta_mu_init, sigma2_init]

def estimate_init_params_lsf(data):
    """Para LSF (sem reparametrização): retorna mu2 diretamente."""

    data_reshaped = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gmm.fit(data_reshaped)
    
    # ordena os componentes por média (menor primeiro)
    order = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[order]
    stds = np.sqrt(gmm.covariances_.flatten())[order]
    weights = gmm.weights_[order]
    
    mu1_init, mu2_init = means[0], means[1]
    sigma1_init, sigma2_init = stds[0], stds[1]
    gamma_init = weights[1]  # peso do componente de maior média
    
    #median = np.median(data)
    #lower = data[data <= median]
    #upper = data[data > median]
    
    #mu1_init = np.mean(lower)
    #mu2_init = np.mean(upper)
    #sigma1_init = np.std(lower) if len(lower) > 1 else 0.5
    #sigma2_init = np.std(upper) if len(upper) > 1 else 0.5
    #gamma_init = 0.5

    return [gamma_init, mu1_init, sigma1_init, mu2_init, sigma2_init]

def gaussian(x, mu, sigma):
    """Standard unnormalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_pdf(x, mu, sigma):
    """Standard normalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)/ (sigma * np.sqrt(2 * np.pi))


def mixture(x, gamma, mu1, sigma1, mu2, sigma2):
    """The unormalized Double Gaussian Mixture."""
    return (1 - gamma) * gaussian(x, mu1, sigma1) + (gamma) * gaussian(x, mu2, sigma2)


def mu_order_constraint(params):
# Constraint function: mu2 - mu1 >= 0
    return params[3] - params[1]  # mu2 - mu1



# ----- LEAST SQUARES FITTING -----

def mixture_gaussian(x, params):

#params: [gamma, mu1, sigma1, mu2, sigma2]
#gamma: mixing proportion (0 to 1)
#mu1, sigma1: mean and std of Gaussian 1
#mu2, sigma2: mean and std of Gaussian 2

# Builds the unnormalized weighted mixture, then normalizes by xi, the
# integral of that raw curve over the actual x grid it was evaluated on
# (so it stays correct whether x is the coarse fitting grid or the fine
# plotting grid).

    gamma, mu1, sigma1, mu2, sigma2 = params
    f_t = mixture(x, gamma, mu1, sigma1, mu2, sigma2)
    xi = np.trapezoid(f_t, x)
    if not np.isfinite(xi) or xi <= 0:
        return np.full_like(x, np.inf)
    return f_t / xi


def objective_log(params, data):
    # This is the function we want to minimize. It calculates the sum of squared differences
    gamma, mu1, sigma1, mu2, sigma2 = params
    
    bin_centers, hist_density = data_histogram(data, log = True)
    if not (0 <= gamma <= 1) or sigma1 <=0 or sigma2 <=0:
        return np.inf
    model = mixture_gaussian(bin_centers, params)
    return np.sum((hist_density - model)**2)


def objective(params, data):
    # This is the function we want to minimize. It calculates the sum of squared differences
    gamma, mu1, sigma1, mu2, sigma2 = params
     
    bin_centers, hist_density = data_histogram(data)
    if not (0 <= gamma <= 1) or sigma1 <=0 or sigma2 <=0:
        return np.inf
    model = mixture_gaussian(bin_centers, params)
    return np.sum((hist_density - model)**2)



def LSF(data, log = False):
    
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

        init_params = estimate_init_params_lsf(log_data)

    
        result = minimize(
        objective_log,
        init_params,
        args=(data,),
        bounds=bounds,
        constraints=[mu_constraint],
        options={'disp': True},
        method = 'SLSQP'
        )

        bin_centers_curve, hist_density_curve = data_histogram(data, log = True, calc_curve = True)

        bin_centers_hist, hist_density_hist = data_histogram(data, log = True)
  

    else:
        
        init_params = estimate_init_params_lsf(data)
       
        # Fit with constraint
        result = minimize(
        objective,
        init_params,
        args=(data,),
        bounds=bounds,
        constraints=[mu_constraint],
        options={'disp': True},
        method = 'SLSQP'
        )

        bin_centers_curve, hist_density_curve = data_histogram(data, calc_curve = True)

        bin_centers_hist, hist_density_hist = data_histogram(data)


    # Extract fitted parameters
    gamma, mu1, sigma1, mu2, sigma2 = result.x


    g1 = gaussian(bin_centers_curve, mu1, sigma1)
    g2 = gaussian(bin_centers_curve, mu2, sigma2)
    f_t = (1 - gamma)*g1 + gamma*g2
    xi = np.trapezoid(f_t, bin_centers_curve)
    P_t = f_t/xi
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

    return {'bin_centers_curve': bin_centers_curve, 'hist_density_curve': hist_density_curve, \
            'bin_centers_hist': bin_centers_hist, 'hist_density_hist': hist_density_hist, \
            'P_t': P_t, 'P1': P1, 'P2': P2, 'df': df, 'param_text': param_text}
  


# ----- MAXIMUM LIKELIHOOD ESTIMATION (MLE) -----


def negative_log_likelihood(params, data):
    
    gamma, mu1, sigma1, delta_mu, sigma2 = params 

    mu2 = mu1 + delta_mu
    # Calculate the probability of every raw data point
    #P1 = (1-gamma) * gaussian(data, mu1, sigma1)
    #P2 = (gamma) * gaussian(data, mu2, sigma2)

    P1 = (1 - gamma) * gaussian_pdf(data, mu1, sigma1)
    P2 = gamma * gaussian_pdf(data, mu2, sigma2)
    probabilities = P1 + P2
    

    # Safety check: prevent log(0) by adding a tiny number (epsilon)
    epsilon = 1e-9
    probabilities = np.maximum(probabilities, epsilon)
    
    # Sum the natural logs and make it negative
    nll = -np.sum(np.log(probabilities))
    return nll


def calculate_gamma_2(gamma, mu1, sigma1, mu2, sigma2):
    """Alternative mixing weight based on the weighted height of each Gaussian at mu2
    (EM-style responsibility of component 2 at its own peak):
    gamma_2 = P2(mu2) / (P1(mu2) + P2(mu2)), where P1 = (1-gamma)*gaussian_pdf(.,mu1,sigma1)
    and P2 = gamma*gaussian_pdf(.,mu2,sigma2)."""
    height_g1_at_mu1 = (1 - gamma) * gaussian_pdf(mu1, mu1, sigma1)
    height_g2_at_mu2 = gamma * gaussian_pdf(mu2, mu2, sigma2)
    gamma_2 = height_g2_at_mu2 / (height_g1_at_mu1 + height_g2_at_mu2)
    return gamma_2


def MLE(data, log = False):

    bounds = [
    (0,1),       # gamma
    (None,None), # mu1
    (1e-6,None), # sigma1
    (0,None),    # delta_mu >= 0 <---- garante mu2 >= mu1
    (1e-6,None)  # sigma2
    ]

    if log==True:

        data = data[data > 0]  # Filter out non-positive values for log transformation  
        log_data = np.log(data)

        bin_centers_curve, hist_density_curve = data_histogram(data, bin_width=bin_width, log = True, calc_curve = True)
        init_params = estimate_init_params(log_data)
       
        
        result_nll = minimize(
        negative_log_likelihood,
        init_params,
        args=(log_data,),
        bounds=bounds,
        method = 'L-BFGS-B'
        )

       
    else:

        bin_centers_curve, hist_density_curve = data_histogram(data, bin_width=bin_width, calc_curve = True)
        init_params = estimate_init_params(data)
        
        result_nll = minimize(
        negative_log_likelihood,
        init_params,
        args=(data,),
        bounds=bounds,
        method = 'L-BFGS-B'
        )
        
    
    fit_gamma, fit_mu1, fit_sigma1, fit_delta_mu, fit_sigma2 = result_nll.x
    fit_mu2 = fit_mu1 + fit_delta_mu


    P1 = (1- fit_gamma) * gaussian_pdf(bin_centers_curve, fit_mu1, fit_sigma1)
    P2 = fit_gamma * gaussian_pdf(bin_centers_curve, fit_mu2, fit_sigma2)
    P_t = P1 + P2

    # Second (height-based) gamma, kept only for comparison in the parameter table
    gamma_2 = calculate_gamma_2(fit_gamma, fit_mu1, fit_sigma1, fit_mu2, fit_sigma2)


    if log:

        bin_centers_hist, hist_density_hist = data_histogram(data, bin_width=bin_width, log = True)
        
        mean1, sd1 = lognormal_mean_sd(fit_mu1, fit_sigma1)
        mean2, sd2 = lognormal_mean_sd(fit_mu2, fit_sigma2)

        # Parameter text
        param_text = (
            f"$\\gamma$ = {fit_gamma:.3f}\n"
            f"$\\gamma_2$ = {gamma_2:.3f}\n"
            f"$\\mu_1$ = {fit_mu1:.3f}\n"
            f"$\\sigma_1$ = {fit_sigma1:.3f}\n"
            f"$\\mu_2$ = {fit_mu2:.3f}\n"
            f"$\\sigma_2$ = {fit_sigma2:.3f}"
        )
    else:

        bin_centers_hist, hist_density_hist = data_histogram(data, bin_width=bin_width)

        # Parameter text
        param_text = (
            f"$\\gamma$ = {fit_gamma:.3f}\n"
            f"$\\gamma_2$ = {gamma_2:.3f}\n"
            f"$\\mu_1$ = {fit_mu1:.3f}\n"
            f"$\\sigma_1$ = {fit_sigma1:.3f}\n"
            f"$\\mu_2$ = {fit_mu2:.3f}\n"
            f"$\\sigma_2$ = {fit_sigma2:.3f}"
        )
    df = pd.DataFrame({'sigma1': fit_sigma1,
                       'mu1': fit_mu1,
                       'sigma2': fit_sigma2,
                       'mu2': fit_mu2,
                       'gamma': fit_gamma,
                       'gamma_2': gamma_2}, index=[0])
    
    
    # --- Print Results --
    print("--- MLE Optimization Results ---")
    #print(f"Optimization Success: {result_nll.success}")
    print(f"Gamma (Trial Proportion): {fit_gamma:.3f} ({(fit_gamma*100):.1f}% in Gaussian 2)")
    print(f"Gamma_2 (height-based): {gamma_2:.3f} ({(gamma_2*100):.1f}% in Gaussian 2)")
    print(f"Gaussian 1: Mean = {fit_mu1:.3f}, StdDev = {fit_sigma1:.3f}")
    print(f"Gaussian 2: Mean = {fit_mu2:.3f}, StdDev = {fit_sigma2:.3f}")

    return {'bin_centers_curve': bin_centers_curve, 'hist_density_hist': hist_density_curve, \
            'bin_centers_hist': bin_centers_hist, 'hist_density_hist': hist_density_hist, \
            'P_t': P_t, 'P1': P1, 'P2': P2, \
            'gamma_2': gamma_2, 'df': df, 'param_text': param_text}
  

def learning_curve(s, A0, Ainf, tau):
    return Ainf - (Ainf - A0) * np.exp(-s / tau)

# ---------- Rota 1: curve_fit ponderado (rápido, precisa de init decente) ----------
def fit_tau_curvefit(sessions, mu2, se_mu2=None):
    s = np.asarray(sessions, float)
    y = np.asarray(mu2, float)
    p0 = [y[0], np.mean(y[-4:]), max(1.0, 0.3*len(s))]      # chutes iniciais
    bounds = ([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(learning_curve, s, y, p0=p0, bounds=bounds,
                           sigma=se_mu2, absolute_sigma=(se_mu2 is not None),
                           maxfev=10000)
    A0, Ainf, tau = popt
    perr = np.sqrt(np.diag(pcov))
    corr = pcov[1,2] / (perr[1]*perr[2])                    # corr A_inf–tau
    return dict(A0=A0, Ainf=Ainf, tau=tau, tau_se_wald=perr[2],
                corr_Ainf_tau=corr, pcov=pcov)


CRITERION_10S_LOG_ASYMPTOTE = np.log(10)  # mu2 is fit on log-duration; log(10) = the 10s criterion on that scale
CRITERION_11S_LOG_ASYMPTOTE = np.log(11)  # Ainf pinned just above the 10s criterion, so a fixed-Ainf fit still
                                            # crosses CRITERION_10S_LOG_ASYMPTOTE in finite time (see sessions_to_reach_criterion)


# ---------- Rota 2: curve_fit com A_infinito fixo (ex.: na assíntota do critério 10s) ----------
def fit_tau_curvefit_fixed_ainf(sessions, mu2, ainf, se_mu2=None):
    """Same as fit_tau_curvefit, but Ainf is pinned at `ainf` instead of
    being freely estimated -- only A0 and tau are fit."""

    s = np.asarray(sessions, float)
    y = np.asarray(mu2, float)

    def model(s, A0, tau):
        return learning_curve(s, A0, ainf, tau)

    p0 = [y[0], max(1.0, 0.3*len(s))]
    bounds = ([-np.inf, 1e-3], [np.inf, np.inf])
    popt, pcov = curve_fit(model, s, y, p0=p0, bounds=bounds,
                           sigma=se_mu2, absolute_sigma=(se_mu2 is not None),
                           maxfev=10000)
    A0, tau = popt
    perr = np.sqrt(np.diag(pcov))
    return dict(A0=A0, Ainf=ainf, tau=tau, tau_se_wald=perr[1], pcov=pcov)


def sessions_to_reach_criterion(A0, Ainf, tau, target=CRITERION_10S_LOG_ASYMPTOTE):
    """Solves learning_curve(s, A0, Ainf, tau) = target for s: the number of
    sessions a fitted learning curve predicts are needed to reach `target`
    (default CRITERION_10S_LOG_ASYMPTOTE = ln(10), the log-duration value
    corresponding to the 10s criterion). Returns NaN if the fitted curve
    never reaches the target -- i.e. its asymptote is at or below the target
    (Ainf <= target) or it isn't increasing toward it (A0 >= target)."""

    if Ainf <= target or A0 >= target:
        return np.nan

    ratio = (Ainf - A0) / (Ainf - target)
    if ratio <= 1:
        return np.nan

    return tau * np.log(ratio)


def fit_learning_curve_per_rat(df_parameters, ainf=None):
    """Fit `learning_curve` to each rat's own mu2-vs-session trajectory (e.g. the
    19 sessions of a group) instead of to the group mean. If `ainf` is given,
    Ainf is pinned at that value (fit_tau_curvefit_fixed_ainf, e.g.
    CRITERION_10S_LOG_ASYMPTOTE) and only A0/tau are estimated; otherwise
    (default) Ainf is freely fit alongside A0/tau (fit_tau_curvefit).

    Returns:
        df_fit_curve: one row per rat/session with the model-predicted mu2
            value (rat, group, session, mu2_fit) evaluated at that rat's actual
            session points, for overlaying a group-averaged fitted curve on the
            raw mu2 boxplot.
        df_tau: one row per rat with the fitted A0, Ainf, tau (rat, group,
            A0, Ainf, tau).
    """

    df_fit_curve = pd.DataFrame()
    df_tau = pd.DataFrame()

    for rat in df_parameters['rat'].unique():
        rat_data = df_parameters.query('rat == @rat').sort_values('session')

        sessions_arr = rat_data['session'].values.astype(float)
        mu2 = rat_data['mu2'].values
        group = rat_data['group'].iloc[0]

        try:
            if ainf is None:
                fit = fit_tau_curvefit(sessions_arr, mu2)
            else:
                fit = fit_tau_curvefit_fixed_ainf(sessions_arr, mu2, ainf)
        except RuntimeError:
            continue

        mu2_fit = learning_curve(sessions_arr, fit['A0'], fit['Ainf'], fit['tau'])

        df_fit_curve = pd.concat([df_fit_curve, pd.DataFrame({
            'rat': rat, 'group': group, 'session': sessions_arr, 'mu2_fit': mu2_fit
        })], ignore_index=True)

        df_tau = pd.concat([df_tau, pd.DataFrame({
            'rat': [rat], 'group': [group],
            'A0': [fit['A0']], 'Ainf': [fit['Ainf']], 'tau': [fit['tau']]
        })], ignore_index=True)

    return df_fit_curve, df_tau

# ----- MEAN RESPONSE DURATION -----

def mean_response_duration(data, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):

    ALL_GROUPS = sorted(data['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    df_mean = data.groupby(['rat', 'session', 'group']).duration.mean().reset_index(name = 'mean_duration')
    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x='session', y='mean_duration', hue='group', data=df_mean, errorbar = 'se', palette=COLOR_MAP)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Session')
    plt.ylabel('Mean Response Duration (s)')
    plt.title(f'{PREFIX}: Mean Response Duration per Session')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_Response_Duration.png'))
    plt.show()
    plt.close()
    return


# ------ % OF CORRECT RESPONSES -----

def proportion_correct_responses(data, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):

    ALL_GROUPS = sorted(data['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    df_correct = data.groupby(['rat', 'session', 'group']).reinforced.mean().reset_index(name = 'correct')
    
    plt.figure()    
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x='session', y='correct', hue='group', data=df_correct, palette = COLOR_MAP, errorbar = 'se')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Session')
    plt.ylabel('Proportion of Correct Responses')
    plt.title(f'{PREFIX}: Proportion of Correct Responses per Session')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses.png'))
    plt.show()
    plt.close()
    return


# ------ % OF CORRECT RESPONSES, SPLIT BY WHETHER THE RAT REACHED THE MAX CRITERION -----

def proportion_correct_responses_by_criterion(data, crit=10, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Proportion of correct responses per session, faceted by group, splitting each
    group into two subgroups: rats that reached the max criterion (`crit`, default
    10s) at some point during the experiment, and rats that never did."""

    ALL_GROUPS = sorted(data['group'].unique())
    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    reached_label = f'Reached {crit}s'
    not_reached_label = f'Did not reach {crit}s'
    SUBGROUP_COLORS = {reached_label: '#5B9E95', not_reached_label: '#C5AA97'}

    df = data.copy()
    reached_max = df.query('criterion >= @crit').rat.unique()
    df['criterion_subgroup'] = np.where(df['rat'].isin(reached_max), reached_label, not_reached_label)


    df_prop_correct = df.groupby(['rat', 'session', 'group', 'criterion_subgroup']).reinforced.mean().reset_index(name='correct')

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(ALL_GROUPS):
        ax = axes[i]
        group_data = df_prop_correct.query('group == @group')

        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.lineplot(x='session', y='correct', hue='criterion_subgroup', hue_order=[reached_label, not_reached_label],
                     data=group_data, ax=ax, palette=SUBGROUP_COLORS, errorbar='se')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Session')
        ax.set_ylabel('Proportion of Correct Responses')
        ax.set_title(f'Group {group}')
        ax.legend(fontsize=8, title=None)

    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{PREFIX}: Proportion of Correct Responses per Session by Criterion Subgroup', fontsize=16)
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_by_Criterion.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return


# ------ % OF CORRECT RESPONSES, ONE LINE PER RAT -----

def proportion_correct_responses_by_rat(data, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Lineplot of the proportion of correct (reinforced) responses per session,
    with one line per rat, colored by rat and styled by group."""

    df_correct = data.groupby(['rat', 'session', 'group']).reinforced.mean().reset_index(name='correct')
    df_correct = df_correct.sort_values(['group', 'rat'])

    rats = sorted(df_correct['rat'].unique())
    palette = dict(zip(rats, sns.color_palette('tab10', n_colors=len(rats))))

    plt.figure(figsize=(8, 5))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x='session', y='correct', hue='rat', style='group', data=df_correct,
                 palette=palette, marker='o')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Session')
    plt.ylabel('Proportion of Correct Responses')
    plt.title(f'{PREFIX}: Proportion of Correct Responses per Session, per Rat')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_by_Rat.png'))
    plt.show()
    plt.close()
    return


# ------- CRITERION PROGRESSION PER RAT/SESSION/GROUP ------

def criterion_progression(data, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Point plot of the criterion progression across sessions for each group,
    taking the highest criterion each rat reached within each session."""

    ALL_GROUPS = sorted(data['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    df_criterion = data.groupby(['rat', 'session', 'group']).criterion.max().reset_index()

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='session', y='criterion', hue='group', data=df_criterion,
                  palette=COLOR_MAP, dodge=0.2, errorbar='se')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Session')
    plt.ylabel('Criterion')
    plt.title(f'{PREFIX}: Criterion Progression per Session')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Criterion_Progression.png'))
    plt.show()
    plt.close()

    return


# ------- % OF RATS FROM A GROUP THAT REACHED A CRITERIA ------

def percentage_rats_criteria(data, crit = 5, criterion = '5s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):

    ALL_GROUPS = sorted(data['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    crit = crit
    experiment = data.query('criterion >= @crit').groupby(['rat', 'group', 'session']).first().reset_index()

    number_rats= experiment.groupby(['group', 'session']).rat.nunique().reset_index(name = 'number_rats')
    total_rats = data.groupby(['group', 'session']).rat.nunique().reset_index(name = 'total_rats')
    df_tot = pd.merge(number_rats, total_rats, on=['group', 'session'])
    df_tot['percentage'] = df_tot['number_rats'] / df_tot['total_rats']

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='session', y='percentage', hue='group', data=df_tot, palette = COLOR_MAP, dodge = 0.2, errorbar = 'se')
    plt.ylabel("Fraction of Rats")
    plt.xlabel("Session")
    plt.title(f'Percentage of rats that reached criterion {criterion}')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Percentage_of_rats_reached_criterion_{criterion}.png'))
    plt.show()

    return

#--------- NUMBER OF SESSIONS TO REACH CRITERIA -------

def sessions_reach_criteria(data, criteria = [2,4,8,10], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):

    ALL_GROUPS = sorted(data['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))
    criteria = criteria

    df = data.copy()
    df['criterion'] = df['criterion'].round()
    df_total_sessions = df.groupby(['rat', 'group', 'criterion']).session.first().reset_index(name = 'n_sessions')
    df_total_sessions = df_total_sessions.query('criterion == @criteria')
    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='criterion', y='n_sessions', hue='group',data=df_total_sessions, palette=COLOR_MAP, dodge = 0.2, errorbar = 'se')
    plt.xlabel("Criterion")
    plt.ylabel("Number of sessions")
    plt.title('Mean of sessions to achieve criteria')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Sessions_per_group.png'))
    plt.show()
    plt.close() 

    return

#--------- NUMBER OF TRIALS TO REACH CRITERIA ---------

def trials_reach_criteria(data, criteria = [2,4,8,10], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):

    ALL_GROUPS = sorted(data['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    criteria = criteria
    df_total_trials = pd.DataFrame()

    for r in data.rat.unique():

        df = data.copy() 
        df_rat = df.query('rat == @r')
        df_rat['total_trials'] = range(1,df_rat['rat'].count()+1)
        df_total_trials = pd.concat([df_total_trials, df_rat])
        
    df_total_trials['criterion'] = df_total_trials['criterion'].round()
    df_ = df_total_trials.query('criterion == @criteria').groupby(['rat','group','criterion'])['total_trials'].first().reset_index()
    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.pointplot(x='criterion', y='total_trials', hue='group',data=df_, palette=COLOR_MAP, dodge = 0.2, errorbar = 'se')
    plt.xlabel("Criterion")
    plt.ylabel("Number of trials")
    plt.title('Mean of trials per group to achieve criteria')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Trials_per_group.png'))
    plt.show()
    plt.close() 

    return


#--------- NUMBER OF TRIALS TO REACH CRITERIA, ONE LINE PER RAT ---------

def trials_reach_criteria_by_rat(data, criteria=[1, 2, 3, 4, 5], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Number of trials each rat needed to first reach each criterion in `criteria`,
    plotted as one line per rat (instead of averaged per group)."""

    df_total_trials = pd.DataFrame()

    for r in data.rat.unique():
        df_rat = data.query('rat == @r').copy()
        df_rat['total_trials'] = range(1, len(df_rat) + 1)
        df_total_trials = pd.concat([df_total_trials, df_rat])

    df_total_trials['criterion'] = df_total_trials['criterion'].round()
    df_ = df_total_trials.query('criterion == @criteria').groupby(
        ['rat', 'group', 'criterion'])['total_trials'].first().reset_index()

    rats = sorted(df_['rat'].unique())
    palette = dict(zip(rats, sns.color_palette('tab10', n_colors=len(rats))))

    plt.figure(figsize=(8, 5))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x='criterion', y='total_trials', hue='rat', style='group', data=df_,
                 palette=palette, marker='o')
    plt.xlabel("Criterion (s)")
    plt.ylabel("Number of trials")
    plt.title(f'{PREFIX}: Number of Trials per Rat to Reach Criteria')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Trials_per_rat.png'))
    plt.show()
    plt.close()

    return df_

def sessions_reach_criteria_by_rat(data, criteria=[1, 2, 3, 4, 5], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Number of sessions each rat needed to first reach each criterion in `criteria`,
    plotted as one line per rat (instead of averaged per group)."""

    df_total_sessions = pd.DataFrame()

    for r in data.rat.unique():
        df_rat = data.query('rat == @r').copy()
        df_rat['total_sessions'] = range(1, len(df_rat) + 1)
        df_total_sessions = pd.concat([df_total_sessions, df_rat])

    df_total_sessions['criterion'] = df_total_sessions['criterion'].round()
    df_ = df_total_sessions.query('criterion == @criteria').groupby(
        ['rat', 'group', 'criterion'])['total_sessions'].first().reset_index()

    rats = sorted(df_['rat'].unique())
    palette = dict(zip(rats, sns.color_palette('tab10', n_colors=len(rats))))

    plt.figure(figsize=(8, 5))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x='criterion', y='total_sessions', hue='rat', style='group', data=df_,
                 palette=palette, marker='o')
    plt.xlabel("Criterion (s)")
    plt.ylabel("Number of sessions")
    plt.title(f'{PREFIX}: Number of Sessions per Rat to Reach Criteria')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Sessions_per_rat.png'))
    plt.show()
    plt.close()

    return df_


# -------- BOXPLOT OF THE AVERAGE DURATION RESPONSE OF THE LAST THREE SESSIONS --------

def boxplot_groups(data, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):

    ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    last_sessions = data.query('session >= 16').groupby(['rat', 'group']).duration.mean().reset_index(name = 'mean_duration')
    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='mean_duration', data=last_sessions , width=0.5, showfliers=False, boxprops=dict(alpha=0.3), palette=COLOR_MAP)
    plt.xlabel("Groups")
    plt.ylabel("Average duration (s)")
    plt.title('Average response duration of the last sessions')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Average_per_group.png'))
    plt.show()
    plt.close()

    return


def boxplot_groups_key_sessions(data, key_sessions=[2, 12, 19], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """
    Distribuicao da duracao media de resposta por rato (um ponto = um rato)
    em sessoes-chave selecionadas, com boxplot + pontos individuais,
    coloridos por grupo (hue = group).
    """

    ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    df_key = data.query('session in @key_sessions').groupby(['rat', 'session', 'group']).duration.mean().reset_index(name='mean_duration')

    plt.figure(figsize=(2.5 * len(key_sessions) + 2, 5))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    ax = sns.boxplot(x='session', y='mean_duration', hue='group', data=df_key, order=key_sessions, hue_order=ALL_GROUPS,
                      width=0.6, showfliers=False, boxprops=dict(alpha=0.3), palette=COLOR_MAP)
    sns.stripplot(x='session', y='mean_duration', hue='group', data=df_key, order=key_sessions, hue_order=ALL_GROUPS,
                  dodge=True, palette=COLOR_MAP, edgecolor='black', linewidth=0.5, size=4, ax=ax, legend=False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(ALL_GROUPS)], labels[:len(ALL_GROUPS)], title='Group')

    plt.xlabel('Session')
    plt.ylabel('Mean Response Duration (s)')
    plt.title(f'{PREFIX}: Mean Response Duration Distribution (Key Sessions)')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_MeanDuration_KeySessions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def duration_by_trial_last_session(data, n_trials=30, last_session=None, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """
    Duracao da resposta (eixo y) por trial (eixo x) na ultima sessao,
    restrita as primeiras e ultimas `n_trials` tentativas de cada rato. Uma
    figura maior por grupo, com um subplot por rato (trial x duracao),
    colorido por segmento (primeiras vs ultimas tentativas).
    """

    ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order

    if last_session is None:
        last_session = data['session'].max()

    df_last = data.query('session == @last_session').copy()
    df_last['max_trial'] = df_last.groupby('rat')['trial'].transform('max')

    df_sel = df_last[(df_last['trial'] <= n_trials) | (df_last['trial'] > df_last['max_trial'] - n_trials)].copy()

    first_label, last_label = f'First {n_trials}', f'Last {n_trials}'
    df_sel['trial_segment'] = np.where(df_sel['trial'] <= n_trials, first_label, last_label)
    segment_palette = {first_label: '#5B9E95', last_label: '#C5AA97'}

    # posiciona os dois blocos lado a lado (1..n_trials e depois n_trials+gap..2*n_trials+gap),
    # em vez do numero de trial cru — que deixaria um vao enorme e vazio entre os dois blocos
    # quando a sessao tem muito mais que 2*n_trials tentativas.
    gap = max(3, round(n_trials * 0.15))
    last_mask = df_sel['trial_segment'] == last_label
    df_sel['trial_plot_pos'] = df_sel['trial']
    df_sel.loc[last_mask, 'trial_plot_pos'] = (
        n_trials + gap + (df_sel.loc[last_mask, 'trial'] - (df_sel.loc[last_mask, 'max_trial'] - n_trials))
    )
    separator_x = n_trials + gap / 2

    for group in ALL_GROUPS:
        group_data = df_sel.query('group == @group')
        rats = sorted(group_data['rat'].unique())
        n_rats = len(rats)
        if n_rats == 0:
            continue

        n_cols = 3
        n_rows = math.ceil(n_rats / n_cols)

        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                                  constrained_layout=True, sharey=True)
        axes = np.atleast_1d(axes).flatten()

        for i, rat in enumerate(rats):
            ax = axes[i]
            rat_data = group_data.query('rat == @rat').sort_values('trial')
            sns.scatterplot(x='trial_plot_pos', y='duration', hue='trial_segment', hue_order=[first_label, last_label],
                             data=rat_data, ax=ax, palette=segment_palette, s=25, legend=(i == 0))
            ax.axvline(separator_x, color='grey', linestyle=':', lw=1)
            ax.set_title(f'Rat {rat}')
            ax.set_xlabel('Trial (position within first/last block)')
            ax.set_ylabel('Duration (s)')
            ax.set_xticks([1, n_trials, n_trials + gap + 1, 2 * n_trials + gap])
            ax.set_xticklabels(['1', str(n_trials), '1', str(n_trials)])
            if i == 0:
                ax.legend(fontsize=8, title=None, loc='upper right')

        for j in range(n_rats, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f'{PREFIX} - {group}: Response Duration by Trial '
                     f'(Session {last_session}, first/last {n_trials} trials)', fontsize=16)
        fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_{group}_Duration_by_Trial_LastSession.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    return


#-------- DISTRIBUTION OF RESPONSE DURATIONS ---------


def plot_fitting_mle_rats(data, session = '1,2,3', log = False, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
     
    """Double Gaussian (MLE) fit for each group, with one subplot per rat.
        `data` must contain the trials of all groups and animals and the selected sessions."""
    
    df_parameters = pd.DataFrame()
    rats = data.rat.unique()
    n_rats = len(rats)
    n_cols = 3
    n_rows = math.ceil(n_rats / n_cols)
    group_name = data['group'].iloc[0]  # Assuming all rats in the data belong to the same group

    subtitle = f'Double Gaussian (MLE Fit) - {group_name}'


    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
    axes = axes.flatten()

    
    for i, rat in enumerate(rats):

        ax = axes[i]  

        hist = data.query(f'rat == {rat}').duration.values
        
        res = MLE(hist, log=log)

        res['df']['rat'] = rat

        res['df']['session'] = session

        res['df']['group'] = group_name

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)
        
        bin_width = np.diff(res['bin_centers_hist'])[0]  # Calculate bin width from bin centers

        
        if log:

            # choose the real-world tick values you want to display
            tick_values = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30]

            # convert them to log-space positions (since your data/axis is in natural log)
            tick_positions = np.log(tick_values)

            ax.xaxis.set_major_locator(FixedLocator(tick_positions))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{np.exp(x):.3g}'))


        ax.bar(res['bin_centers_hist'], res['hist_density_hist'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers_curve'], res['P_t'], color = 'black', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers_curve'], res['P1'], '--', color = 'C0', lw=2, label='Component 1')
        ax.plot(res['bin_centers_curve'], res['P2'], '--', color = 'C1', lw=2, label='Component 2')

        last_criterion = data.query(f'rat == {rat}').criterion.max()
        criterion_x = np.log(last_criterion) if log else last_criterion
        ax.axvline(criterion_x, color='grey', linestyle='--', lw=1.5, label='Last Criterion')

        ax.text(
            0.98, 0.95, res['param_text'],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f' Rat {rat}')
        ax.legend(loc='upper left')


    plt.suptitle(f'{subtitle} - Session {session}', fontsize=16)
    #plt.tight_layout()
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit_{group_name}_{session}.png'))
    plt.show()
    plt.close()

    return df_parameters



def plot_fitting_mle_sessions_per_rat(data, log=False, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Double Gaussian (MLE) fit for a single rat, with one subplot per session.
    `data` must contain the trials of a single rat across all of its sessions."""

    df_parameters = pd.DataFrame()
    rat = data['rat'].iloc[0]
    group_name = data['group'].iloc[0]
    sessions = sorted(data['session'].unique())
    n_sessions = len(sessions)
    n_cols = 3
    n_rows = math.ceil(n_sessions / n_cols)

    subtitle = f'Double Gaussian (MLE Fit) - Rat {rat} (Group {group_name})'

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, session in enumerate(sessions):

        ax = axes[i]

        hist = data.query('session == @session').duration.values

        res = MLE(hist, log=log)

        res['df']['rat'] = rat
        res['df']['session'] = session
        res['df']['group'] = group_name

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)

        bin_width = np.diff(res['bin_centers_hist'])[0]  # Calculate bin width from bin centers

        if log:

            # choose the real-world tick values you want to display
            tick_values = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30]

            # convert them to log-space positions (since your data/axis is in natural log)
            tick_positions = np.log(tick_values)

            ax.xaxis.set_major_locator(FixedLocator(tick_positions))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{np.exp(x):.3g}'))

        ax.bar(res['bin_centers_hist'], res['hist_density_hist'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers_curve'], res['P_t'], color='black', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers_curve'], res['P1'], '--', color='C0', lw=2, label='Component 1')
        ax.plot(res['bin_centers_curve'], res['P2'], '--', color='C1', lw=2, label='Component 2')

        last_criterion = data.query('session == @session').criterion.max()
        criterion_x = np.log(last_criterion) if log else last_criterion
        ax.axvline(criterion_x, color='grey', linestyle='--', lw=1.5, label='Last Criterion')

        ax.text(
            0.98, 0.95, res['param_text'],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Session {session}')
        ax.legend(loc='upper left')

    for j in range(n_sessions, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(subtitle, fontsize=16)
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit_Rat{rat}_all_sessions.png'))
    plt.show()
    plt.close(fig)

    return df_parameters


def plot_fitting_lsf_rats(data, session = 'early', log = False, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    
    df_parameters = pd.DataFrame()
    rats = data.rat.unique()
    n_rats = len(rats)
    n_cols = 3
    n_rows = math.ceil(n_rats / n_cols)
    group_name = data['group'].iloc[0]  # Assuming all rats in the data belong to the same group

    subtitle = f'Double Gaussian (LSF Fit) - {group_name}'


    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
    axes = axes.flatten()

    
    for i, rat in enumerate(rats):

        ax = axes[i]  

        hist = data.query(f'rat == {rat}').duration.values
        
        res = LSF(hist, log=log)

        res['df']['rat'] = rat

        res['df']['session'] = session

        res['df']['group'] = group_name

        df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)
        
        bin_width = np.diff(res['bin_centers_hist'])[0]  # Calculate bin width from bin centers

        
        if log:

            # choose the real-world tick values you want to display
            tick_values = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30]

            # convert them to log-space positions (since your data/axis is in natural log)
            tick_positions = np.log(tick_values)

            ax.xaxis.set_major_locator(FixedLocator(tick_positions))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{np.exp(x):.3g}'))


        ax.bar(res['bin_centers_hist'], res['hist_density_hist'], width=bin_width, alpha=0.4, label='Normalized Histogram')
        ax.plot(res['bin_centers_curve'], res['P_t'], color = 'black', lw=2, label='Fitted P(t)')
        ax.plot(res['bin_centers_curve'], res['P1'], '--', color = 'C0', lw=2, label='Component 1')
        ax.plot(res['bin_centers_curve'], res['P2'], '--', color = 'C1', lw=2, label='Component 2')

        last_criterion = data.query(f'rat == {rat}').criterion.max()
        criterion_x = np.log(last_criterion) if log else last_criterion
        ax.axvline(criterion_x, color='grey', linestyle='--', lw=1.5, label='Last Criterion')

        ax.text(
            0.98, 0.95, res['param_text'],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f' Rat {rat}')
        ax.legend(loc='upper left')


    plt.suptitle(f'{subtitle} - Session {session}', fontsize=16)
    #plt.tight_layout()
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_MLE_Fit_{group_name}_{session}.png'))
    plt.show()
    plt.close()

    return df_parameters


def fit_params_all_sessions(data, fit_function, log=True):
    """Fit `fit_function` (MLE or LSF) per rat and per individual session (no plotting),
    returning a single concatenated parameters dataframe with a 'session' column per fit."""

    df_parameters = pd.DataFrame()

    for session in sorted(data['session'].unique()):
        session_data = data.query('session == @session')

        for rat in session_data['rat'].unique():
            rat_data = session_data.query('rat == @rat')
            hist = rat_data.duration.values
            if len(hist) == 0:
                continue

            res = fit_function(hist, log=log)
            res['df']['rat'] = rat
            res['df']['session'] = session
            res['df']['group'] = rat_data['group'].iloc[0]

            df_parameters = pd.concat([df_parameters, res['df']], ignore_index=True)

    return df_parameters


def plot_param_method_comparison_boxplot(df_mle, df_lsf, param, order=None, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Boxplot of `param` (mu1, mu2, sigma1, sigma2 or gamma) across all sessions,
    faceted by group, comparing the MLE and LSF fits."""

    df_mle = df_mle[['session', 'group', param]].copy()
    df_mle['fit_type'] = 'MLE'

    df_lsf = df_lsf[['session', 'group', param]].copy()
    df_lsf['fit_type'] = 'LSF'

    df_param = pd.concat([df_mle, df_lsf], ignore_index=True)

    ALL_GROUPS = sorted(df_param['group'].unique())
    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    if order is None:
        order = sorted(df_param['session'].unique())

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(ALL_GROUPS):
        ax = axes[i]
        parameter = df_param.query('group == @group')

        sns.boxplot(x='session', y=param, hue='fit_type', data=parameter, order=order,
                    ax=ax, width=0.6, showfliers=False, boxprops=dict(alpha=0.5))
        ax.set_xlabel('Session')
        ax.set_ylabel(param)
        ax.set_title(f'Group {group}')

    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{PREFIX}: {param} - MLE vs LSF', fontsize=16)
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_{param}_MLE_vs_LSF_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return


def plot_gamma_vs_gamma2_boxplot(df_mle, order=None, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """Boxplot comparing gamma and gamma_2 across all sessions, faceted by group, for the MLE fit."""

    df_long = df_mle[['session', 'group', 'gamma', 'gamma_2']].melt(
        id_vars=['session', 'group'], var_name='parameter', value_name='value')

    ALL_GROUPS = sorted(df_long['group'].unique())
    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    if order is None:
        order = sorted(df_long['session'].unique())

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(ALL_GROUPS):
        ax = axes[i]
        parameter = df_long.query('group == @group')

        sns.boxplot(x='session', y='value', hue='parameter', data=parameter, order=order,
                    ax=ax, width=0.6, showfliers=False, boxprops=dict(alpha=0.5))
        ax.set_xlabel('Session')
        ax.set_ylabel('Gamma')
        ax.set_title(f'Group {group}')

    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{PREFIX}: Gamma vs Gamma_2 (MLE Fit)', fontsize=16)
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_Gamma_vs_Gamma2_MLE_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return



def set_nice_log_ticks(ax):
    ymin, ymax = ax.get_ylim()
    rmin, rmax = np.exp(ymin), np.exp(ymax)

    locator = LogLocator(base=10, subs='auto')
    nice_values = locator.tick_values(rmin, rmax)
    nice_values = nice_values[(nice_values >= rmin) & (nice_values <= rmax)]

    ax.set_yticks(np.log(nice_values))
    ax.set_yticklabels([f'{v:g}' for v in nice_values])


def mask_last(data, number_sessions = 2):

    last_sessions = data.groupby('rat')['session'].unique().apply(lambda x: x[-number_sessions:]).explode().reset_index()

    valid_keys = list(zip(last_sessions['rat'], last_sessions['session']))

    data['rat_session'] = list(zip(data['rat'], data['session']))

    data_final = data[data['rat_session'].isin(valid_keys)].drop(columns=['rat_session'])
    return data_final


def plot_parameters_boxplot(df_parameters, order=['early', 'late'], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE', log=True):
    """If `log` is True (the fit was performed on log-duration data), mu1/sigma1/mu2/sigma2
    are exponentiated (e raised to the parameter value) so their axis reads in seconds;
    gamma/gamma_2 are mixture proportions and are always left as-is."""

    params = ['mu1', 'sigma1', 'mu2', 'sigma2', 'gamma', 'gamma_2']
    time_params = {'mu1', 'sigma1', 'mu2', 'sigma2'}

    df_plot = df_parameters.copy()
    if log:
        for p in time_params:
            df_plot[p] = np.exp(df_plot[p])

    ALL_GROUPS = sorted(df_plot['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    for param in params:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                  figsize=(6 * n_cols, 5 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        for i, group in enumerate(ALL_GROUPS):
            ax = axes[i]
            parameter = df_plot.query('group == @group')

            sns.boxplot(x='session', y=param, data=parameter, order=order,
                        ax=ax, width=0.5, showfliers=False,
                        boxprops=dict(alpha=0.3), color=COLOR_MAP[group])

            ax.set_ylabel(f'{param} (s)' if log and param in time_params else param)

            ax.set_xlabel('Session')
            ax.set_title(f'Group {group}')

        # remove eixos vazios se n_groups não preencher o grid
        for j in range(n_groups, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f'{PREFIX}: Parameter {param}', x =0.5, fontsize=16)
        fig.savefig(os.path.realpath(
            f'{OUTPUT_PATH}/{PREFIX}_Boxplot_{param}_{fit_type}_group_average.png'),
            dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    return


def plot_parameters_lineplot_by_rat(df_parameters, order=None, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE', log=True):
    """Lineplot of each fitted parameter across sessions, one line per rat (all rats
    overlaid on the same axes, styled by group), with one subplot per parameter.

    If `log` is True (the fit was performed on log-duration data), mu1/sigma1/mu2/sigma2
    are exponentiated (e raised to the parameter value) so their axis reads in seconds;
    gamma/gamma_2 are mixture proportions and are always left as-is."""

    params = ['mu1', 'sigma1', 'mu2', 'sigma2', 'gamma', 'gamma_2']
    time_params = {'mu1', 'sigma1', 'mu2', 'sigma2'}

    df_plot = df_parameters.copy()
    if log:
        for p in time_params:
            df_plot[p] = np.exp(df_plot[p])

    if order is None:
        order = sorted(df_plot['session'].unique())

    rats = sorted(df_plot['rat'].unique())
    palette = dict(zip(rats, sns.color_palette('tab10', n_colors=len(rats))))

    n_params = len(params)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    handles, labels = None, None

    for i, param in enumerate(params):
        ax = axes[i]

        sns.lineplot(x='session', y=param, hue='rat', style='group', data=df_plot,
                     ax=ax, palette=palette, marker='o')

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Session')
        ax.set_ylabel(f'{param} (s)' if log and param in time_params else param)
        ax.set_title(param)

    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=8)

    plt.suptitle(f'{PREFIX}: Parameters per Rat ({fit_type} Fit)', fontsize=16)
    fig.savefig(os.path.realpath(
        f'{OUTPUT_PATH}/{PREFIX}_Lineplot_Parameters_{fit_type}_by_rat.png'),
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return


def plot_parameters_lineplot(df_parameters, order=['early', 'late'], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE', log=True):
    """If `log` is True (the fit was performed on log-duration data), mu1/sigma1/mu2/sigma2
    are exponentiated (e raised to the parameter value) so their axis reads in seconds;
    gamma/gamma_2 are mixture proportions and are always left as-is."""

    params = ['mu1', 'sigma1', 'mu2', 'sigma2', 'gamma', 'gamma_2']
    time_params = {'mu1', 'sigma1', 'mu2', 'sigma2'}

    df_plot = df_parameters.copy()
    if log:
        for p in time_params:
            df_plot[p] = np.exp(df_plot[p])

    ALL_GROUPS = sorted(df_plot['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    for param in params:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                  figsize=(6 * n_cols, 5 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        for i, group in enumerate(ALL_GROUPS):
            ax = axes[i]
            parameter = df_plot.query('group == @group')

            sns.lineplot(x='session', y=param, data=parameter,
                        ax=ax,errorbar = 'se', color=COLOR_MAP[group])

            ax.set_ylabel(f'{param} (s)' if log and param in time_params else param)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Session')
            ax.set_title(f'Group {group}')

        # remove eixos vazios se n_groups não preencher o grid
        for j in range(n_groups, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f'{PREFIX}: Parameter {param}', fontsize=16)
        fig.savefig(os.path.realpath(
            f'{OUTPUT_PATH}/{PREFIX}_Lineplot_{param}_{fit_type}_group_average.png'),
            dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    return


def plot_mu2_learning_curve_boxplot(df_parameters, order=None, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot of raw mu2 per session, faceted by group (same layout as
    plot_parameters_boxplot), with the group-average learning curve overlaid as a
    lineplot (errorbar='se'). The learning curve is fit per rat over that rat's
    OWN full session range (via fit_learning_curve_per_rat on the unfiltered
    df_parameters -- so rats with more sessions than others, e.g. a
    continuation phase, are each fit on however much data they actually have).

    `order` -- as in the other plot_*_boxplot functions -- both sets the
    x-axis session order AND, independently of the fit, caps which sessions
    are actually drawn: pass e.g. order=list(range(1,20)) to fit every rat on
    its full session range but only display sessions 1-19 in the boxplot/
    curve overlay (rats with more sessions just contribute more precisely
    fit A0/Ainf/tau, without stretching the plotted x-axis). Defaults to
    every session present in df_parameters (no cap) if not given.

    Returns df_tau (one row per rat: rat, group, A0, Ainf, tau, fit over the
    rat's full session range regardless of `order`) so the caller can feed it
    to plot_tau_boxplot.
    """

    ALL_GROUPS = sorted(df_parameters['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    if order is None:
        order = sorted(df_parameters['session'].unique())
    order_arr = np.asarray(order, dtype=float)

    df_fit_curve, df_tau = fit_learning_curve_per_rat(df_parameters)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(ALL_GROUPS):
        ax = axes[i]
        parameter = df_parameters.query('group == @group and session in @order')

        sns.boxplot(x='session', y='mu2', data=parameter, order=order,
                    ax=ax, width=0.5, showfliers=False,
                    boxprops=dict(alpha=0.3), color=COLOR_MAP[group])

        # cap the curve overlay to the displayed sessions too -- otherwise a rat
        # with sessions beyond `order` would have those points clamped by
        # np.interp onto the last displayed position instead of being dropped
        curve_group = df_fit_curve.query('group == @group and session in @order').copy()
        if not curve_group.empty:
            curve_group['session_pos'] = np.interp(curve_group['session'], order_arr, np.arange(len(order_arr)))
            sns.lineplot(x='session_pos', y='mu2_fit', data=curve_group,
                        ax=ax, errorbar='se', color='black', lw=2,
                        label='Learning curve (mean $\\pm$ SE)')
            ax.legend(fontsize=8)

        ax.set_ylabel('mu2')
        ax.set_xlabel('Session')
        ax.set_title(f'Group {group}')

    # remove eixos vazios se n_groups não preencher o grid
    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{PREFIX}: mu2 Learning Curve (per-rat fit)', fontsize=16)
    fig.savefig(os.path.realpath(
        f'{OUTPUT_PATH}/{PREFIX}_Boxplot_mu2_LearningCurve_{fit_type}_group.png'),
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return df_tau


def plot_mu2_learning_curve_boxplot_free_vs_fixed(df_parameters, ainf=CRITERION_10S_LOG_ASYMPTOTE, order=None,
                                                    PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot of raw mu2 per session, faceted by group, with two group-average
    learning curves overlaid (both fit per rat via fit_learning_curve_per_rat,
    then averaged across the rats of each group with errorbar='se'): one with
    Ainf freely fit, one with Ainf pinned at `ainf` (default
    CRITERION_10S_LOG_ASYMPTOTE, the 10s-criterion log asymptote).

    Returns (df_tau_free, df_tau_fixed): the per-rat A0/Ainf/tau tables from
    each fit, so the caller can feed them e.g. to
    plot_tau_free_vs_fixed_ainf_boxplot.
    """

    ALL_GROUPS = sorted(df_parameters['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    if order is None:
        order = sorted(df_parameters['session'].unique())
    order_arr = np.asarray(order, dtype=float)

    df_fit_curve_free, df_tau_free = fit_learning_curve_per_rat(df_parameters, ainf=None)
    df_fit_curve_fixed, df_tau_fixed = fit_learning_curve_per_rat(df_parameters, ainf=ainf)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(ALL_GROUPS):
        ax = axes[i]
        parameter = df_parameters.query('group == @group and session in @order')

        sns.boxplot(x='session', y='mu2', data=parameter, order=order,
                    ax=ax, width=0.5, showfliers=False,
                    boxprops=dict(alpha=0.3), color=COLOR_MAP[group])

        # cap both curve overlays to the displayed sessions -- otherwise a rat
        # with sessions beyond `order` would have those points clamped by
        # np.interp onto the last displayed position instead of being dropped
        curve_free = df_fit_curve_free.query('group == @group and session in @order').copy()
        if not curve_free.empty:
            curve_free['session_pos'] = np.interp(curve_free['session'], order_arr, np.arange(len(order_arr)))
            sns.lineplot(x='session_pos', y='mu2_fit', data=curve_free,
                        ax=ax, errorbar='se', color='black', lw=2,
                        label='Free $A_\\infty$')

        curve_fixed = df_fit_curve_fixed.query('group == @group and session in @order').copy()
        if not curve_fixed.empty:
            curve_fixed['session_pos'] = np.interp(curve_fixed['session'], order_arr, np.arange(len(order_arr)))
            sns.lineplot(x='session_pos', y='mu2_fit', data=curve_fixed,
                        ax=ax, errorbar='se', color='crimson', lw=2, ls='--',
                        label='Fixed $A_\\infty$')

        ax.set_ylabel('mu2')
        ax.set_xlabel('Session')
        ax.set_title(f'Group {group}')
        ax.legend(fontsize=8)

    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{PREFIX}: mu2 Learning Curve - Free vs Fixed $A_\\infty$', fontsize=16)
    fig.savefig(os.path.realpath(
        f'{OUTPUT_PATH}/{PREFIX}_Boxplot_mu2_LearningCurve_{fit_type}_free_vs_fixed_Ainf.png'),
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return df_tau_free, df_tau_fixed


def plot_tau_boxplot(df_tau, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot of the per-rat learning-curve tau (from fit_learning_curve_per_rat),
    one box per group."""

    ALL_GROUPS = sorted(df_tau['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='tau', data=df_tau, order=ALL_GROUPS, width=0.5,
                showfliers=False, boxprops=dict(alpha=0.3), palette=COLOR_MAP)
    sns.stripplot(x='group', y='tau', data=df_tau, order=ALL_GROUPS,
                  color='black', size=4, alpha=0.6)
    plt.xlabel('Groups')
    plt.ylabel(r'$\tau$ (sessions)')
    plt.title(f'{PREFIX}: Learning curve $\\tau$ per group')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_tau_{fit_type}_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def plot_ainf_boxplot(df_tau_free, criterion_line=CRITERION_10S_LOG_ASYMPTOTE,
                       PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot of the per-rat Ainf from the free-Ainf learning-curve fit
    (fit_learning_curve_per_rat, ainf=None -- e.g. the df_tau returned by
    plot_mu2_learning_curve_boxplot), one box per group. A dashed reference
    line is drawn at `criterion_line` (default CRITERION_10S_LOG_ASYMPTOTE =
    ln(10)) so it's visible whether each group's fitted asymptote lands
    above or below the criterion."""

    ALL_GROUPS = sorted(df_tau_free['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='Ainf', data=df_tau_free, order=ALL_GROUPS, width=0.5,
                showfliers=False, boxprops=dict(alpha=0.3), palette=COLOR_MAP)
    sns.stripplot(x='group', y='Ainf', data=df_tau_free, order=ALL_GROUPS,
                  color='black', size=4, alpha=0.6)
    if criterion_line is not None:
        plt.axhline(criterion_line, color='crimson', ls='--', lw=1.5,
                    label=r'$\ln(10)$ criterion')
        plt.legend(fontsize=8)
    plt.xlabel('Groups')
    plt.ylabel(r'$A_\infty$ (log-duration)')
    plt.title(f'{PREFIX}: Learning curve $A_\\infty$ per group (free fit)')
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_Ainf_{fit_type}_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def plot_sessions_to_criterion_boxplot(df_tau, target=CRITERION_10S_LOG_ASYMPTOTE, label='',
                                        PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot, per group, of the number of sessions a fitted learning curve
    predicts each rat needs to reach mu2 = `target` (default
    CRITERION_10S_LOG_ASYMPTOTE = ln(10), the log-duration value
    corresponding to the 10s criterion). `df_tau` is any per-rat table with
    rat/group/A0/Ainf/tau columns, e.g. the df_tau returned by
    fit_learning_curve_per_rat or plot_mu2_learning_curve_boxplot -- with
    Ainf either freely fit, or fixed at a value strictly above `target` (e.g.
    ainf=CRITERION_11S_LOG_ASYMPTOTE) so the curve actually crosses it in
    finite time (a fit with Ainf pinned AT or below the target never reaches
    it, see sessions_to_reach_criterion). Rats whose fitted curve never
    reaches the target are excluded and reported, and a table of them (rat,
    group) is returned alongside the main result.

    `label` is appended to the title/output filename, to tell apart calls
    that use different `df_tau` sources (e.g. free vs. a fixed-Ainf fit)."""

    df = df_tau.copy()
    df['sessions_to_criterion'] = df.apply(
        lambda row: sessions_to_reach_criterion(row['A0'], row['Ainf'], row['tau'], target), axis=1)

    df_excluded = df.loc[df['sessions_to_criterion'].isna(), ['rat', 'group']].reset_index(drop=True)
    if len(df_excluded) > 0:
        print(f'plot_sessions_to_criterion_boxplot{" (" + label + ")" if label else ""}: '
              f'{len(df_excluded)} rat(s) excluded (fitted learning curve never reaches the target):')
        print(df_excluded.to_string(index=False))

    df_valid = df.dropna(subset=['sessions_to_criterion'])

    ALL_GROUPS = sorted(df_valid['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='sessions_to_criterion', data=df_valid, order=ALL_GROUPS, width=0.5,
                showfliers=False, boxprops=dict(alpha=0.3), palette=COLOR_MAP)
    sns.stripplot(x='group', y='sessions_to_criterion', data=df_valid, order=ALL_GROUPS,
                  color='black', size=4, alpha=0.6)
    plt.xlabel('Groups')
    plt.ylabel('Sessions to reach criterion (predicted)')
    title_suffix = f' ({label})' if label else ''
    plt.title(f'{PREFIX}: Sessions to Reach mu2 = ln(10){title_suffix}')
    filename_suffix = f'_{label}' if label else ''
    plt.savefig(os.path.realpath(
        f'{OUTPUT_PATH}/{PREFIX}_Boxplot_SessionsToCriterion{filename_suffix}_{fit_type}_group.png'),
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return df, df_excluded


def plot_mu2_learning_curve_boxplot_by_criterion(df_parameters, raw_data, crit=10, order=None,
                                                   PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Alternative to plot_mu2_learning_curve_boxplot: within each group, splits rats
    into two subgroups -- those that reached the max criterion (`crit`, default 10s)
    at some point in `raw_data`, and those that never did -- and fits a separate
    per-rat learning curve (fit_learning_curve_per_rat) for each subgroup, overlaid
    on the mu2 boxplot (now dodged by subgroup).

    Returns df_tau: rat, group, criterion_subgroup, A0, Ainf, tau.
    """

    reached_label = f'Reached {crit}s'
    not_reached_label = f'Did not reach {crit}s'
    SUBGROUP_COLORS = {reached_label: '#5B9E95', not_reached_label: '#C5AA97'}

    reached_max_rats = raw_data.query('criterion >= @crit').rat.unique()

    df = df_parameters.copy()
    df['criterion_subgroup'] = np.where(df['rat'].isin(reached_max_rats), reached_label, not_reached_label)

    ALL_GROUPS = sorted(df['group'].unique())
    n_groups = len(ALL_GROUPS)
    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    if order is None:
        order = sorted(df['session'].unique())
    order_arr = np.asarray(order, dtype=float)

    df_fit_curve = pd.DataFrame()
    df_tau = pd.DataFrame()

    for subgroup_label in [reached_label, not_reached_label]:
        subgroup_data = df.query('criterion_subgroup == @subgroup_label')
        if subgroup_data.empty:
            continue
        fit_curve, tau = fit_learning_curve_per_rat(subgroup_data)
        fit_curve['criterion_subgroup'] = subgroup_label
        tau['criterion_subgroup'] = subgroup_label
        df_fit_curve = pd.concat([df_fit_curve, fit_curve], ignore_index=True)
        df_tau = pd.concat([df_tau, tau], ignore_index=True)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, group in enumerate(ALL_GROUPS):
        ax = axes[i]
        group_data = df.query('group == @group and session in @order')

        sns.boxplot(x='session', y='mu2', hue='criterion_subgroup', data=group_data, order=order,
                    hue_order=[reached_label, not_reached_label],
                    ax=ax, width=0.6, showfliers=False,
                    boxprops=dict(alpha=0.3), palette=SUBGROUP_COLORS)

        # cap the curve overlay to the displayed sessions -- otherwise a rat
        # with sessions beyond `order` would have those points clamped by
        # np.interp onto the last displayed position instead of being dropped
        curve_group = df_fit_curve.query('group == @group and session in @order').copy()
        if not curve_group.empty:
            curve_group['session_pos'] = np.interp(curve_group['session'], order_arr, np.arange(len(order_arr)))
            sns.lineplot(x='session_pos', y='mu2_fit', hue='criterion_subgroup',
                        hue_order=[reached_label, not_reached_label],
                        data=curve_group, ax=ax, errorbar='se', palette=SUBGROUP_COLORS,
                        lw=2, legend=False)

        ax.set_ylabel('mu2')
        ax.set_xlabel('Session')
        ax.set_title(f'Group {group}')
        ax.legend(fontsize=8, title=None)

    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{PREFIX}: mu2 Learning Curve by Criterion Subgroup', fontsize=16)
    fig.savefig(os.path.realpath(
        f'{OUTPUT_PATH}/{PREFIX}_Boxplot_mu2_LearningCurve_{fit_type}_by_criterion.png'),
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return df_tau


def plot_tau_boxplot_by_criterion(df_tau_by_criterion, crit=10, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot of the per-rat learning-curve tau (free Ainf), by group,
    dodged by criterion subgroup -- takes the df_tau returned by
    plot_mu2_learning_curve_boxplot_by_criterion (already has a
    criterion_subgroup column)."""

    reached_label = f'Reached {crit}s'
    not_reached_label = f'Did not reach {crit}s'
    SUBGROUP_COLORS = {reached_label: '#5B9E95', not_reached_label: '#C5AA97'}

    ALL_GROUPS = sorted(df_tau_by_criterion['group'].unique())

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='tau', hue='criterion_subgroup', data=df_tau_by_criterion,
                order=ALL_GROUPS, hue_order=[reached_label, not_reached_label],
                width=0.6, showfliers=False, boxprops=dict(alpha=0.5), palette=SUBGROUP_COLORS)
    sns.stripplot(x='group', y='tau', hue='criterion_subgroup', data=df_tau_by_criterion,
                  order=ALL_GROUPS, hue_order=[reached_label, not_reached_label],
                  dodge=True, color='black', size=3, alpha=0.5, legend=False)
    plt.xlabel('Groups')
    plt.ylabel(r'$\tau$ (sessions)')
    plt.title(f'{PREFIX}: Learning curve $\\tau$ by group and criterion subgroup')
    plt.legend(fontsize=8, title=None)
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_tau_by_criterion_{fit_type}_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def plot_ainf_boxplot_by_criterion(df_tau_by_criterion, crit=10, criterion_line=CRITERION_10S_LOG_ASYMPTOTE,
                                    PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot of the per-rat Ainf (free-Ainf fit), by group, dodged by
    criterion subgroup -- takes the df_tau returned by
    plot_mu2_learning_curve_boxplot_by_criterion (already has a
    criterion_subgroup column). A dashed reference line is drawn at
    `criterion_line` (default CRITERION_10S_LOG_ASYMPTOTE = ln(10))."""

    reached_label = f'Reached {crit}s'
    not_reached_label = f'Did not reach {crit}s'
    SUBGROUP_COLORS = {reached_label: '#5B9E95', not_reached_label: '#C5AA97'}

    ALL_GROUPS = sorted(df_tau_by_criterion['group'].unique())

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='Ainf', hue='criterion_subgroup', data=df_tau_by_criterion,
                order=ALL_GROUPS, hue_order=[reached_label, not_reached_label],
                width=0.6, showfliers=False, boxprops=dict(alpha=0.5), palette=SUBGROUP_COLORS)
    sns.stripplot(x='group', y='Ainf', hue='criterion_subgroup', data=df_tau_by_criterion,
                  order=ALL_GROUPS, hue_order=[reached_label, not_reached_label],
                  dodge=True, color='black', size=3, alpha=0.5, legend=False)
    if criterion_line is not None:
        plt.axhline(criterion_line, color='crimson', ls='--', lw=1.5, label=r'$\ln(10)$ criterion')
    plt.xlabel('Groups')
    plt.ylabel(r'$A_\infty$ (log-duration)')
    plt.title(f'{PREFIX}: Learning curve $A_\\infty$ by group and criterion subgroup')
    plt.legend(fontsize=8, title=None)
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_Ainf_by_criterion_{fit_type}_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def plot_tau_free_vs_fixed_ainf_boxplot(df_tau_free, df_tau_fixed, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Boxplot comparing tau from the free-Ainf learning-curve fit against
    tau from the fixed-Ainf fit (Ainf pinned at the 10s-criterion log
    asymptote, see CRITERION_10S_LOG_ASYMPTOTE), by group."""

    df_free = df_tau_free[['rat', 'group', 'tau']].copy()
    df_free['model_type'] = 'Free A_inf'

    df_fixed = df_tau_fixed[['rat', 'group', 'tau']].copy()
    df_fixed['model_type'] = 'Fixed A_inf'

    df_tau = pd.concat([df_free, df_fixed], ignore_index=True)
    ALL_GROUPS = sorted(df_tau['group'].unique())

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.boxplot(x='group', y='tau', hue='model_type', data=df_tau, order=ALL_GROUPS,
                hue_order=['Free A_inf', 'Fixed A_inf'],
                width=0.6, showfliers=False, boxprops=dict(alpha=0.5))
    plt.xlabel('Groups')
    plt.ylabel(r'$\tau$ (sessions)')
    plt.title(f'{PREFIX}: Learning curve $\\tau$ - Free vs Fixed $A_\\infty$')
    plt.legend(fontsize=8, title=None)
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_tau_free_vs_fixed_Ainf_{fit_type}_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


def plot_tau_all_vs_successful_boxplot(df_tau_all, df_tau_successful, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Side-by-side boxplots of tau by group: all animals (left) vs only the
    animals that reached the max criterion (right)."""

    ALL_GROUPS = sorted(set(df_tau_all['group']).union(df_tau_successful['group']))
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    for ax, (label, panel_data) in zip(axes, [('All animals', df_tau_all),
                                               ('Animals that reached criterion', df_tau_successful)]):
        sns.boxplot(x='group', y='tau', data=panel_data, order=ALL_GROUPS, ax=ax,
                    width=0.5, showfliers=False, boxprops=dict(alpha=0.3), palette=COLOR_MAP)
        sns.stripplot(x='group', y='tau', data=panel_data, order=ALL_GROUPS, ax=ax,
                      color='black', size=4, alpha=0.6)
        ax.set_xlabel('Groups')
        ax.set_ylabel(r'$\tau$ (sessions)')
        ax.set_title(label)

    plt.suptitle(f'{PREFIX}: Learning curve $\\tau$ - All vs Successful Animals', fontsize=14)
    fig.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Boxplot_tau_all_vs_successful_{fit_type}_group.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return


# ------- CRITERION-TRACKING INDEX: i = (C - mu2) / sigma2 -------

def calculate_criterion_tracking_index(df_parameters, raw_data, log=True):
    """Index i = (C(session) - mu2(session)) / sigma2(session), per rat/session:
    how many sigma2's the currently-active criterion C sits away from the mean
    of the long-response (Gaussian 2) component. i > 0: the response distribution
    hasn't caught up to the criterion yet; i near 0 (or negative): responses are
    tracking (or exceeding) the criterion. Watching this across sessions -- especially
    once a rat is holding at the 10s criterion -- shows whether the gap stabilizes.

    `df_parameters` -- one row per rat/session/group with mu2/sigma2 (e.g. from
    fit_params_all_sessions). `raw_data` -- the trial-level dataframe (same one the
    fit was run on) with a rat/session/group/criterion column; C is taken as the
    highest criterion the rat reached within that session (same definition as
    criterion_progression: data.groupby(['rat','session','group']).criterion.max()).

    `log` must match the `log` flag used to produce df_parameters: when True (the
    usual MLE fit), mu2/sigma2 are on the log-duration scale, so C is log-transformed
    before differencing to keep the numerator in the same units.

    Returns rat, group, session, criterion, mu2, sigma2, index.
    """

    df_criterion = raw_data.groupby(['rat', 'session', 'group']).criterion.max().reset_index()

    df_index = df_parameters[['rat', 'session', 'group', 'mu2', 'sigma2']].merge(
        df_criterion, on=['rat', 'session', 'group'], how='inner')

    c = np.log(df_index['criterion']) if log else df_index['criterion']
    df_index['index'] = (c - df_index['mu2']) / df_index['sigma2']

    return df_index


def plot_criterion_tracking_index_lineplot(df_index, order=None, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE'):
    """Lineplot (mean +/- SE) of the criterion-tracking index (see
    calculate_criterion_tracking_index) per session, one line per group. A
    dashed reference line at i = 0 marks where the long-response mean has
    caught up to the currently-active criterion."""

    ALL_GROUPS = sorted(df_index['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    if order is None:
        order = sorted(df_index['session'].unique())

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x='session', y='index', hue='group', data=df_index, hue_order=ALL_GROUPS,
                 palette=COLOR_MAP, errorbar='se')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
    plt.xlabel('Session')
    plt.ylabel(r'Criterion-tracking index $i = (C - \mu_2)/\sigma_2$')
    plt.title(f'{PREFIX}: Criterion-Tracking Index per Session')
    plt.savefig(os.path.realpath(
        f'{OUTPUT_PATH}/{PREFIX}_Lineplot_CriterionTrackingIndex_{fit_type}_group.png'),
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    
    PREFIX = 'AU'
    DATA_PATH = os.path.realpath('../../data/processed/')+'/'
    OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_10s_Groups/')+'/'
    LAST_SESSION = 19
    ANIMALS = list(range(15,27)) 
    
    data = pd.read_csv(DATA_PATH +f'{PREFIX}.csv')

    data = data.astype({'rat':int,'session':int,'reinforced':int})
    #data = data.query('duration < 20').reset_index(drop=True)

    ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    df = data.copy()

    early = [1,2,3]

    late = [17, 18, 19]

    df_early = df.loc[(df['group'] == 'LINEAR') & (df['session'].isin(early))].reset_index(drop=True)
    
    df_late = df.loc[(df['group'] == 'LINEAR') & (df['session'].isin(late))].reset_index(drop=True)
    

    plot_fitting_lsf_rats(df_early, session = str(early), log = True, \
                                                  PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
    
    plot_fitting_mle_rats(df_early, session = str(early), log = True, \
                                                  PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
    
    plot_fitting_lsf_rats(df_late, session = str(late), log = True, \
                                                  PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
    
    plot_fitting_mle_rats(df_late, session = str(late), log = True, \
                                                PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

    # ---- Gamma across all sessions (LINEAR group only) ----

    df_linear = df.query('group == "LINEAR"').reset_index(drop=True)

    df_parameters_mle = fit_params_all_sessions(df_linear, MLE, log=True)
    df_parameters_lsf = fit_params_all_sessions(df_linear, LSF, log=True)

    plot_param_method_comparison_boxplot(df_parameters_mle, df_parameters_lsf, 'gamma', \
                                          PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

    plot_gamma_vs_gamma2_boxplot(df_parameters_mle, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

    