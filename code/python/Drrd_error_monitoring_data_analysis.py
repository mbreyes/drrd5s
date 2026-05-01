#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""
#%%

import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import model_fitting as mf
import matplotlib.pyplot as plt
import os
import pingouin as pg
from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize import curve_fit
from scipy.stats import shapiro

from fitter import Fitter, get_common_distributions, get_distributions

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIXES = ['BD2', 'BD3', 'BD4', 'BD5']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_Error_Monitoring/')+'/'
ANIMALS = range(117,123)
PREFIX = 'DRRD_Error_Monitoring'

#%%

list_of_dfs = []
for i in range(len(PREFIXES)):
    list_of_dfs.append(None)

#---- LOADING DATA ----
for i, prefix in enumerate(PREFIXES):
    list_of_dfs[i] = pd.read_csv(DATA_PATH +f'{prefix}.csv')
    list_of_dfs[i] = list_of_dfs[i].astype({'rat':int,'session':int,'reinforced':int})


# ---- ANALYSING DATA FROM PHASE 1 - DRRD 2s  -----

data_phase_1 = list_of_dfs[0].reset_index(drop=True)

# ----- MEAN RESPONSE DURATION  AND PROPORTION OF CORRECT RESPONSES -----

mf.mean_response_duration(data_phase_1, 'DRRD 2s', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)

mf.proportion_correct_responses(data_phase_1, 'DRRD 2s', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)


# ----- ANALYSING DATA FROM PHASE 2 - DRRD 2s retract  -----

data_phase_2 = list_of_dfs[1].reset_index(drop=True)

# ----- MEAN RESPONSE DURATION  AND PROPORTION OF CORRECT RESPONSES -----

mf.mean_response_duration(data_phase_2, 'DRRD retract', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)

mf.proportion_correct_responses(data_phase_2, 'DRRD retract', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)


#-------- ANALYSING DATA FROM PHASE 3 - DRRD precision 3s  -----

data_phase_3 = list_of_dfs[2].reset_index(drop = True)


# ----- MEAN RESPONSE DURATION -----

mf.mean_response_duration(data_phase_3, 'DRRD precision 3s', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)

mf.proportion_correct_responses(data_phase_3, 'DRRD precision 3s', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)


#-------- ANALYSING DATA FROM PHASE 4 - DRRD precision 2.5s  -----

data_phase_4 = list_of_dfs[3].reset_index(drop = True)


# ----- MEAN RESPONSE DURATION -----

mf.mean_response_duration(data_phase_4, 'DRRD precision 2.5s', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)

mf.proportion_correct_responses(data_phase_4, 'DRRD precision 2.5s', OUTPUT_PATH=OUTPUT_PATH, PREFIX=PREFIX)



#-------- DISTRIBUTION OF RESPONSE DURATIONS ---------

# DRRD 2s - EARLY AND LATE SESSIONS

for rat in data_phase_1.rat.unique():
    plt.hist(data_phase_1.query(f'rat == {rat} and session == 1').duration, \
             bins=n_bins(data_phase_1.query(f'rat == {rat} and session == 1').duration, bin_width=0.1), density=True)
    LAST_SESSION = data_phase_1.query('rat == @rat').session.max()
    plt.hist(data_phase_1.query(f'rat == {rat} and session == {LAST_SESSION}').duration, \
             bins=n_bins(data_phase_1.query(f'rat == {rat} and session == {LAST_SESSION}').duration, bin_width=0.1), density=True)
    plt.xlabel('Response Duration')
    plt.ylabel('Density')
    plt.title(f'Rat {rat}: Response Duration Distribution - Session 1 vs Session {LAST_SESSION}')
    plt.legend(['Session 1', f'Session {LAST_SESSION}'])
    plt.show()
    #plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Duration_Distribution_Rat{rat}.png'))
    
# %%

# DRRD 2s (Late sessions) vs DRRD 2s retract (Late sessions) 

for rat in data_phase_1.rat.unique():
    LAST_SESSION = data_phase_2.query('rat == @rat').session.max()
    plt.hist(data_phase_2.query(f'rat == {rat} and session == {LAST_SESSION}').duration, \
             bins=n_bins(data_phase_2.query(f'rat == {rat} and session == {LAST_SESSION}').duration, bin_width=0.1), density=True)
    LAST_SESSION = data_phase_1.query('rat == @rat').session.max()
    plt.hist(data_phase_1.query(f'rat == {rat} and session == {LAST_SESSION}').duration, \
             bins=n_bins(data_phase_1.query(f'rat == {rat} and session == {LAST_SESSION}').duration, bin_width=0.1), density=True)
    plt.xlabel('Response Duration')
    plt.ylabel('Density')
    plt.title(f'Rat {rat}: Response Duration Distribution - DRRD 2s vs DRRD retract')
    plt.legend([f'Session {LAST_SESSION}', 'Session 1'])
    plt.show()


# %%
# DRRD PRECISION 3s - LATE SESSIONS

# for rat in data_phase_3.rat.unique():
#     sns.set_style('white')
#     sns.set_context("paper", font_scale = 2)
#     LAST_SESSION = data_phase_3.query('rat == @rat').session.max()
#     data = data_phase_3.query(f"rat == {rat} and session == {LAST_SESSION}").duration.reset_index()
#     sns.displot(data, x = 'duration', kind = 'hist', bins='auto', aspect=1.5)
#     plt.xlabel('Duration (s)')
#     plt.ylabel('Count')
#     plt.title(f'Rat {rat} DRRD precision 3: Session {LAST_SESSION} - Response Duration Distribution')
#     duration = data['duration'].values
#     f = Fitter(duration, distributions = ['gama', 'lognorm', 'beta', 'burr', 'norm'])
#     f.fit()
#     f.summary()

# DRRD 2s retract vs DRRD precision 3s (Late sessions)


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,8))

# A 2x2 grid returns a 2D array of axes. Flattening it makes it a simple 1D list,
# which is much easier to loop through!
axes = axes.flatten()

for i, rat in enumerate(data_phase_3.rat.unique()):

    ax = axes[i]

    #Plot the histograms in this specific 'ax'
    LAST_SESSION_1 = data_phase_3.query(f'rat == {rat}').session.max()
    ax.hist(data_phase_3.query(f'rat == {rat} and session == {LAST_SESSION_1}').duration, \
            bins=n_bins(data_phase_3.query(f'rat == {rat} and session == {LAST_SESSION_1}').duration, bin_width=0.1), alpha=0.5, label=f'Session {LAST_SESSION_1}', density=True, color = 'blue')
    LAST_SESSION_2 = data_phase_2.query(f'rat == {rat}').session.max()
    ax.hist(data_phase_2.query(f'rat == {rat} and session == {LAST_SESSION_2}').duration, \
            bins=n_bins(data_phase_2.query(f'rat == {rat} and session == {LAST_SESSION_2}').duration, bin_width=0.1), alpha = 0.5, label=f'Session {LAST_SESSION_2}', density=True, color = 'orange')
    ax.set_xlabel('Response Duration')
    ax.set_ylabel('Density')
    ax.set_title(f'Rat {rat}')
    ax.legend(['Precision 3s', 'Retract 2s'])

plt.suptitle('Response Duration Distribution - DRRD precision 3s vs DRRD retract', fontsize=16)
plt.tight_layout() # This prevents titles and labels from overlapping
plt.show()


# DRRD precision 3s vs DRRD precision 2.5s (Late sessions)

for rat in data_phase_4.rat.unique():
    last_4 = data_phase_4.query(f'rat == {rat}').session.max()
    plt.hist(data_phase_4.query(f'rat == {rat} and session == {last_4}').duration, \
             bins=n_bins(data_phase_4.query(f'rat == {rat} and session == {last_4}').duration, bin_width=0.1), density=True, alpha=0.5)
    last_3 = data_phase_3.query(f'rat == {rat}').session.max()
    plt.hist(data_phase_3.query(f'rat == {rat} and session == {last_3}').duration, \
             bins=n_bins(data_phase_3.query(f'rat == {rat} and session == {last_3}').duration, bin_width=0.1), density=True, alpha=0.5)
    plt.xlabel('Response Duration')
    plt.ylabel('Density')
    plt.title(f'Rat {rat}: Response Duration Distribution - DRRD precision 3s vs DRRD retract')
    plt.legend([f'Session {last_4}', f'Session {last_3}'])
    plt.show()

#%%
# --------- FITTING A DOUBLE GAUSSIAN TO THE RESPONSE DURATION DISTRIBUTION ---------


# -------- Fitting a Double Gaussian using Maximum Likelihood Estimation (MLE) ---------

raw_data = data_phase_4.query('rat == 117 and session == 1').duration.values

# ----- DEFINING FUNCTIONS -----    


# 1. Define the Normalized Gaussian Mixture (True PDF)
def gaussian_pdf(x, mu, sigma):
    """Standard normalized single Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)/ (sigma * np.sqrt(2 * np.pi))

def mixture_pdf(x, gamma, mu1, sigma1, mu2, sigma2):
    """The normalized Double Gaussian Mixture."""
    return gamma * gaussian_pdf(x, mu1, sigma1) + (1 - gamma) * gaussian_pdf(x, mu2, sigma2)

# 2. Define the Objective Function (Negative Log-Likelihood)
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

# Constraint function: mu2 - mu1 >= 0
def mu_order_constraint(params):
    return params[3] - params[1]  # mu2 - mu1

def data_histogram(data, bin_width=0.01):

    bins = np.arange(min(data), max(data) + bin_width, bin_width)
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    hist_density = counts / (np.sum(counts) * bin_width)
    bin_centers = bin_edges[:-1] + bin_width/ 2
    return counts, bin_centers, bin_edges, hist_density

# NonlinearConstraint object
mu_constraint = NonlinearConstraint(mu_order_constraint, 0, np.inf)
# ==========================================
# 3. LOAD YOUR RAW DATA
# Using the same bimodal dummy data as before

# ==========================================

# 4. Set Initial Guesses and Bounds
# Order: [gamma, mu1, sigma1, mu2, sigma2]
initial_guesses = [0.5, 0.5, 0.1, 2.5, 0.5]

# Bounds are CRITICAL for MLE to prevent impossible math
bounds = [
    (0.01, 0.99),       # Gamma must be between 0 and 1 (avoiding exact 0 or 1)
    (None, None),       # mu1 can be anything
    (0.01, None),       # sigma1 must be positive
    (None, None),       # mu2 can be anything
    (0.01, None)        # sigma2 must be positive
]

# 5. Run the Optimization
# We minimize the NLL function, passing our raw_data to it
result = minimize(
    negative_log_likelihood, 
    initial_guesses, 
    args=(raw_data,), 
    bounds=bounds, 
    constraints=[mu_constraint],
    options={'disp': True},
    method='L-BFGS-B' # A highly efficient algorithm for bounded problems
)

# Unpack the results
fit_gamma, fit_mu1, fit_sigma1, fit_mu2, fit_sigma2 = result.x

counts, bin_centers, bin_edges, hist_density = data_histogram(raw_data, bin_width=0.05)
bin_width = 0.05
# 6. Calculate peak heights and your custom Gamma
# Note: In MLE, gamma represents the proportion of *area* (total trials).
# If you still specifically need the ratio of *peak heights*, calculate them:
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

# ==========================================
# 7. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 6))

# IMPORTANT: Because MLE calculates a PDF (area=1), we MUST plot the 
# histogram with density=True so their y-axes match.


plt.figure(figsize=(10,6))
plt.bar(bin_centers, hist_density, width=bin_width, alpha=0.4, label='Normalized Histogram')
plt.plot(bin_centers, P_t, 'r-', lw=2, label='Fitted P(t)')
plt.plot(bin_centers, P1, 'g--', lw=2, label='Component 1')
plt.plot(bin_centers, P2, 'b--', lw=2, label='Component 2')
plt.xlabel('Time (s)')
plt.ylabel('Probability Density')
plt.title('Double Gaussian (MLE Fit) - Rat 117')
plt.legend()

plt.text(
    0.65, 0.95, param_text,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.show()


# %%
