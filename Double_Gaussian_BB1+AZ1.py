#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 17:28:18 2025

@author: ailacamara
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import os
import pingouin as pg

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = "AZ1 and BB1"
PREFIXES = ['AZ1', 'BB1']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/AZ+BB/')+'/'
ANIMALS = list(range(87,93)) + list(range(105,117))
list_data = []

GROUPS = ['no_timeout','retract','random']
# Defining functions to plot graphs
for prefix in PREFIXES:
     
    data    = pd.read_csv(DATA_PATH +f'{prefix}.csv')
    data    = data.astype({'rat':int,'session':int,'reinforced':int, 'group':str})
    list_data.append(data)

data_all = pd.concat(list_data, ignore_index = True)
df = data_all.copy()
df = df.groupby(['rat', 'session', 'group'], group_keys = True)\
            .apply(lambda x : pd.Series(dict(nresps = (x.duration).count(),  
                        frac_correct = ((x.reinforced).sum()/(x.duration).count()), \
                        duration_mean = (x.duration).mean(), 
                        last_criterion = (x.criterion).max()))).reset_index()
                
columns = ['nresps', 'frac_correct','duration_mean'] 

data_histogram = data_all.query('rat == 90 and session == 2 and duration <=7.5').duration


# Binning
bin_width = 0.05
bins = np.arange(0, 4, bin_width)
hist_counts, bin_edges = np.histogram(data_histogram, bins=bins)
hist_density = hist_counts / (np.sum(hist_counts) * bin_width)
bin_centers = bin_edges[:-1] + bin_width/2

# Model
def P_t(t, params):
    gamma, mu1, sigma1, mu2, sigma2 = params
    g1 = np.exp(-((t - mu1)**2) / (2 * sigma1**2))
    g2 = np.exp(-((t - mu2)**2) / (2 * sigma2**2))
    f_t = (1 - gamma)*g1 + gamma*g2
    xi = np.sum(f_t) * bin_width
    return f_t / xi

# Objective
def objective(params):
    gamma, mu1, sigma1, mu2, sigma2 = params
    if not (0 <= gamma <= 1) or sigma1 <=0 or sigma2 <=0:
        return np.inf
    model = P_t(bin_centers, params)
    return np.sum((hist_density - model)**2)

# Constraint function: mu2 - mu1 >= 0
def mu_order_constraint(params):
    return params[3] - params[1]  # mu2 - mu1

# NonlinearConstraint object
mu_constraint = NonlinearConstraint(mu_order_constraint, 0, np.inf)

# Initial guess
init_params = [
    0.5,  # gamma
    5.0,  # mu1
    0.3,  # sigma1
    5.0,  # mu2
    0.5   # sigma2
]

# Bounds
bounds = [
    (0,1),       # gamma
    (None,None), # mu1
    (1e-6,None), # sigma1
    (None,None), # mu2
    (1e-6,None)  # sigma2
]

# Fit with constraint
result = minimize(
    objective,
    init_params,
    bounds=bounds,
    constraints=[mu_constraint],
    options={'disp': True}
)

# Extract fitted parameters
gamma, mu1, sigma1, mu2, sigma2 = result.x

print("Fitted parameters:")
print(f"gamma = {gamma:.3f}")
print(f"mu1 = {mu1:.3f}")
print(f"sigma1 = {sigma1:.3f}")
print(f"mu2 = {mu2:.3f}")
print(f"sigma2 = {sigma2:.3f}")

# Compute components
g1 = np.exp(-((bin_centers - mu1)**2) / (2 * sigma1**2))
g2 = np.exp(-((bin_centers - mu2)**2) / (2 * sigma2**2))
f_t = (1 - gamma)*g1 + gamma*g2
xi = np.sum(f_t)*bin_width
P = f_t/xi
P1 = ((1 - gamma)*g1)/xi
P2 = (gamma*g2)/xi

# Parameter text
param_text = (
    f"$\\gamma$ = {gamma:.3f}\n"
    f"$\\mu_1$ = {mu1:.3f}\n"
    f"$\\sigma_1$ = {sigma1:.3f}\n"
    f"$\\mu_2$ = {mu2:.3f}\n"
    f"$\\sigma_2$ = {sigma2:.3f}"
)

# Plot
plt.figure(figsize=(10,6))
plt.bar(bin_centers, hist_density, width=bin_width, alpha=0.4, label='Normalized Histogram')
plt.plot(bin_centers, P, 'r-', lw=2, label='Fitted P(t)')
plt.plot(bin_centers, P1, 'g--', lw=2, label='Component 1')
plt.plot(bin_centers, P2, 'b--', lw=2, label='Component 2')
plt.xlabel('Time (s)')
plt.ylabel('Probability Density')
plt.title('Double Gaussian Fit -  Rat 111')
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
