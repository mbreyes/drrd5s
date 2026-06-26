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


#%%

#-------- DISTRIBUTION OF RESPONSE DURATIONS ---------

# -------- Fitting a Double Gaussian using Maximum Likelihood Estimation (MLE) ---------


# DRRD 2s - EARLY AND LATE SESSIONS

init_params = [0.5, 0.5, 0.1, 2.5, 0.5]

data_initial = data_phase_1.query('session <= 2')
    
parameters_initial = mf.plot_fitting_mle(data = data_initial, subtitle = '(MLE Fit) - DRRD 2s', init_params = init_params, session = 'early')

data_final = mf.mask_last(data_phase_1, number_sessions = 2)

parameters_final = mf.plot_fitting_mle(data = data_final, init_params = init_params, subtitle = '(MLE Fit) - DRRD 2s', session = 'late')

df_parameters_1 = pd.concat([parameters_initial, parameters_final], ignore_index=True)

mf.plot_parameters_mle(df_parameters_1, experiment = 'DRRD 2s', PREFIX= PREFIX, OUTPUT_PATH=OUTPUT_PATH)
# %%

# DRRD 2s retract - EARLY AND LATE SESSIONS 

init_params = [0.5, 0.5, 0.1, 2.5, 0.5]

data_initial = data_phase_2.query('session <= 2')
    
parameters_initial = mf.plot_fitting_mle(data = data_initial, init_params = init_params, subtitle = '(MLE Fit) - DRRD 2s retract', session = 'early')

data_final = mf.mask_last(data_phase_2, number_sessions = 2)

parameters_final = mf.plot_fitting_mle(data = data_final, init_params = init_params, subtitle = '(MLE Fit) - DRRD 2s retract', session = 'late')

df_parameters_2 = pd.concat([parameters_initial, parameters_final], ignore_index=True)

mf.plot_parameters_mle(df_parameters_2, experiment = 'DRRD 2s retract', PREFIX= PREFIX, OUTPUT_PATH=OUTPUT_PATH)


#%%
# DRRD precision 3s - EARLY AND LATE SESSIONS

init_params = [0.5, 0.5, 0.1, 2.5, 0.5]

data_initial = data_phase_3.query('session <= 2')
    
parameters_initial = mf.plot_fitting_mle(data = data_initial, init_params = init_params, subtitle = '(MLE Fit) - DRRD precision 3s', session = 'early')

data_final = mf.mask_last(data_phase_3, number_sessions = 2)

parameters_final = mf.plot_fitting_mle(data = data_final, init_params = init_params, subtitle = '(MLE Fit) - DRRD precision 3s', session = 'late')

df_parameters_3 = pd.concat([parameters_initial, parameters_final], ignore_index=True)

mf.plot_parameters_mle(df_parameters_3, experiment = 'DRRD precision 3s', PREFIX= PREFIX, OUTPUT_PATH=OUTPUT_PATH)


# %%

# DRRD precision 2.5s - EARLY AND LATE SESSIONS

init_params = [0.5, 0.5, 0.1, 2.5, 0.5]

data_initial = data_phase_4.query('session <= 2')

parameters_initial = mf.plot_fitting_mle(data = data_initial, init_params = init_params, subtitle = '(MLE Fit) - DRRD precision 2.5s', session = 'early')

data_final = mf.mask_last(data_phase_4, number_sessions = 2)

parameters_final = mf.plot_fitting_mle(data = data_final, init_params = init_params, subtitle = '(MLE Fit) - DRRD precision 2.5s', session = 'late')

df_parameters_4 = pd.concat([parameters_initial, parameters_final], ignore_index=True)

mf.plot_parameters_mle(df_parameters_4, experiment = 'DRRD precision 2.5s', PREFIX= PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- COMPARING PARAMETERS ACROSS PHASES ---------

for param in ['mu1', 'sigma1', 'mu2', 'sigma2', 'gamma']:
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15,10))
    axes = axes.flatten()

    for i, df in enumerate([df_parameters_1, df_parameters_2, df_parameters_3, df_parameters_4]):

        sns.pointplot(data=df, x='session', y=param, hue='rat', ax=axes[i], palette='Set2', dodge=0.5, markers='o', linestyles='-')
        sns.boxplot(data=df, x='session', y=param, ax=axes[i], color = 'lightgray', showfliers=False, width=0.5, dodge=True,  boxprops=dict(alpha=0.3))
        axes[i].set_title(f'Fases {i+1}')
        axes[i].set_xlabel('Session')
        axes[i].set_ylabel(param)
        axes[i].legend(title='Rat')

    plt.tight_layout()
    plt.suptitle(f'Comparison of {param} across Phases', y=1.02, fontsize=16)
    plt.show()
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_{param}_Comparison.png'))


# -------- Fitting a Double Gaussian using Least Squares ---------



# %%
