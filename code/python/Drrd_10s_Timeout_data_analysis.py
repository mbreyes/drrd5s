#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""

#%%
import drrdTools as dr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import drrd_functions as drrd
from scipy.stats import pearsonr
import os
import model_fitting as mf

plt.rcParams['figure.dpi'] = 500

# plt.style.use('ggplot')
# sns.set(style='ticks')

# sns.set_palette(['#F2A65A', '#772F1A'])
sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

DATA_PATH = '../../data/processed/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_10s_Timeout/')+'/'
PREFIX = 'DRRD 10s with Timeout'
PREFIXES = ['AY4','AY5','AZ4','BB5']
LAST_SESSION = 6

#%%
#---- LOADING DATA ----
data = pd.concat(
    [pd.read_csv(DATA_PATH + f'{prefix}.csv') for prefix in PREFIXES],
    ignore_index=True
)

data = data.astype({'rat':int,'session':int,'reinforced':int})

data_without_ay5 = pd.concat(
    [pd.read_csv(DATA_PATH + f'{prefix}.csv') for prefix in PREFIXES if prefix != 'AY5'],
    ignore_index=True
)

data_without_ay5 = data_without_ay5.astype({'rat':int,'session':int,'reinforced':int})

# ---- CHECKING EVOLUTION OF BEHAVIOR OVER SESSIONS ----

mf.proportion_correct_responses(data, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
mf.proportion_correct_responses(data_without_ay5, PREFIX = PREFIX + ' (without AY5)', OUTPUT_PATH = OUTPUT_PATH)

mf.mean_response_duration(data, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)    
mf.mean_response_duration(data_without_ay5, PREFIX = PREFIX + ' (without AY5)', OUTPUT_PATH = OUTPUT_PATH)

mf.boxplot_first_session_early_late(data, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
mf.boxplot_first_session_early_late(data_without_ay5, PREFIX = PREFIX + ' (without AY5)', OUTPUT_PATH = OUTPUT_PATH)

# %%

# ------- Analysing only the independent data between groups (without AY5) ---------

# ------- MLE Fitting of Double Gaussian Distribution to Response Duration Data ----

for group in data_without_ay5.group.unique():
    mf.plot_fitting_mle_rats(data_without_ay5.query('group == @group and session == 1'), session = '1', log = True, PREFIX = PREFIX + f' ({group})', OUTPUT_PATH = OUTPUT_PATH)
    mf.plot_fitting_mle_rats(data_without_ay5.query('group == @group and session == 6'), session = '6', log = True, PREFIX = PREFIX + f' ({group})', OUTPUT_PATH = OUTPUT_PATH)

#%%
df_parameters = mf.fit_params_all_sessions(data_without_ay5, fit_function = mf.MLE, log = True)

mf.plot_parameters_boxplot(df_parameters, order = np.arange(1, LAST_SESSION+1), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#%%
# ------- Lineplot with all groups on the same axis (hue = group) --------

mf.plot_parameters_lineplot_combined(df_parameters, order = np.arange(1, LAST_SESSION+1), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
# %%


# ---- Comparing with baseline ------

PREFIXES = ['AY3', 'AZ3', 'BB4']
LAST_SESSION = 5

baseline = pd.concat(
    [pd.read_csv(DATA_PATH + f'{prefix}.csv') for prefix in PREFIXES],
    ignore_index=True
)

baseline = baseline.astype({'rat':int,'session':int,'reinforced':int})

rat_to_group = data_without_ay5.drop_duplicates(subset='rat').set_index('rat')['group']
baseline['group'] = baseline['rat'].map(rat_to_group)
baseline = baseline.dropna(subset=['group']).reset_index(drop=True)

#%%
baseline_parameters = mf.fit_params_all_sessions(baseline, fit_function = mf.MLE, log = True)

mf.plot_parameters_boxplot(baseline_parameters, order = np.arange(1, LAST_SESSION+1), PREFIX = PREFIX + ' Baseline', OUTPUT_PATH = OUTPUT_PATH)

for group in baseline.group.unique():
    mf.plot_fitting_mle_rats(baseline.query('group == @group and session == 1'), session = '1', log = True, PREFIX = PREFIX + f' ({group})', OUTPUT_PATH = OUTPUT_PATH)
    mf.plot_fitting_mle_rats(baseline.query('group == @group and session == 5'), session = '6', log = True, PREFIX = PREFIX + f' ({group})', OUTPUT_PATH = OUTPUT_PATH)


# %%
mf.proportion_correct_responses(baseline, PREFIX = PREFIX + ' Baseline', OUTPUT_PATH = OUTPUT_PATH)
mf.mean_response_duration(baseline, PREFIX = PREFIX + ' Baseline', OUTPUT_PATH = OUTPUT_PATH)

#%%
# ------- Comparing proportion of correct responses: Baseline vs Timeout --------

timeout_labeled = data_without_ay5.copy()
timeout_labeled['condition'] = 'Timeout'

baseline_labeled = baseline.copy()
baseline_labeled['condition'] = 'Baseline'

baseline_vs_timeout = pd.concat([baseline_labeled, timeout_labeled], ignore_index=True)

mf.proportion_correct_responses_by_condition(baseline_vs_timeout, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
mf.proportion_correct_responses_by_condition_combined(baseline_vs_timeout, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#%%
# ------- Comparing mean response duration: Baseline vs Timeout --------

mf.mean_response_duration_by_condition(baseline_vs_timeout, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
mf.mean_response_duration_by_condition_combined(baseline_vs_timeout, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# %%

mf.plot_parameters_lineplot_combined(baseline_parameters, order = np.arange(1, LAST_SESSION+1), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#%%
# ------- mu2 across Baseline (5 sessions) + Timeout (6 sessions), as a continuation, hue = group --------

mf.plot_parameter_continuous_baseline_timeout(baseline_parameters, df_parameters, param = 'mu2',
                                               baseline_last_session = 5, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#%%
# ------- Boxplot of mu2 and gamma_2: Baseline (sessions 1, 5) vs Timeout (sessions 1, 6) --------

mf.plot_parameters_boxplot_baseline_timeout_key_sessions(baseline_parameters, df_parameters,
                                                          params = ['mu2', 'gamma_2'],
                                                          baseline_sessions = [1, 5], timeout_sessions = [1, 6],
                                                          PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# %%
mf.plot_parameter_continuous_baseline_timeout(baseline_parameters, df_parameters, param = 'sigma2',
                                               baseline_last_session = 5, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# %%
mf.plot_parameter_continuous_baseline_timeout(baseline_parameters, df_parameters, param = 'mu1',
                                               baseline_last_session = 5, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# %%
mf.plot_parameter_continuous_baseline_timeout(baseline_parameters, df_parameters, param = 'sigma1',
                                               baseline_last_session = 5, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
# %%
mf.plot_parameter_continuous_baseline_timeout(baseline_parameters, df_parameters, param = 'gamma',
                                               baseline_last_session = 5, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# %%
mf.plot_parameter_continuous_baseline_timeout(baseline_parameters, df_parameters, param = 'gamma_2',
                                               baseline_last_session = 5, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# %%
