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
import drrd_functions as drrd
import matplotlib.pyplot as plt
import os
import pingouin as pg
from scipy.optimize import curve_fit
from scipy.stats import shapiro
from fitter import Fitter, get_common_distributions, get_distributions

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIXES = ['BD2', 'BD3', 'BD4']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_Error_Monitoring/')+'/'
LAST_SESSION = 4
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

# ----- MEAN RESPONSE DURATION -----

df_mean = data_phase_1.groupby(['rat', 'session']).duration.mean().reset_index()
plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='rat', data=df_mean)
plt.xlabel('Session')
plt.ylabel('Mean Response Duration Phase 1')
plt.title('DRRD 2s: Mean Response Duration per Session')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_Response_Duration_Phase_1.png'))


# ------ % OF CORRECT RESPONSES -----

df_correct = data_phase_1.groupby(['rat', 'session']).reinforced.sum().reset_index()
df_tot = data_phase_1.groupby(['rat', 'session']).reinforced.count().reset_index()
df_correct['correct'] = df_correct['reinforced'] / df_tot['reinforced']

plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='correct', hue='rat', data=df_correct)
plt.xlabel('Session')
plt.ylabel('Proportion of Correct Responses')
plt.title('DRRD 2s: Proportion of Correct Responses per Session')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_Phase_1.png'))

#%%
# ----- ANALYSING DATA FROM PHASE 2 - DRRD 2s retract  -----

data_phase_2 = list_of_dfs[1].reset_index(drop=True)

# ----- MEAN RESPONSE DURATION -----

df_mean = data_phase_2.groupby(['rat', 'session']).duration.mean().reset_index()
plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='rat', data=df_mean)
plt.xlabel('Session')
plt.ylabel('Mean Response Duration Phase 2')
plt.title('DRRD 2s retract: Mean Response Duration per Session')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_Response_Duration_Phase_2.png'))

# ------ % OF CORRECT RESPONSES -----

df_correct = data_phase_2.groupby(['rat', 'session']).reinforced.sum().reset_index()
df_tot = data_phase_2.groupby(['rat', 'session']).reinforced.count().reset_index()
df_correct['correct'] = df_correct['reinforced'] / df_tot['reinforced']
plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='correct', hue='rat', data=df_correct)
plt.xlabel('Session')
plt.ylabel('Proportion of Correct Responses')
plt.title('DRRD 2s retract: Proportion of Correct Responses per Session')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_Phase_2.png'))
#%%

#-------- FIT DOUBLE GAUSSIAN TO LOG OF DURATION DISTRIBUTION ---------

# BY GROUP EARLY AND LATE SESSIONS

for rat in data_phase_1.rat.unique():
    plt.hist(data_phase_1.query(f'rat == {rat} and session == 1').duration, bins=100, density=True)
    plt.hist(data_phase_1.query(f'rat == {rat} and session == {LAST_SESSION}').duration, bins=100, density=True, alpha=0.5)
    plt.xlabel('Response Duration')
    plt.ylabel('Density')
    plt.title(f'Rat {rat}: Response Duration Distribution - Session 1 vs Session {LAST_SESSION}')
    plt.legend(['Session 1', f'Session {LAST_SESSION}'])
    plt.show()
    #plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Duration_Distribution_Rat
    
# %%

# BY GROUP EARLY AND LATE SESSIONS

for rat in data_phase_1.rat.unique():
    plt.hist(data_phase_2.query(f'rat == {rat} and session == 1').duration, bins=100, density=True, alpha=0.5)
    plt.hist(data_phase_1.query(f'rat == {rat} and session == {LAST_SESSION}').duration, bins=80, density=True)
    plt.xlabel('Response Duration')
    plt.ylabel('Density')
    plt.title(f'Rat {rat}: Response Duration Distribution - DRRD 2s vs DRRD retract')
    plt.legend([f'Session {LAST_SESSION}', 'Session 1'])
    plt.show()
    #plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Duration_Distribution_Rat
    
# %%

data_phase_3 = list_of_dfs[2].reset_index(drop = True)


df_mean = data_phase_3.groupby(['session', 'rat']).duration.mean().reset_index()

plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='rat', data=df_mean)
plt.xlabel('Session')
plt.ylabel('Duration Mean')
plt.title('DRRD precision 3s: Duration mean per Session')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Proportion_Correct_Responses_Phase_2.png'))

#%%

for rat in data_phase_3.rat.unique():
    sns.set_style('white')
    sns.set_context("paper", font_scale = 2)
    data = data_phase_3.query(f"rat == {rat} and session == 2").duration.reset_index()
    sns.displot(data, x = 'duration', kind = 'hist', bins=80, aspect=1.5)
    plt.xlabel('Duration (s)')
    plt.ylabel('Count')
    plt.title(f'Rat {rat} DRRD precision 3: Session 2')
    
    duration = data['duration'].values
    f = Fitter(duration, distributions = ['gama', 'lognorm', 'beta', 'burr', 'norm'])
    f.fit()
    f.summary()
    
    
# %%


