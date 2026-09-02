#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ailacamara
"""
#%%
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import model_fitting as mf

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIX = 'BG1'
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_longo_pilot/')+'/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

#%%
#---- LOADING DATA ----

data = pd.read_csv(DATA_PATH + f'{PREFIX}_tentative.csv')
data = data.astype({'rat':int,'session':int,'reinforced':int})

ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order
COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

df = data.copy()

sessions = sorted(df.session.unique())

#%%
# ------ Taxa de acerto (proporção de respostas reforçadas) ao longo das sessões, por animal ------

mf.proportion_correct_responses_by_rat(df, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------ Número de tentativas por animal para atingir os critérios 1s, 2s, 3s, 4s e 5s ------

bg_trials_to_criteria = mf.trials_reach_criteria_by_rat(df, criteria=[1, 2, 3, 4, 5], \
                                                          PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

bg_sessions_to_criteria = mf.sessions_reach_criteria_by_rat(df, criteria=[1, 2, 3, 4, 5], \
                                  PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)


#%%
# ------- Fit Double Gaussian to Log of Duration Distribution -------

# ------- MLE fitting, per rat and per individual session -------

bg_parameters = mf.fit_params_all_sessions(df, mf.MLE, log=True)
bg_parameters.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Parameters.csv', index=False)

#%%
# -------- Parâmetros da dupla gaussiana por animal (lineplot), um subplot por parâmetro --------

mf.plot_parameters_lineplot_by_rat(bg_parameters, order=sessions, \
                           PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- Ajuste da dupla gaussiana por animal, um subplot por sessão --------

for rat in sorted(df.rat.unique()):
    df_rat = df.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df_rat, log=True, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# %%
