#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""

import drrdTools as dr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import drrd_functions as drrd
from scipy.stats import pearsonr
import os

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


#---- LOADING DATA ----
df = None

for prefix in PREFIXES:
    if df is None:
        df = pd.read_csv(DATA_PATH +f'{prefix}.csv')
    else:
        data = pd.concat([df, pd.read_csv(DATA_PATH +f'{prefix}.csv')], ignore_index=True)

data = data.astype({'rat':int,'session':int,'reinforced':int})


# ---- CHECKING EVOLUTION OF BEHAVIOR OVER SESSIONS ----

drrd.early_versus_late(data)

# ---- CHECK THE DURATION FOR ENTIRE SESSION PER GROUP ----

# plot variability of each rat for stable sessions

#drrd.check_individual_variability(data, above_sess= 1)
    
# check if the durations correlate between animals

#drrd.correlation_in_long_responses(data)
    
    
# check distribution of all responses

drrd.check_response_distribution(data, log_scale=False, bins= np.arange(-1,20,0.25),
                                xlim=[0,20])
drrd.check_response_distribution(data)
    
# ----- COMPARE GROUPS KDE PLOTS -----

drrd.compare_group_kdes(data, xlim=[0,25])
drrd.compare_group_kdes(data, log_scale=True, xlim= [-3,4])
    

# ------ FIT DOUBLE GAUSSIAN ---------

drrd.fit_double_gaussian(data.query('session>=1'), title=f'{PREFIX} all rats all sessions')    
    

# check evolution of dg fit over sessions
drrd.plot_evolution_double_gaussian_over_sessions(data)
    




