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

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIXES = ['BD2']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_Error_Monitoring/')+'/'
LAST_SESSION = 4
ANIMALS = range(117,123)
PREFIX = 'DRRD_Error_Monitoring'

#%%

#---- LOADING DATA ----
data = None

for prefix in PREFIXES:
    if data is None:
        data = pd.read_csv(DATA_PATH +f'{prefix}.csv')

data = data.astype({'rat':int,'session':int,'reinforced':int})



# ----- MEAN RESPONSE DURATION -----

df_mean = data.groupby(['rat', 'session']).duration.mean().reset_index()
plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='rat', data=df_mean)
plt.xlabel('Session')
plt.ylabel('Mean Response Duration')
plt.title('Mean Response Duration per Session and Group')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_Response_Duration_per_rat.png'))

#%%

#-------- FIT DOUBLE GAUSSIAN TO LOG OF DURATION DISTRIBUTION ---------

# BY GROUP EARLY AND LATE SESSIONS

for rat in data.rat.unique():
    df_rat = data.query('rat == @rat and session == 4').reset_index(drop=True)
    drrd.check_response_distribution(df_rat,log_scale=False, criterion=2, bins = np.arange(-0.5,4.5,0.05),lbls=[0.025,0.05,0.75,1.0,1.5,2.0,2.5,3.0,5.0])

#%%    

#---------- CHECK RESPONSE DISTRIBUTION PER GROUP ------------

for rat in data.rat.unique():
    thisdf = data.query('rat == @rat').reset_index(drop=True)
    drrd.fit_double_gaussian(thisdf,log_scale=False, bins = np.arange(-0.5,4.5,0.1),\
                              bins_fine= np.arange(-0.5,4.5,0.01), title=f'{PREFIX} rat {rat}', savefig=True)

# %%
