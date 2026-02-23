#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:22:37 2025

@author: ailacamara
"""


import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pingouin as pg
from scipy.optimize import curve_fit
from scipy.stats import shapiro
import drrd_functions as drrd



plt.style.use('ggplot')
sns.set(style='ticks')

PREFIXES = ['AAZ1','AZ1','BB1']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_Short_Learning/')+'/'
ANIMALS = list(range(87,93)) + list(range(105,117))
LAST_SESSION = 2
PREFIX = 'DRRD_Short'

#---- LOADING DATA ----
df = None

for prefix in ['AAZ1','AZ1']:
    if df is None:
        df = pd.read_csv(DATA_PATH +f'{prefix}.csv')
    else:
        data_a = pd.concat([df, pd.read_csv(DATA_PATH +f'{prefix}.csv')], ignore_index=True)
        data_a['trial'] = data_a.groupby(['rat', 'session']).cumcount() + 1

data_b = pd.read_csv(DATA_PATH +f'BB1.csv')
data = pd.concat([data_a, data_b], ignore_index=True)
data = data.astype({'rat':int,'session':int,'reinforced':int})



# ----- NUMBER OF TRIALS TO REACH CRITERIA -----

df_total_trials = pd.DataFrame()

for r in ANIMALS:

    df = data.copy() 
    df_rat = df.query('rat == @r')
    df_rat['total_trials'] = range(1,df_rat['rat'].count()+1)
    df_total_trials = pd.concat([df_total_trials, df_rat])
    
df_ = df_total_trials.query('criterion == [0.5,1.0,1.2]').groupby(['rat','group','criterion'])['total_trials'].first().reset_index()
plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='criterion', y='total_trials', hue='group',data=df_)
plt.xlabel("Criterion")
plt.ylabel("Number of trials")
plt.title('Mean of trials per group to achieve criteria')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Trials_per_group.png'))


# ----- COMPARE BEGINNING OF SESSION 1 AND END OF SESSION 1 -----
df = data.copy()
db = df.query('session ==1 and duration < 7.5')
min_trials = db.groupby(['rat']).max('trial').reset_index().loc[:, 'trial'].min()
df_begin = db.query('trial <= 100')  
df_begin['session'] = df_begin['session'].replace(1, 'begin')
df_end = db.query('trial <= @min_trials')
df_end = df_end.groupby(['rat', 'session']).tail(100)
df_end['session'] = df_end['session'].replace(1, 'end')


df_compare = pd.concat([df_begin, df_end]).reset_index()
df_compare_mean = df_compare.groupby(['rat','session', 'group'], group_keys = True).duration.mean().reset_index()

plt.figure()
sns.barplot(x='session',y='duration',hue='group',data = df_compare_mean)
plt.ylabel('Duration Mean (s)') 
plt.xlabel('Session 1')
plt.title('Duration Mean First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_barplot_comparing_session_1.png'))


plt.figure()
sns.boxplot(x='session',y='duration',hue='group',data = df_compare_mean)
plt.ylabel('Duration (s)') 
plt.xlabel('Session 1')
plt.title('Duration First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_boxplot_comparing_session_1.png'))


plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='group', data=df_compare_mean, dodge = 0.2 )
plt.ylabel('Duration (s)') 
plt.xlabel('Session 1')
plt.title('Duration First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_pointplot_comparing_session_1.png'))



# ------ COMPARE BEGINNING OF SESSION 1, END OF SESSION 1 AND END OF SESSION 2 -----

df = data.copy()
df_begin = df.query('session == 1 and  trial <= 100 and duration < 7.5')  
df_begin['session'] = df_begin['session'].replace(1, 'begin')
df_compare_session = df.query('trial <= @min_trials and duration < 7.5')
df_compare_session = df_compare_session.groupby(['rat', 'session']).tail(100)
df_compare_session['session'] = df_compare_session['session'].replace(to_replace = [1,2], value = ['end', 'session 2'])

df_compare = pd.concat([df_begin, df_compare_session]).reset_index()

df_compare_mean = df_compare.groupby(['rat','session', 'group'], group_keys = True).duration.mean().reset_index()


plt.figure()
sns.barplot(x='session',y='duration',hue='group',data = df_compare_mean)
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_barplot_comparing_sessions.png'))


plt.figure()
sns.boxplot(x='session',y='duration',hue='group',data = df_compare_mean)
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_boxplot_comparing_sessions.png'))


plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='group', data=df_compare_mean, dodge = 0.2 )
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_pointplot_comparing_sessions.png'))


# ------ COMPARE GROUPS KDE PLOTS -----

#mínimo sessão 1
min_ = data.groupby(['rat', 'session']).max('trial').reset_index()
min_session_1 = min_.loc[min_['session']==1, 'trial'].min()
df_session_1 = data.query('session==1 and trial<= @min_session_1 and duration < 7.5')

#mínimo sessão 2
min_session_2 = min_.loc[min_['session']==2, 'trial'].min()
df_session_2 = data.query('session==2 and trial<= @min_session_2 and duration < 7.5')

df_begin = df_session_1.query('trial <= 30 and duration < 7.5')
df_begin['session'] = df_begin['session'].replace(1, 'beginning')
df_end = df_session_1.query('trial <= @min_session_1 and duration < 7.5')
df_end = df_end.groupby(['rat', 'session']).tail(30)
df_end['session'] = df_end['session'].replace(1, 'end')

df_session_2 = df_session_2.groupby(['rat', 'session']).tail(30)

drrd.compare_group_kdes(df= df_begin, title = 'KDE of duration at beginning session 1', xlim = 0, session ='beginning')

drrd.compare_group_kdes(df = df_end,\
                         title = 'KDE of duration at end session 1', xlim = 0, session = 'end')

drrd.compare_group_kdes(df = df_session_2, \
                        title = 'KDE of duration at end session 2', xlim = 0, session = 'session_2')

plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(data=df_begin, x='duration', hue='group', common_norm=False)
sns.kdeplot(data=df_end, x='duration', hue='group', common_norm=False)
plt.title('KDE of duration by group')
plt.xlabel('Duration (s)')
plt.ylabel('Density')
#plt.xlim([0,4])
plt.tight_layout()
plt.savefig(OUTPUT_PATH+'kde_linear_begin_end.png', dpi=500)
plt.show()




# ------ ANOVA AND ASSUMPTIONS CHECKING -----

# ANOVA
aov = pg.mixed_anova(dv="duration", within="session", between="group", subject="rat", data=df_mean)
print("\nMixed ANOVA results:\n", aov)
test = pg.pairwise_tests(dv='duration', within = 'session', between = 'group', subject = 'rat', data = df_mean)
print("\n Pairwise tests results:\n", test)

# 🔹 Check assumptions
# Normality of residuals (Shapiro-Wilk)
for g in df_mean["group"].unique():
    for t in df_mean["session"].unique():
        df_shapiro = df_mean.query("group == @g and session == @t")["duration"]
        stat, p = shapiro(df_shapiro)
        print(f"Group {g}, Session {t}: W={stat:.3f}, p={p:.3f}")
        
        
for g in df_compare["group"].unique():
    for t in df_compare["session"].unique():
        df_shapiro = df_compare.query("group == @g and session == @t")["duration"]
        stat, p = shapiro(df_shapiro)
        print(f"Group {g}, Session {t}: W={stat:.3f}, p={p:.3f}")
        

# Homogeneity of variances across groups (Levene’s test)
print("\nHomogeneity of variances:\n", pg.homoscedasticity(df_mean, dv="duration", group="group"))



