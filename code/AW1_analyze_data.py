#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""

# import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = 'AW1'
DATA_PATH = '../data/processed/'
OUTPUT_PATH = '../../../output/temp/'


def plot_crit_progress(df):
    plt.figure()
    # checking the max criterion achieved
    df_aux = df.groupby(['rat','group']).criterion.max().reset_index()
    sns.boxplot(x='group',y='criterion', data= df_aux)
    plt.ylabel('max crit achieved')

def plot_n_trials(df):
    plt.figure()
    # checking the max criterion achieved
    df_aux = df.groupby(['rat','group']).trial.max().reset_index()
    ax = sns.boxplot(x='group',y='trial', data= df_aux)
    plt.ylabel('number of trials')
    return ax

def plot_all_histograms(dt:float = 0.1, tmax:float = 4):
    
    rng = np.arange(0,tmax+dt,step=dt)

    for rat in range(39,51):
        session = 1
        
        dfs = df.query(f'rat =={rat} and session=={session}')
        # print(dfs)
        crit= dfs.criterion.max()
    
        plt.figure()
        plt.hist(dfs.duration,bins=rng)
        #plt.axvline(dfs.duration.mean())
        plt.axvline(dfs.duration.mean()+dfs.duration.std())
        #plt.axvline(dfs.duration.mean()+dfs.duration.std()*1.5)
        plt.axvline(np.percentile(dfs.duration,85), color='k', ls='--')
        plt.axvline(crit, color='c', ls='-.')
        plt.xlabel('time (s)')
        plt.ylabel('number of responses')
        plt.title(f'rat {rat} group {dfs.group.unique()}')

def plot_group_averages_over_sessions(var:str='duration',\
    by=['rat','session','group'], compl= ' (s)', gtype='lm'):

    # grouping the variable to obtain one value per animal
    dfaux = df.groupby(by= by)[[var]].mean().reset_index()
    
    # plotting the mean duration for each rat
    plt.figure()
    
    if gtype == 'lm': 
        ax = sns.lmplot(x='session',y= var,hue='group',\
                        markers=['>','o','+','<'], data= dfaux)
    elif gtype == 'bar':
        ax = sns.barplot(x='session',y= var,hue='group',\
                        data= dfaux)
    else:
        print(f'Unknown type of graph ({gtype}), use lm or bar')
        
    # adding labels to the ylabel
    plt.ylabel(var+compl)
    
    return ax

# --- main ---
# reading data from disk
df    = pd.read_csv(DATA_PATH + PREFIX+'.csv')
dfab5 = pd.read_csv(DATA_PATH + PREFIX+'_frac.csv')


plot_group_averages_over_sessions(var='duration'  , compl= ' (s)')
plot_group_averages_over_sessions(var='reinforced', compl= ' (fraction)')
plot_group_averages_over_sessions(var='duration'  , compl= ' (s)', gtype= 'bar')
plot_group_averages_over_sessions(var='reinforced', compl= ' (fraction)', gtype= 'bar')

ax = plot_n_trials(df)

plot_crit_progress(df)

plt.figure()    
sns.catplot(x='session',y='frac',hue='rat',data=dfab5)

plt.figure()
sns.lmplot(x='session',y='frac',hue='group',data=dfab5)

plt.figure()
sns.boxplot(x='session',y='frac',hue='group',data=dfab5)

plt.figure()
sns.barplot(x='session',y='n_trials',hue='group',data=dfab5, palette='tab10')
