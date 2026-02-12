#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""

import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = 'AZ1'
DATA_PATH = '../data/processed'+'/'
OUTPUT_PATH = '../output/temp/'

def read_data(prefix= PREFIX, data_path= None):
    df = pd.read_csv(DATA_PATH + PREFIX+'.csv')
    return df

    
def assign_trial_phase(df):
    """
    Assigns 'early', 'middle', or 'late' to trials within each (rat, session)
    combination.
    """
    df = df.copy()
    
    def classify_trials(sub_df):
        n_trials = len(sub_df)
        early_threshold = n_trials // 3
        late_threshold = 2 * (n_trials // 3)
        
        phases = ['early'] * early_threshold\
                 + ['middle'] * (late_threshold - early_threshold)\
                 + ['late'] * (n_trials - late_threshold)
        sub_df['phase'] = phases
        return sub_df
    
    return df.groupby(['rat', 'session'], group_keys=False).apply(classify_trials)

def plot_early_versus_late(df, title= None):
    
    plt.figure()
    sns.barplot(data= df, x= 'phase', y= 'duration',\
                hue= 'group',
                hue_order= ['no_timeout','no_timeout_retract','timeout'])
    
    plt.title(title)
   

if __name__=='__main__':
    
    df = read_data(prefix = PREFIX, data_path= DATA_PATH)
    df = df.query('trial<641 and duration <50')
    
    df = assign_trial_phase(df)
    
    dfr = df.query('phase in ["early","late"]')
    
    plot_early_versus_late(dfr.query('session==1'), "Session 1")
    
    plot_early_versus_late(dfr.query('session==2'), "Session 2")
    
    sns.lmplot(data=df.query('session==1'), x= 'trial', y='duration', hue= 'group', scatter_kws={'s':3})
    
    

    dfgood = dfr.groupby(['rat','session','phase','group']).duration.mean().reset_index()
    
    plt.figure()
    sns.barplot(data= dfgood.query('session==1'), x= 'phase', y= 'duration',\
                hue= 'group',
                hue_order= ['no_timeout','no_timeout_retract','timeout'])
    
    plt.title('AZ1 session 1')
   
    plt.figure()
    sns.catplot(data= dfgood.query('session==1'), x= 'phase', y= 'duration',\
                hue= 'group',
                hue_order= ['no_timeout','no_timeout_retract','timeout'])
    
    plt.title('AZ1 session 1')
   