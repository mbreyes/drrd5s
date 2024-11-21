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
import drrdTools as dr

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

def get_group(df,rat):
    
    group= df.query(f'rat=={rat}').group.unique()
    
    if len(group)==1:
        return group[0]
    else:
        print('No group or mor than one group found')
        return None
    
def plot_percentile_over_sessions(df, percentile=90):
    
    # making a list with percentile over session for all rats
    data = [[rat, sess, np.percentile(df.query(f'rat=={rat} and session=={sess}').duration.to_numpy(),percentile)] for sess in range(1,10) for rat in range(39,51)]

    
    data = pd.DataFrame(data, columns=['rat','session',f't({percentile}%)'])
    
    data.loc[:,'group'] = [get_group(df, rat) for rat in data.rat]
    print(data)
    plt.figure()
    sns.lmplot(x='session', y= f't({percentile}%)', hue= 'group', data=data )


def analyze_double_gauss_fit(df):
    
    mus = []
    rat = 40
    for rat in df.rat.unique():
        for session in df.query(f'rat=={rat}').session.unique():
        
            print( f"rat: {rat}, session: {session}")
            plt.figure()    
            p = dr.fit_single_animal(animal=rat,session=session,x=np.arange(0,11,0.1),\
                    plotFlag=True, prefix='AW1', dataPath='../data/raw/AW1/',\
                    xlimits=[0,15], boundsDoubleGauss=(0,[1,2,8,15,8]),\
                    initParsDoubleGauss=(0.5, 0.2, 0.1, 1, 0.5));
    
            mus.append([rat, session, p[3]])
    return mus


def check_log_scale(df):
    
    for rat in df.rat.unique():
        for session in df.query(f'rat=={rat}').session.unique():
        
            print( f"rat: {rat}, session: {session}")
            plt.figure()    

            D = dr.drrd(prefix='AW1', animalID=rat, sessions=[session],\
                        dataPath='../data/raw/AW1/'); 
            plt.figure()
            plt.hist(D[:,0], bins=np.arange(0,12,0.1));
            plt.hist(np.log(D[:,0]),bins=np.arange(-4,5,0.2))
    
def compare_kde(df, x= 'logdur', sess= [5,15], cummulative= True):

    df.loc[:,'logdur']= np.log(df.duration)
    
    for rat in df.rat.unique():
        plt.figure()
        plt.title(f'rat={rat}')
        dfaux =df.query(f'rat=={rat} and session in {sess}')
        sns.kdeplot(x=x, data= dfaux, fill= True, palette="crest",\
                    alpha=.5, hue='session',common_norm=False,\
                    cumulative= cummulative)
        plt.axvline(np.log(10))

        
# --- main ---
# reading data from disk
df    = pd.read_csv(DATA_PATH + PREFIX+'.csv')
dfab5 = pd.read_csv(DATA_PATH + PREFIX+'_frac.csv')

compare_kde(df, x= 'logdur', sess= [13,14,15], cummulative= False)


plot_group_averages_over_sessions(var='duration'  , compl= ' (s)')
plot_group_averages_over_sessions(var='reinforced', compl= ' (fraction)')
plot_group_averages_over_sessions(var='duration'  , compl= ' (s)', gtype= 'bar')
plot_group_averages_over_sessions(var='reinforced', compl= ' (fraction)', gtype= 'bar')


plot_percentile_over_sessions(df, percentile=95)

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

plt.figure()
df_reinf_a5 = df.query('duration>5').groupby(['rat','session','group']).reinforced.count().reset_index()
sns.boxplot(x='session',y='reinforced',hue='group',data=df_reinf_a5, palette='tab10')
plt.ylabel('number of resps above 5s')

# mus = analyze_double_gauss_fit(df)

# check_log_scale(df)
