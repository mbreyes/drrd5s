
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
import drrdTools as dr
from scipy.stats import pearsonr
import os

plt.rcParams['figure.dpi'] = 500

# plt.style.use('ggplot')
# sns.set(style='ticks')

# sns.set_palette(['#F2A65A', '#772F1A'])
sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIXES = ['AY1', 'AW1']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/Drrd_10s_groups/')+'/'
LAST_SESSION = 19
ANIMALS = range(39,75)
PREFIX = 'DRRD_10s'
list_data = []




def classify_stage(x:int, early_cutoff= 33, late_cutoff= 66):
    '''
    Parameters
    ----------
    x : Int
        Trial number.
    early_cutoff : Int, optional
        Up to what trial it can be classified as early. The default is 33.
    late_cutoff : int, optional
        Above what trial should be class as late. The default is 66.

    Returns
    -------
    "Early", "Interm "or "late" labels.

    '''    
    
    classes = []
    
    for thisx in x:
        if thisx <= early_cutoff:
            classes.append('early')
        elif thisx > late_cutoff:
            classes.append('late')
        else:
            classes.append('interm')

    return(classes)


def mean_for_each_stage(df):
    
    stages = ['early','late']
    
    # add a stage column to identify as early or late in training
    df.loc[:,'stage'] = classify_stage(df.trial)
    for st in stages:
        
        f,ax = plt.subplots(1,1,figsize=(4,3))
        sns.boxplot(x='session',y='duration',hue= 'group',
        data=df.query(f"stage == '{st}'").groupby(['rat','session','group'])\
                   .duration.mean().reset_index())
        plt.ylabel('duration (s)')
        plt.title(f'Average response duration {st} in each session')
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH+f'{st}_over_sessions.png')

def make_all_histograms(df):
    for rat in df.rat.unique():
        for session in df.query(f'rat=={rat}').session.unique():
    
            dfs = df.query(f'rat =={rat} and session=={session}')
    
            plt.figure()
            plt.hist(dfs.duration)
            plt.title(f'Experiment: {PREFIX}  rat: {rat}  session: {session}')

def plot_average_long_responses(df, above_time= 5, above_sess= 1):
    
    # selecting only responses above a certain time
    dfabove = df.query(f'duration>{above_time} and session>{above_sess}').groupby(['rat','session','group']).duration.mean().reset_index()

    plt.figure(figsize=(4,3))
    sns.boxplot(x='session',y= 'duration', hue= 'group', hue_order = dfabove.group.unique(), data= dfabove)
    
    plt.ylabel('duration (s)')
    plt.title(f'Mean duration of responses above {above_time}s')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+'average_long_responses.png')

def check_individual_variability(df, above_time= 5, above_sess= 1):
    
    # selecting only responses above a certain time
    dfabove = df.query(f'duration>{above_time} and session>{above_sess}').groupby(['rat','session','group']).duration.mean().reset_index()

    plt.figure()
    sns.boxplot(x='rat',y= 'duration', hue= 'group', hue_order=('5s','10s', '20s'),
                data= dfabove.groupby(['rat','session','group']).duration.mean().reset_index())
    
    plt.ylabel('duration (s)')
    plt.title(f'Individual duration of responses above {above_time}s')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+'individual_variability.png')
    
def check_response_distribution(df, bins= None, criterion= 10, log_scale= True,
                                lbls= [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                                new_fig= True, xlim= None):
    if new_fig:    
        plt.figure(figsize=(4,3))
    
    if type(bins) == type(None):
        bins = np.arange(-3,3.3,0.05)
    else:
        ticks = None
        lbls = None
    
    if log_scale:
        x = np.log(df.duration)
        criterion = np.log(criterion)
        ticks = np.log(lbls)
        xlabel= 'duration (s) - log scale'
        scale_name = 'log'
    else:
        x = df.duration
        ticks = lbls
        xlabel = 'duration (s)'
        scale_name = 'linear'
        
    plt.hist(x,bins= bins)
    plt.axvline(criterion, **{'color':'k', 'linestyle':'--','lw':0.5})
    plt.xticks(ticks, labels=lbls)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title('Distribution of response duration')

    if xlim is not None:
        plt.xlim(xlim)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+f'response_distribution_{scale_name}.pdf')
    
def check_response_distribution_per_group(df, bins= None, criterion= 10, log_scale= True,
                                lbls= [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                                new_fig= True, xlim= None, group = '5s'):
    if new_fig:    
        plt.figure(figsize=(4,3))
    
    if type(bins) == type(None):
        bins = np.arange(-3,3.3,0.05)
    else:
        ticks = None
        lbls = None
    
    if log_scale:
        x = np.log(df.query(f"group == '{group}'").duration)
        criterion = np.log(criterion)
        ticks = np.log(lbls)
        xlabel= 'duration (s) - log scale'
        scale_name = 'log'
    else:
        x = df.query(f"group == '{group}'").duration
        ticks = lbls
        xlabel = 'duration (s)'
        scale_name = 'linear'
        
    plt.hist(x,bins= bins)
    plt.axvline(criterion, **{'color':'k', 'linestyle':'--','lw':0.5})
    plt.xticks(ticks, labels=lbls)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title(f'Distribution of response duration group {group}')

    if xlim is not None:
        plt.xlim(xlim)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+f'response_distribution_{scale_name}_group_{group}.pdf')    

def compare_group_kdes(df, log_scale=False, xlabel='Duration (s)', xlim=None, session=None, title='KDE of duration by group'):
    
    plt.figure(figsize=(4,3))
    
    if log_scale:
        df = df.copy()
        df['log_duration'] = np.log(df['duration'])
        x_var = 'log_duration'
        xlabel = 'Log(Duration (s))'
        scale_name = 'log'
    else:
        x_var = 'duration'
        scale_name= 'linear'
        
    sns.kdeplot(data=df, x=x_var, hue='group', common_norm=False)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+f'kde_{scale_name}_{session}.png')
    plt.show()
    
def fit_double_gaussian(df, log_scale= True, title= None, savefig= False):    
    
    if log_scale:
        bins = np.arange(-4,4,0.1)
        bins_fine = np.arange(-4,4,0.01)
        init_pars = (0.858, -4, 3.4, 2.45, 0.233)
        bonds = ([0, -5, 0, 0, 0], [1, 5, 5, 20, 10])    
    else:
        bins = np.arange(-1,25,2)
        bins_fine = np.arange(-1,25,0.01)
        init_pars = (0.5, 0, 0.5, 11, 10)
        bonds = ([0, 0, 0, 0, 0], [1, 3, 4, 15, 100])
    # selecting variable to make histogram (x)
    x = x = df.duration
    if log_scale:
        x = np.log(x)
    plt.figure(figsize=(4,3))
    counts = dr.calc_histogram(x, bins)
    popt = dr.fit_single_animal_from_matrix2(bins, counts,
                initParsDoubleGauss= init_pars,
                boundsDoubleGauss= bonds,
                xlimits=[-4,4])
    f = dr.double_gaussian(bins_fine, *popt)
    plt.plot(bins_fine,f,'k')
    plt.title(title)
    
    if (title is not None) and (savefig):
        plt.savefig(OUTPUT_PATH+title.replace(' ','_')+'.pdf')
    
    plt.show()
    return popt

def plot_individual_double_gaussian_fit(df):
    # loop over all rats and sessions and show the double gaussian fit
    popts = []
    for rat in df.rat.unique():
        for session in df.query(f'rat=={rat}').session.unique():
            thisdf = df.query(f'rat=={rat} and session=={session}')
            popts.append(fit_double_gaussian(thisdf, title=f'{PREFIX} rat {rat} session {session}'))
    return popts

def plot_evolution_double_gaussian_over_sessions(df):
    
    for session in df.session.unique():
        thisdf = df.query(f'session=={session}')
        popt = fit_double_gaussian(thisdf, title=f'{PREFIX} session {session}', savefig=True)
        print(np.round(popt,2))
        
    
    
def plot_individual_double_gaussian_fit_sessions(df):
    # loop over all rats and sessions and show the double gaussian fit
    popts = []
    for rat in df.rat.unique():
        for gr in df.query(f"rat == {rat}").group.unique():
            thisdf = df.query(f"rat=={rat} and group=='{gr}'")
            popts.append(fit_double_gaussian(thisdf, title=f'{PREFIX} rat {rat} - group {gr}'))
    return popts    

def main():
    # read dataframe from file
    for prefix in PREFIXES:
        data    = pd.read_csv(DATA_PATH +f'{prefix}.csv', index_col=0)
        list_data.append(data)
        
    df_all = pd.concat(list_data)
    df = df_all.query(' 15 <= session <= 19')

    # checking differences between beginning and end of each session
    mean_for_each_stage(df)
    # plot the average long responses as a function of trials
    plot_average_long_responses(df, above_time= 5, above_sess= 1)
    
    # plot variability of each rat for stable sessions
    #check_individual_variability(df, above_sess= 1)
    
    #check if the durations correlate between animals
    #correlation_in_long_responses(df)
    
    # check distribution of all responses
    check_response_distribution(df, log_scale=False, bins= np.arange(-1,20,0.25),
                                xlim=[0,20])
    check_response_distribution(df)
    
    check_response_distribution_per_group(df, log_scale=False, bins= np.arange(-1,20,0.25),
                                xlim=[0,20], group='linear')
    
    check_response_distribution_per_group(df, log_scale=False, bins= np.arange(-1,20,0.25),
                                xlim=[0,20], group ='exp10')
    
    check_response_distribution_per_group(df, log_scale=False, bins= np.arange(-1,20,0.25),
                                xlim=[0,20], group = 'exp20')
    
    
    check_response_distribution_per_group(df,group = 'linear')
    
    check_response_distribution_per_group(df,group = 'exp10')
    
    check_response_distribution_per_group(df,group = 'exp20')
    
    # compare the distributions using a kde
    compare_group_kdes(df, xlim=[0,25])
    compare_group_kdes(df, log_scale=True, xlim= [-3,4])
    
    # fit double gaussian
    fit_double_gaussian(df.query('session>= 17'), title=f'{PREFIX} all rats all sessions')   
    
    df = df.reset_index()
    plot_individual_double_gaussian_fit(df)
    
    plot_individual_double_gaussian_fit_sessions(df)
    # show individual double gaussian fit
    # plot_individual_double_gaussian_fit(df)
    
    # check evolution of dg fit over sessions
    # plot_evolution_double_gaussian_over_sessions(df)
    
    return df

if __name__=='__main__':
   df = main()
