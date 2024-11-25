#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""

# import drrdTools as dr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import drrdTools as dr

# plt.style.use('ggplot')
# sns.set(style='ticks')

# sns.set_palette(['#F2A65A', '#772F1A'])
sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

DATA_PATH = '../data/processed/'
OUTPUT_PATH = '../output/temp/'
PREFIX = 'AY4'


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

def early_versus_late(df):
    
    f,ax = plt.subplots(1,1,figsize=(4,3))
    # add a stage column to identify as early or late in training
    df.loc[:,'stage'] = classify_stage(df.trial)

    sns.boxplot(x='session',y='duration',hue= 'stage', hue_order= ['early','late'],
        data=df.groupby(['rat','session','group','stage'])\
               .duration.mean().reset_index())
    plt.ylabel('duration (s)')
    plt.title('Average response duration early and late in each session')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+'early_vs_late.pdf')

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
    sns.boxplot(x='session',y= 'duration', hue= 'group', hue_order=('5s','10s'),
                data= dfabove)
    
    plt.ylabel('duration (s)')
    plt.title(f'Mean duration of responses above {above_time}s')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+'average_long_responses.pdf')

def check_individual_vabiability(df, above_time= 5, above_sess= 1):
    
    # selecting only responses above a certain time
    dfabove = df.query(f'duration>{above_time} and session>{above_sess}').groupby(['rat','session','group']).duration.mean().reset_index()

    plt.figure()
    sns.boxplot(x='rat',y= 'duration', hue= 'group', hue_order=('5s','10s'),
                data= dfabove.groupby(['rat','session','group']).duration.mean().reset_index())
    
    plt.ylabel('duration (s)')
    plt.title(f'Individual duration of responses above {above_time}s')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+'individual_variability.pdf')
    
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
        
    plt.hist(x,bins= bins);
    plt.axvline(criterion, **{'color':'k', 'linestyle':'--','lw':0.5})
    plt.xticks(ticks, labels=lbls)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title('Distribution of response duration')

    if xlim is not None:
        plt.xlim(xlim)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH+f'response_distribution_{scale_name}.pdf')

def compare_group_kdes(df, log_scale=False, xlabel='Duration (s)', xlim=None, title='KDE of duration by group'):
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
    plt.savefig(OUTPUT_PATH+f'kde_{scale_name}.pdf')
    plt.show()
    
def fit_double_gaussian(df, log_scale= True, title= None):    
    
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
    
    # plt.title(f'rat={rat}  session {session}')
    plt.show()

def fit_double_gaussian_old(df, log_scale= True):    
    
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
        
        
    for rat in df.rat.unique():
        for session in range(1,6):

            # selecting variable to make histogram (x)
            x = x = df.query(f'rat=={rat} and session=={session}').duration
            
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
            
            plt.title(f'rat={rat}  session {session}')
            plt.show()

def main():
       
    
    # read dataframe from file
    df= pd.read_csv(DATA_PATH + f'{PREFIX}.csv')

    # checking differences between beginnind and end of each session
    early_versus_late(df)

    # check the duration for the entire session as a function of the sessions per group
    sns.lmplot(x='session',y='duration',hue='group', markers=['>','o'],\
                data= df.groupby(by=['rat','session','group']).duration.mean().reset_index())

    # plot the fraction of reinforced trials
    sns.lmplot(x='session',y='reinforced',hue='group', markers=['>','o'],\
                data= df.query('session>2').groupby(by=['rat','session','group']).reinforced.mean().reset_index())

    # plot the average long responses as a function of trials
    plot_average_long_responses(df, above_time= 5, above_sess= 1)
    
    # plot variability of each rat for stable sessions
    check_individual_vabiability(df, above_sess= 1)
    
    # check distribution of all responses
    check_response_distribution(df, log_scale=False, bins= np.arange(-1,20,0.25),
                                xlim=[0,20])
    check_response_distribution(df)
    
    # compare the distributions using a kde
    compare_group_kdes(df, xlim=[0,25])
    compare_group_kdes(df, log_scale=True, xlim= [-3,4])
    
    # fit double gaussian
    fit_double_gaussian(df)    
    
    return df

if __name__=='__main__':
    df = main()
