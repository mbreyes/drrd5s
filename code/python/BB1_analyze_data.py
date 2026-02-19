#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:25:35 2025

@author: ailacamara
"""

# import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = 'BB1'
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/BB/BB1/')+'/'
LAST_SESSION = 2
ANIMALS = list(range(105,117))
GROUPS = ['notimeout','retract','random']


data    = pd.read_csv(DATA_PATH +f'{PREFIX}.csv')
data    = data.astype({'rat':int,'session':int,'reinforced':int, 'group':str})
df = data.copy()
df      = df.groupby(['rat', 'session', 'group'], group_keys = True)\
        .apply(lambda x : pd.Series(dict(nresps = (x.duration).count(),  
                    frac_correct = ((x.reinforced).sum()/(x.duration).count()), \
                    duration_mean = (x.duration).mean(), 
                    last_criterion = (x.criterion).max()))).reset_index()
            

df.rename(columns = {'duration_mean': 'Mean Duration', \
                     'frac_correct': 'Fraction responses above 5s', \
                         'last_criterion':'Last Criterion'}, inplace = True) 
              
columns = ['Mean Duration','Fraction responses above 5s'] 



def plot_all_histograms(dt:float = 0.1, tmax:float = 5):
    
    rng = np.arange(0,tmax+dt,step=dt)

    for groups in GROUPS:
        
        for session in [1,2]:
        
            dfs = data.query(f'group=="{groups}" and session=={session}')
            crit= dfs.criterion.max()
        
            plt.figure()
            plt.hist(dfs.duration,bins=rng)
            plt.axvline(dfs.duration.mean()+dfs.duration.std())
            plt.axvline(np.percentile(dfs.duration,85), color='k', ls='--')
            plt.axvline(crit, color='c', ls='-.')
            plt.xlabel('time (s)')
            plt.ylabel('number of responses')
            plt.title(f'Histogram session {session} for group {groups}')
    
    return

#plot_all_histograms()
# Plotting main graphics giving a overview of group progression through sessions

def plot_over_sessions(data = df, var='duration_mean', \
                       compl = "(s)", hue = 'group', gtypes=['bar','lm', 'box', 'point']):
    
    for gtype in gtypes:
              
        if gtype == 'bar':
            plt.figure()    
            sns.barplot(x='session',y=var,hue=hue,data = data)    
        
        elif gtype == 'lm':
            plt.figure()    
            sns.lmplot(x='session',y=var,hue=hue,\
                            markers=['>','o','+'], data= data)
            
        elif gtype == 'box':
            plt.figure()    
            sns.boxplot(x='session',y=var,hue=hue,data=data)
            
        elif gtype == 'point':
            plt.figure()    
            sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
            sns.pointplot(x='session', y=var, hue=hue, data=data, dodge = 0.2 )
               
        else:
            
            print(f'Unknown type of graph ({gtype})')
            
        plt.ylabel(var+compl)
        plt.title(f'{var} per session')
        plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_{gtype}plot_{var}_per_group.png'))
        
    return


for var in columns:
    
    plot_over_sessions(var = var, data = df, 
                       gtypes = ['bar','lm', 'box', 'point'])

       

# Graphics of Probability Density of duration_mean per group
df_crit = data.query('trial < 801 and duration <7.5')

plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(df_crit[df_crit.session ==1], x = "duration", hue = "group", fill=True, 
            common_norm=False, common_grid = True, alpha=.5, linewidth=0)
plt.xlabel("Time (s)")
plt.ylabel("Probability Density")
plt.title("Density Probability First Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_first_session.png'))
 
plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(df_crit[df_crit.session ==LAST_SESSION], x= "duration", hue ="group", fill=True, 
            common_norm=False, common_grid = True, alpha=.5, linewidth=0)
plt.xlabel("Time (s)")
plt.ylabel("Probability Density")
plt.title("Density Probability Last Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_last_session.png'))


# Comparing beginning of session 1 and end of session 1


df_begin = data.query('session == 1 and  trial <= 100 and duration < 7.5')  
df_begin['session'].replace(to_replace = 1, value = 'begin', inplace = True)
df_end = data.query('session == 1 and duration < 7.5')
df_end = df_end.groupby(['rat', 'session']).tail(100)
df_end['session'].replace(to_replace = 1, value ='end', inplace = True)


df_compare = pd.concat([df_begin, df_end]).reset_index()
df_compare_mean = df_compare.groupby(['rat','session', 'group'], group_keys = True).duration.mean().reset_index()

plt.figure()
sns.barplot(x='session',y='duration',hue='group',data = df_compare)
plt.ylabel('Duration Mean (s)') 
plt.xlabel('Session 1')
plt.title('Duration Mean First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_barplot_comparing_session_1.png'))


plt.figure()
sns.boxplot(x='session',y='duration',hue='group',data = df_compare)
plt.ylabel('Duration (s)') 
plt.xlabel('Session 1')
plt.title('Duration First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_boxplot_comparing_session_1.png'))


plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='group', data=df_compare, dodge = 0.2 )
plt.ylabel('Duration (s)') 
plt.xlabel('Session 1')
plt.title('Duration First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_pointplot_comparing_session_1.png'))


#Comparing end session 1 and end session 2

df_begin = data.query('session == 1 and  trial <= 100 and duration < 7.5')  
df_begin['session'].replace(to_replace = 1, value = 'begin', inplace = True)
df_compare_session = data.query('duration < 7.5')
df_compare_session = df_compare_session.groupby(['rat', 'session']).tail(100)
df_compare_session['session'].replace(to_replace = [1,2], value = ['end', 'session 2'], inplace = True)

df_compare = pd.concat([df_begin, df_compare_session]).reset_index()

plt.figure()
sns.barplot(x='session',y='duration',hue='group',data = df_compare)
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_barplot_comparing_sessions.png'))


plt.figure()
sns.boxplot(x='session',y='duration',hue='group',data = df_compare)
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_boxplot_comparing_sessions.png'))


plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='group', data=df_compare, dodge = 0.2 )
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_pointplot_comparing_sessions.png'))


