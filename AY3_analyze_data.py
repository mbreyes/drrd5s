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
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = 'AY3'
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/AY/AY3/')+'/'
LAST_SESSION = 5

rats_g1 = [55,59,63,67,71] # grupo linear
rats_g2 = [52,56,60,64,68,72] # grupo exp 10% 
rats_g3 = [53,61,65,69] # grupo exp 20% 
rats_g4 = [58,66,74] # grupo percentile 
rats = rats_g1 + rats_g2 + rats_g3 + rats_g4
rats.sort()

ANIMALS =rats

# Defining functions to plot graphs

data    = pd.read_csv(DATA_PATH +f'{PREFIX}.csv')
data    = data.astype({'rat':int,'session':int,'reinforced':int})
df      = data.groupby(['rat', 'session', 'group'], group_keys = True)\
        .apply(lambda x : pd.Series(dict(nresps = (x.duration).count(),\
                    duration_mean = (x.duration).mean(), \
                    frac_above_5 = (x.duration>5).sum()/(x.duration).count(),\
                    correct_mean = x.loc[x.reinforced == 1, "duration"].sum()/(x.reinforced).sum(), \
                    last_criterion = (x.criterion).max()))).reset_index()

df.rename(columns = {'duration_mean': 'Mean Duration', \
                     'frac_above_5': 'Fraction responses above 5s', 'correct_mean': 'Mean Reinforced Responses', \
                         'last_criterion':'Last Criterion'}, inplace = True) 
              
columns = ['Mean Duration',
       'Fraction responses above 5s', 'Mean Reinforced Responses'] 


def plot_all_histograms(dt:float = 0.1, tmax:float = 10):
    
    rng = np.arange(0,tmax+dt,step=dt)

    for rat in range(51,75):
        session = 20
        
        dfs = data.query(f'rat =={rat} and session=={session}')
        crit= dfs.criterion.max()
        
        plt.figure()
        plt.hist(dfs.duration,bins=rng)
        plt.axvline(dfs.duration.mean()+dfs.duration.std())
        plt.axvline(np.percentile(dfs.duration,85), color='k', ls='--')
        plt.axvline(crit, color='c', ls='-.')
        plt.xlabel('time (s)')
        plt.ylabel('number of responses')
        plt.title(f'rat {rat} group {dfs.group.unique()}')
    
    return


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
                            markers=['>','o','+','<'], data= data)
            
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
        plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_{gtype}plot_{var}_per_group.png'))
        
    return


for var in columns:
    
    plot_over_sessions(var = var, data = df, hue = 'group', 
                       gtypes = ['bar','lm', 'box', 'point'])

       
# Graphics of Probability Density of duration_mean per group

df_above = data[data['duration'] > 5]
plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(data[data.session ==1], x = "duration", hue = "group", fill=True, 
            common_norm=False, common_grid = True, alpha=.5, linewidth=0)
plt.xlabel("Time Average (s)")
plt.ylabel("Probability Density")
plt.title("Density Probability First Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_first_session.png'))
 
plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(df_above[df_above.session ==LAST_SESSION], x= "duration", hue ="group", fill=True, 
            common_norm=False, common_grid = True, alpha=.5, linewidth=0)
plt.xlabel("Time Average (s)")
plt.ylabel("Probability Density")
plt.title("Density Probability Above 5s Last Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_above 5_last_session.png'))

plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(data[data.session ==LAST_SESSION], x= "duration", hue ="group", fill=True, 
            common_norm=False, common_grid = True, alpha=.5, linewidth=0)
plt.xlabel("Time Average (s)")
plt.ylabel("Probability Density")
plt.title("Density Probability Last Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_last_session.png'))



# Set of graphs of individual rats 

plt.figure()
g = sns.FacetGrid(df, col="rat", col_wrap=6, height=3, hue = "session", palette = "GnBu_d")
g.map(sns.scatterplot, "nresps", "Mean Duration")
g.add_legend()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_duration_mean_per_rat.png'))

plt.figure()
g = sns.FacetGrid(df, col = "rat", col_wrap=6, height=3, hue ='group')
g.map(sns.scatterplot, "session", "Last Criterion")
g.add_legend()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_criterion_progression.png'))


plt.figure()
g = sns.FacetGrid(df, col = "rat", col_wrap=6, height=3, hue = 'group')
g.map(sns.scatterplot, "session", "Fraction responses above 5s")
g.add_legend()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_fraction_above_5_per_rat.png'))
    

