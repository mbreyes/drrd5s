#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:22:37 2025

@author: ailacamara
"""



# import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pingouin as pg
from scipy.optimize import curve_fit
from scipy.stats import shapiro



plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = "AZ1 and BB1"
PREFIXES = ['AZ1', 'BB1']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/AZ+BB/AZ1+BB1/')+'/'
ANIMALS = list(range(87,93)) + list(range(105,117))
LAST_SESSION = 2
list_data = []

GROUPS = ['no_timeout','retract','random']
# Defining functions to plot graphs
for prefix in PREFIXES:
     
    data    = pd.read_csv(DATA_PATH +f'{prefix}.csv')
    data    = data.astype({'rat':int,'session':int,'reinforced':int, 'group':str})
    list_data.append(data)

data_all = pd.concat(list_data, ignore_index = True)
df = data_all.copy()
df = df.groupby(['rat', 'session', 'group'], group_keys = True)\
            .apply(lambda x : pd.Series(dict(nresps = (x.duration).count(),  
                        frac_correct = ((x.reinforced).sum()/(x.duration).count()), \
                        duration_mean = (x.duration).mean(), 
                        last_criterion = (x.criterion).max()))).reset_index()
                
df.rename(columns = {'duration_mean': 'Mean Duration', \
                         'last_criterion':'Last Criterion'}, inplace = True) 
              
columns = ['Mean Duration'] 




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
        plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_{gtype}plot_{var}_per_group.png'), dpi=500)
        
    return

def compare_group_kdes(df, log_scale=False, xlabel='Duration (s)', session='beginning', xlim=None, title='KDE of duration by group'):
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
    plt.savefig(OUTPUT_PATH+f'kde_{scale_name}_{session}.png', dpi=500)
    plt.show()


for var in columns:
    
    plot_over_sessions(var = var, data = df, 
                       gtypes = ['bar','lm', 'box', 'point'])

       

# Graphics of Probability Density of duration_mean per group


# minímo sessão 1
min_ = data_all.groupby(['rat', 'session']).max('trial').reset_index()
min_session_1 = min_.loc[min_['session']==1, 'trial'].min()

df_session_1 = data_all.query('session==1 and trial<= @min_session_1')

#mínimo sessão 2
min_session_2 = min_.loc[min_['session']==2, 'trial'].min()

df_session_2 = data_all.query('session==2 and trial<= @min_session_2')

# df filtrado
df_crit = pd.concat([df_session_1, df_session_2]).reset_index()
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

df_begin = df_crit.query('session == 1 and  trial <= 30')  
df_begin['session'] = df_begin['session'].replace(to_replace = 1, value = 'begin', inplace = True)
df_end = df_crit.query('session == 1')
df_end = df_end.groupby(['rat', 'session']).tail(30)
df_end['session'] = df_end['session'].replace(to_replace = 1, value ='end', inplace = True)


df_compare = pd.concat([df_begin, df_end]).reset_index()
df_mean = df_compare.groupby(['rat','session', 'group'], group_keys = True).duration.mean().reset_index()


plt.figure()
sns.barplot(x='session',y='duration',hue='group',data = df_mean)
plt.ylabel('Duration Mean (s)') 
plt.xlabel('Session 1')
plt.title('Duration Mean First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_barplot_comparing_session_1.png'), dpi=500)


plt.figure()
sns.boxplot(x='session',y='duration',hue='group',data = df_mean)
plt.ylabel('Duration (s)') 
plt.xlabel('Session 1')
plt.title('Duration Mean First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_boxplot_comparing_session_1.png'), dpi=500)


plt.figure()   
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='group', data=df_mean, dodge = 0.2 )
plt.ylabel('Duration (s)') 
plt.xlabel('Session 1')
plt.title('Duration First Session')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_pointplot_comparing_session_1.png'), dpi=500)

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
compare_group_kdes(df= df_begin, title = 'KDE of duration at beginning session 1', xlim = 0, session ='beginning')
compare_group_kdes(df = df_end, title = 'KDE of duration at end session 1', xlim = 0, session = 'end')


plt.figure()
sns.kdeplot(data=df_begin, x='duration', hue='group', common_norm=False)
sns.kdeplot(data=df_end, x='duration', hue='group', common_norm=False)
plt.title('KDE of duration by group')
plt.xlabel('Duration (s)')
plt.ylabel('Density')
plt.xlim([0,4])
plt.tight_layout()
plt.savefig(OUTPUT_PATH+'kde_linear_begin_end.png', dpi=500)
plt.show()


#Comparing begin session 1, end session 1 and end session 2

df_begin = df_crit.query('session == 1 and  trial <= 30')  
df_begin['session'] = df_begin['session'].replace(to_replace = 1, value = 'begin', inplace = True)
df_compare_session = df_crit.groupby(['rat', 'session']).tail(30)
df_compare_session['session'] = df_compare_session['session'].replace(to_replace = [1,2], value = ['end', 'session 2'], inplace = True)

df_compare = pd.concat([df_begin, df_compare_session]).reset_index()
df_compare_mean = df_compare.groupby(['rat','session', 'group'], group_keys = True).duration.mean().reset_index()


plt.figure()
sns.barplot(x='session',y='duration',hue='group',data = df_compare_mean)
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_barplot_comparing_sessions.png'), dpi=500)


plt.figure()
sns.boxplot(x='session',y='duration',hue='group',data = df_compare_mean)
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_boxplot_comparing_sessions.png'), dpi =500)


plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='duration', hue='group', data=df_compare_mean, dodge = 0.2 )
plt.ylabel('Duration (s)') 
plt.xlabel('Session')
plt.title('Trial Duration over Sessions')
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_pointplot_comparing_sessions.png'), dpi=500)
