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
import drrd_10s as drrd
import matplotlib.pyplot as plt
import os
import pingouin as pg
from scipy.optimize import curve_fit
from scipy.stats import shapiro

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIXES = ['AY1', 'AW1']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_10s_Groups/')+'/'
LAST_SESSION = 19
ANIMALS = range(39,75)
PREFIX = 'DRRD_10s'



#---- LOADING DATA ----
df = None

for prefix in PREFIXES:
    if df is None:
        df = pd.read_csv(DATA_PATH +f'{prefix}_tentative.csv')
    else:
        data = pd.concat([df, pd.read_csv(DATA_PATH +f'{prefix}_tentative.csv')], ignore_index=True)

data = data.astype({'rat':int,'session':int,'reinforced':int})
data = data.query('session <= 19')


df_above_5 = data.query('duration > 5')
df_mean_above_5 = df_above_5.groupby(['rat','session', 'group']).duration.mean().reset_index()
df_mean_above_5.rename(columns = {'duration':'Mean Duration Above 5'}, inplace = True) 


# ----- NUMBER OF SESSIONS TO REACH CRITERIA -----
dfs = data.copy()
dfs['criterion'] = dfs['criterion'].round()
df_sessions = dfs.query('criterion == [2,5,8,10]').groupby(['rat','group','session']).criterion.first().reset_index()

plt.figure()    
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='criterion', y='session', hue='group', data=df_sessions, dodge = 0.2 )
plt.xlabel('Criterion')
plt.ylabel('Number of Sessions')
plt.title('Number of sessions to reach criteria')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Sessions_per_group.png'))



# ----- NUMBER OF TRIALS TO REACH CRITERIA -----

df_total_trials = pd.DataFrame()

for r in range(39,75):

    df = data.copy() 
    df_rat = df.query('rat == @r')
    df_rat['total_trials'] = range(1,df_rat['rat'].count()+1)
    df_total_trials = pd.concat([df_total_trials, df_rat])
    
df_total_trials['criterion'] = df_total_trials['criterion'].round()
df_ = df_total_trials.query('criterion == [2,5,8,10]').groupby(['rat','group','criterion'])['total_trials'].first().reset_index()
plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='criterion', y='total_trials', hue='group',data=df_)
plt.xlabel("Criterion")
plt.ylabel("Number of trials")
plt.title('Mean of trials per group to achieve criteria')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Trials_per_group.png'))


# ----- NUMBER OF RATS THAT STAYED IN MAXIMUM CRITERIA -----

df = data.copy()
df_responses = df.query('criterion == 10').groupby(['rat','group', 'session']).first().reset_index()
df_count_rats = df_responses.groupby(['group', 'session']).rat.count().reset_index()

plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.pointplot(x='session', y='rat', hue='group', data=df_count_rats)
plt.ylabel("Number of Rats")
plt.xlabel("Session")
plt.title('')
plt.show()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_Mean_of_trials_per_group.png'))


# ------ COMPARE GROUPS KDE PLOTS -----

# FIRST 3 SESSIONS
df = data.copy()
df_first_sessions = df.query('session <= 3')
drrd.compare_group_kdes(df_first_sessions, xlim=(-1,5))

# LAST 5 SESSIONS
df_last_sessions = df.query('session >= 15')
drrd.compare_group_kdes(df_last_sessions, xlim=(-1,20))


#-------- FIT DOUBLE GAUSSIAN TO LOG OF DURATION DISTRIBUTION ---------

# BY GROUP EARLY AND LATE SESSIONS

for group in df.group.unique():
    df_group = df.query('group == @group and session >= 16').reset_index(drop=True)
    drrd.fit_double_gaussian(df_group,title=f'Double Gaussian Fit - Group {group}')
    df_group = df.query('group == @group and session <= 3').reset_index(drop=True)
    drrd.fit_double_gaussian(df_group,title=f'Double Gaussian Fit - Group {group}')
    

#---------- CHECK RESPONSE DISTIRBUTION PER GROUP ------------

# BY GROUP EARLY AND LATE SESSIONS

for group in df.group.unique():
    df_group = df.query('group == @group and session >= 16').reset_index(drop=True)
    drrd.check_response_distribution_per_group(df_group, group = group)
    df_group = df.query('group == @group and session <= 3').reset_index(drop=True)
    drrd.check_response_distribution_per_group(df_group, group = group)



def plot_over_sessions(data = df_last_sessions, var='duration_mean', \
                       hue = 'group', gtypes=['bar','lm', 'box', 'point']):
    
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
            
        plt.ylabel(var)
        plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_{gtype}plot_{var}_per_group.png'))
        
    return


def anova_rm(df):
    
    aov = pg.rm_anova(data=df, dv = 'session', within = ['criterion', 'group'], subject = 'rat', detailed = True)
    
    print(aov)
    
def anova_10s(df):
    
    anov = pg.anova(data = df, dv = 'session', between = 'group')
    
    return anov


df_mean = data.groupby(['group', 'session', 'rat'], group_keys = True).duration.mean().reset_index()

df_mean_begin_end = df_mean.query('session ==1 or session==19')

# ANOVA
aov = pg.mixed_anova(dv="duration", within="session", between="group", subject="rat", data=df_mean_begin_end)
print("\nMixed ANOVA results:\n", aov)
test = pg.pairwise_tests(dv='duration', within = 'session', between = 'group', subject = 'rat', data = df_mean_begin_end)
print("\n Pairwise tests results:\n", test)

# 🔹 Check assumptions
# Normality of residuals (Shapiro-Wilk)
for g in df_mean_begin_end["group"].unique():
    for t in df_mean_begin_end["session"].unique():
        df_shapiro = df_mean_begin_end.query("group == @g and session == @t")["duration"]
        stat, p = shapiro(df_shapiro)
        print(f"Group {g}, Session {t}: W={stat:.3f}, p={p:.3f}")
        

plt.figure()
sns.kdeplot(data=df_mean.query('session ==19'), x='duration', hue='group', common_norm=False, log_scale=True)
sns.kdeplot(data=df_mean.query('session==1'), x='duration', hue='group', common_norm=False, log_scale=True)
plt.title('KDE of duration by group')
plt.xlabel('Duration (s)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(OUTPUT_PATH+'kde_linear_begin_end.png', dpi=500)
plt.show()




aov = pg.mixed_anova(dv="session", within="criterion", between="group", subject="rat", data=df_sessions)
print("\nMixed ANOVA results:\n", aov)
test = pg.pairwise_tests(dv='session', within = 'criterion', between = 'group', subject = 'rat', data = df_sessions)
print("\n Pairwise tests results:\n", test)



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
plt.title("Density Probability Last Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_last_session.png'))

plt.figure()
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.kdeplot(data[data.session ==LAST_SESSION], x= "duration", hue ="group", fill=True, 
            common_norm=False, common_grid = True, alpha=.5, linewidth=0)
plt.xlabel("Time Average (s)")
plt.ylabel("Probability Density")
plt.title("Density Probability Last Session")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_density_last_session.png'))






aov = pg.mixed_anova(dv="total_trials", within="criterion", between="group", subject="rat", data=df_)
print("\nMixed ANOVA results:\n", aov)
test = pg.pairwise_tests(dv='total_trials', within = 'criterion', between = 'group', subject = 'rat', data = df_)
print("\n Pairwise tests results:\n", test)








def anova_ultima_sessao(data):
    
    
    df_last = data.query('session == 19 and duration > 5')
    
    df_last_mean = df_last.groupby(['rat', 'group']).duration.mean().reset_index()
    
    sns.boxplot(data = df_last, x = 'session', y = 'duration', hue = 'group')
    
    sns.boxplot(data = df_last_mean, y = 'duration', x = 'group')
    
    pg.anova(data = df_last, dv = 'duration', between= 'group')
    
    return

# Set of graphs of individual rats 

plt.figure()
g = sns.FacetGrid(df_, col="rat", col_wrap=6, height=3, hue = "session", palette = "GnBu_d")
g.map(sns.scatterplot, "nresps", "reinforced")
g.add_legend()
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_duration_mean_per_rat.png'))

plt.figure()
g = sns.FacetGrid(df, col = "rat", col_wrap=6, height=3)
g.map(sns.scatterplot, "session", "Last Criterion")
plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_criterion_progression.png'))



#for animal in ANIMALS:
#       plt.figure()
#      sns.scatterplot(df[df.rat == animal], y="last_criterion", x="frac_above_5",\
#                        hue="session", palette = "GnBu_d")
#       plt.title(f"Tempo médio das respostas - rato {animal}")
       
