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

DATA_PATH = '../data/processed/'
OUTPUT_PATH = '../../../output/temp/'


df= pd.read_csv(DATA_PATH + 'AU0.csv')

# df = df.query('session<3')

sns.lmplot(x='session',y='duration',hue='group', markers=['>','o'],\
           data= df.groupby(by=['rat','session','group']).duration.mean().reset_index())


sns.lmplot(x='session',y='reinforced',hue='group', markers=['>','o'],\
            data= df.groupby(by=['rat','session','group']).reinforced.mean().reset_index())



rat = 2
session = 15

dfs = df.query(f'rat =={rat} and session=={session}')

plt.figure()
plt.hist(dfs.duration)
plt.axvline(dfs.duration.mean())
plt.axvline(dfs.duration.mean()+dfs.duration.std())
plt.axvline(dfs.duration.mean()+dfs.duration.std()*1.5)
plt.axvline(np.percentile(dfs.duration,85), color='k', ls='--')
    
# dfab5 = pd.DataFrame(frac_above_5s,columns=['rat','group','session','frac_above_5s','nresps'])


# sns.catplot(x='session',y='frac_above_5s',hue='rat',data=dfab5)

# plt.figure()
# sns.lmplot(x='session',y='frac_above_5s',hue='group',data=dfab5)

# plt.figure()
# sns.boxplot(x='session',y='frac_above_5s',hue='group',data=dfab5)

# plt.figure()
# sns.boxplot(x='session',y='nresps',hue='group',data=dfab5, palette='tab10')
