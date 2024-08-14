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

PREFIX = 'AW0'
DATA_PATH = '../data/raw/'+PREFIX+'/'
OUTPUT_PATH = '../../../output/temp/'

rats_g1 = [39,43,47] # grupo linear
rats_g2 = [40,44,48] # grupo exp 10% 
rats_g3 = [41,45,49] # grupo exp 20% 
rats_g4 = [42,46,50] # grupo percentile 

rats = rats_g1 + rats_g2 + rats_g3 + rats_g4

LAST_SESSION = 1
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)


def which_group(rat):
    if rat in rats_g1:
        return 'linear'
    elif rat in rats_g2:
        return 'exp10'
    elif rat in rats_g3:
        return 'exp20'
    elif rat in rats_g4:
        return 'perc'
    else:
        return None


frac_above_5s = []
all_data = []


for rat in [41,42]: 
    for session in ALL_SESSIONS:
        print(rat, session, which_group(rat))
        D = dr.drrd(prefix= 'AW', animalID=rat, sessions=[session],
                    dataPath=DATA_PATH, plotFlag=True, events_to_eliminate=(5,9))
        
        frac_above_5s.append( [rat, which_group(rat), session,\
                               len(D[D[:,0]>5]) / len(D), len(D) ] )
           
        if type(all_data) == list:
            all_data = pd.DataFrame(D, columns= ['duration','iti','reinforced',\
                                             'valid','criterion','session'])
            all_data.loc[:,'rat'] = rat
            all_data.loc[:,'group'] = which_group(rat)
            
        else:
            thisD = pd.DataFrame(D, columns= ['duration','iti','reinforced',\
                                             'valid','criterion','session'])
            thisD.loc[:,'rat'] = rat
            thisD.loc[:,'group'] = which_group(rat)
            all_data = pd.concat((all_data,thisD))

df = all_data[['rat', 'group', 'session','duration', 'iti', 'reinforced', 'criterion' ]]
df.to_csv('../data/processed/AU0_tentative.csv', index=False)

