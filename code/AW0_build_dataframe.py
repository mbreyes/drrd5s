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
DATA_PATH = '../data/raw/'+PREFIX+'/'
OUTPUT_PATH = '../../../output/temp/'

rats_g1 = [87,90] # grupo linear
rats_g2 = [88,91] # grupo exp 10% 
rats_g3 = [89,92] # grupo exp 20% 

rats = rats_g1 + rats_g2 + rats_g3

LAST_SESSION = 1
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)


def which_group(rat):
    if rat in rats_g1:
        return 'timeout'
    elif rat in rats_g2:
        return 'no_timeout_retract'
    elif rat in rats_g3:
        return 'no_timeout'
    else:
        return None


frac_above_5s = []
all_data = []


for rat in rats: 
    for session in ALL_SESSIONS:
        print(rat, session, which_group(rat))
        D = dr.drrd(prefix= PREFIX, 
                    animalID=rat, 
                    sessions=[session],
                    dataPath=DATA_PATH, plotFlag=True,
                    elimin_begin= False,
                    events_to_eliminate=(5,9))
        
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
            all_data = pd.concat((all_data,thisD), ignore_index= True)

df = all_data[['rat', 'group', 'session','duration', 'iti', 'reinforced', 'criterion' ]]
df.to_csv('../data/processed/'+ PREFIX + '_tentative.csv', index=False)

