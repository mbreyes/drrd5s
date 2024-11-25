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

PREFIX = 'AY4'
DATA_PATH = '../data/raw/'+PREFIX+'/'
OUTPUT_PATH = '../output/temp/'
ABOVE = 10 # in seconds
LAST_SESSION = 5

rats_g1 = [63,64,65,66,68,71,72,74] # grupo 5s
rats_g2 = [52,55,56,58,59,60,61,69] # grupo 10s

rats = rats_g1 + rats_g2
rats.sort()

ALL_SESSIONS = np.arange(1,LAST_SESSION+1)

def which_group(rat:int):
    if rat in rats_g1:
        return '5s'
    elif rat in rats_g2:
        return '10s'
    else:
        return None

def find_frac_above(rat:int, session:int, D, above= ABOVE):
    return [rat, which_group(rat), session,\
                           len(D[D[:,0]>above]) / len(D), len(D), above]


def select_only_valid_trials(x, valid_col= 3):
    
    return x[x[:,3]==1,:]

# --- main --- 
frac_above = []
all_data   = []


for rat in rats: 
    for session in ALL_SESSIONS:
        print(rat, session, which_group(rat))
        D = dr.drrd(prefix= PREFIX, animalID=rat, sessions=[session],
                    dataPath=DATA_PATH, plotFlag=True, events_to_eliminate=(5,9),
                    elimin_begin= False)
        
        D = select_only_valid_trials(D)
        
        frac_above.append( find_frac_above(rat, session, D) )
           
        if type(all_data) == list:
            all_data = pd.DataFrame(D, columns= ['duration','iti','reinforced',\
                                                 'valid','criterion','session'])
            all_data.loc[:,'rat'] = rat
            all_data.loc[:,'group'] = which_group(rat)
            all_data.loc[:,'trial'] = np.arange(1,len(all_data)+1)
            
        else:
            thisD = pd.DataFrame(D, columns= ['duration','iti','reinforced',\
                                              'valid','criterion','session'])
            thisD.loc[:,'rat'] = rat
            thisD.loc[:,'group'] = which_group(rat)
            thisD.loc[:,'trial'] = np.arange(1,len(thisD)+1)
            
            all_data = pd.concat((all_data,thisD))


df = all_data[['rat', 'group', 'session','trial','duration', 'iti',
               'reinforced', 'valid', 'criterion' ]]

df = df.astype({'rat':int,'session':int,'reinforced':int, 'valid':int,
                'criterion':int})

# select only valid responses
df = df.query('valid==1').copy()

#dropping the valid column since it won't be used
df.drop(columns='valid', inplace= True)

# save data to 
df.to_csv('../data/processed/'+PREFIX+'_tentative.csv', index=False)
