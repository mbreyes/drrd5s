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

PREFIX = 'AZ4'
DATA_PATH = '../data/raw/'+PREFIX+'/'
OUTPUT_PATH = '../output/temp/'

rats_g1 = [87,88,89,90,92] # iti 20 s
rats_g2 = [] # not used in this experiment
rats_g3 = [] # not used in this experiment

RATS = rats_g1 + rats_g2 + rats_g3

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)


def which_group(rat):
    if rat in rats_g1:
        return 'timeout20'
    elif rat in rats_g2:
        return 'Erro2'
    elif rat in rats_g3:
        return 'Erro3'
    else:
        return None


def read_data_from_one_rat(prefix= 'AZ1', rat= 87, sessions= ALL_SESSIONS,\
                           dataPath= DATA_PATH ):
    print('Reading data:\n', rat, sessions, which_group(rat))
    D = dr.drrd(prefix   = prefix   , 
                animalID = rat      , 
                sessions = sessions ,
                dataPath = DATA_PATH, 
                plotFlag = True     ,
                elimin_begin = False,
                events_to_eliminate= (5,9) )
    return D

def format_data_to_dataframe(D, all_data, session, rat):
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
        
    return all_data

def fix_trial_number_in_first_session(df):
    
    df.reset_index(drop=True, inplace= True)

    return df

def load_data_from_all_rats(prefix):

    all_data = []
    
    for rat in RATS: 
        for session in ALL_SESSIONS:
            D = read_data_from_one_rat(prefix= prefix, rat= rat, sessions=[session])
            all_data = format_data_to_dataframe(D, all_data, session, rat)
        
    all_data.reset_index(inplace= True)
    
    df = all_data[['rat','index', 'group','session','duration', 'iti', 'reinforced', 'criterion', 'valid' ]]
    df.columns = ['rat','trial', 'group','session','duration', 'iti', 'reinforced', 'criterion', 'valid' ]
    df = df[['rat', 'session','trial', 'group','duration', 'iti', 'reinforced', 'criterion', 'valid' ]]
    
    
    df = fix_trial_number_in_first_session(df.copy())
    
    return df

if __name__=='__main__':
    
    df = load_data_from_all_rats(prefix = PREFIX)          
    df.to_csv('../data/processed/'+ PREFIX + '_tentative.csv', index=False)

