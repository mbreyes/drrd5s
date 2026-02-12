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
OUTPUT_PATH = '../output/temp/'

rats_g1 = [87,90] # timeout
rats_g2 = [88,91] # no_timeout_retract 
rats_g3 = [89,92] # no_timeout 

RATS = rats_g1 + rats_g2 + rats_g3

LAST_SESSION = 2
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
            
            temp_data = []
            # patch to read data broken in two files for the first session
            if session == 1:
                for letter in ['A','B']:
                    D = read_data_from_one_rat(prefix= prefix+letter, rat= rat, sessions=[session])
                    temp_data = format_data_to_dataframe(D, temp_data, session, rat)
                
                temp_data = fix_trial_number_in_first_session(temp_data)
                
                all_data = format_data_to_dataframe(temp_data, all_data, session, rat)
            else:                    
                
                D = read_data_from_one_rat(prefix= prefix, rat= rat, sessions=[session])
            
                all_data = format_data_to_dataframe(D, all_data, session, rat)
        
    all_data.reset_index(inplace= True)
    
    df = all_data[['rat','index', 'group','session','duration', 'iti', 'reinforced', 'criterion' ]]
    df.columns = ['rat','trial', 'group','session','duration', 'iti', 'reinforced', 'criterion' ]
    df = df[['rat', 'session','trial', 'group','duration', 'iti', 'reinforced', 'criterion' ]]
    
    
    df = fix_trial_number_in_first_session(df.copy())
    
    return df

if __name__=='__main__':
    
    df = load_data_from_all_rats(prefix = PREFIX)          
    df.to_csv('../data/processed/'+ PREFIX + '_tentative.csv', index=False)

