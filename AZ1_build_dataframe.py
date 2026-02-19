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
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = 'AZ1'
DATA_PATH = os.path.realpath("../../data/raw/AY/")+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'
LAST_SESSION = 2

rats_g1 = [89,92]  # no timeout (the lever does not retract)
rats_g2 = [88,91]  # no timeout retract (the lever retract but there is no timeout)
rats_g3 = [87,90]  # random timeout (the lever retract and have timeout)
 # grupo linear  

rats = rats_g1 + rats_g2 + rats_g3
rats.sort()

ALL_SESSIONS = np.arange(1,LAST_SESSION+1)


def which_group(rat:int):
    if rat in rats_g1:
          return 'no_timeout'
    elif rat in rats_g2:
          return 'retract'
    elif rat in rats_g3:
          return 'random' 
    else:
           return None



# --- main --- 
all_data   = []


#Analyze Session 1

for rat in rats: 
    for session in ALL_SESSIONS:
        print(rat, session, which_group(rat))
        
        if session==1:
            
            A = dr.drrd(prefix= PREFIX, animalID=rat, sessions=[session],
                        dataPath=DATA_PATH, plotFlag=True, events_to_eliminate=(5,9))

            
            AA = dr.drrd(prefix= 'AAZ1', animalID=rat, sessions=[session],
                            dataPath=DATA_PATH, plotFlag=True, events_to_eliminate=(5,9))

            joint_data = np.concatenate((AA,A))
           
            if type(all_data) == list:
                all_data = pd.DataFrame(joint_data, columns= ['duration','iti','reinforced',\
                                                     'valid','criterion','session'])
                all_data.loc[:,'rat'] = rat
                all_data.loc[:,'group'] = which_group(rat)
                all_data.loc[:,'trial'] = np.arange(1,len(all_data)+1)
                
            else:
                thisD = pd.DataFrame(joint_data, columns= ['duration','iti','reinforced',\
                                                  'valid','criterion','session'])
                thisD.loc[:,'rat'] = rat
                thisD.loc[:,'group'] = which_group(rat)
                thisD.loc[:,'trial'] = np.arange(1,len(thisD)+1)
                
                all_data = pd.concat((all_data,thisD))
      
        else:
           
           A = dr.drrd(prefix= PREFIX, animalID=rat, sessions=[session],
                       dataPath=DATA_PATH, plotFlag=True, events_to_eliminate=(5,9))
           
              
           if type(all_data) == list:
               all_data = pd.DataFrame(A, columns= ['duration','iti','reinforced',\
                                                    'valid','criterion','session'])
               all_data.loc[:,'rat'] = rat
               all_data.loc[:,'group'] = which_group(rat)
               all_data.loc[:,'trial'] = np.arange(1,len(all_data)+1)
               
           else:
               thisD = pd.DataFrame(A, columns= ['duration','iti','reinforced',\
                                                 'valid','criterion','session'])
               thisD.loc[:,'rat'] = rat
               thisD.loc[:,'group'] = which_group(rat)
               thisD.loc[:,'trial'] = np.arange(1,len(thisD)+1)
               
               all_data = pd.concat((all_data,thisD)) 

df = all_data.query('valid ==1')
df = df[['rat', 'group', 'session','trial','duration', 'iti', 'reinforced', 'criterion' ]]
df = df.astype({'rat':int,'session':int,'reinforced':int})
df.to_csv(os.path.realpath('../../data/processed/')+'/'+f'{PREFIX}_tentative.csv', index=False)



