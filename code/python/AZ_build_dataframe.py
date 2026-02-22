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

PREFIXES = ['AZ1','AZ2','AZ3','AZ4']
DATA_PATH = os.path.realpath('../../data/raw/AZ/')+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'
ABOVE = 5 # in seconds

#----AAZ1----

LAST_SESSION = 1
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
az = [87,88,89,90,91,92]

dr.exp_dataframe('AAZ1', az, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)


#---AZ1----

LAST_SESSION = 2
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
az = [87,88,89,90,91,92]

dr.exp_dataframe(PREFIXES[0], az, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)


#----AZ2----

LAST_SESSION = 20
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
az = [87,88,89,90,91,92]

dr.exp_dataframe(PREFIXES[1], az, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)

#---AZ3----

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
az = [87,88,89,90,91,92]
    
dr.exp_dataframe(PREFIXES[2], az, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)

#---AZ4----

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
az = [87,88,89,90,92]

dr.exp_dataframe_timeout(PREFIXES[3], az, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)
