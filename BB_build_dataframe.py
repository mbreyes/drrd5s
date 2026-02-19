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

PREFIXES = ['BB1', 'BB2', 'BB3', 'BB4', 'BB5', 'BB6', 'BB7']
DATA_PATH = os.path.realpath('../../data/raw/BB/')+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'



#----BB1----

LAST_SESSION = 2
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bb= [105,106,107,108,109,110,111,112,113,114,115,116]

dr.exp_dataframe_timeout(PREFIXES[0],bb, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)

#----BB2-----

LAST_SESSION = 21
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)

dr.exp_dataframe(PREFIXES[1],bb, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)

#-----B4------

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bb= [105,106,108,109,111,113,114,115]

dr.exp_dataframe(PREFIXES[3],bb, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)

#-----B5-----

dr.exp_dataframe_timeout(PREFIXES[4],bb, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)

#-----B6-----

bb = [105,106,108,109,113,114,115]
dr.exp_dataframe(PREFIXES[5],bb, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS = [1])

#----B7-----

dr.exp_dataframe(PREFIXES[6],bb, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS=[1])


