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

PREFIXES = ['BD2']
DATA_PATH = os.path.realpath('../../data/raw/BD/')+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'



#----BD2----

LAST_SESSION = 4
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bd= [117,118,119,120,121,122]

dr.exp_dataframe(PREFIXES[0],bd, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)




