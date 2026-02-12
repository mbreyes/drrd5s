#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
sns.set(style='ticks')

# PREFIX = 'AY5'
DATA_PATH = '../data/processed/'
OUTPUT_PATH = '../output/temp/'
ABOVE = 10 # in seconds
LAST_SESSION = 6

rats_g1 = [63,64,65,66,68,71,72,74] # grupo 5s
rats_g2 = [52,55,56,58,59,60,61,69] # grupo 10s

rats = rats_g1 + rats_g2
rats.sort()


df4 = pd.read_csv(DATA_PATH+'AY4.csv')
df5 = pd.read_csv(DATA_PATH+'AY5.csv')

df = pd.concat((df4,df5),ignore_index= True)

df.to_csv(DATA_PATH+'AY4+5_tentative.csv', index=False)
