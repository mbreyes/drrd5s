
from datetime import datetime
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIXES = ['AY1', 'AY2', 'AY3', 'AY4','AY5']
DATA_PATH = os.path.realpath("../../data/raw/AY/")+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'
ABOVE = 5 # in seconds

# ---- AY1 -----

LAST_SESSION = 21
ALL_SESSIONS = np.setdiff1d(np.arange(1,LAST_SESSION+1),[5])
ay = [51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74]

dr.exp_dataframe(PREFIX= 'AY1', ALL_SESSIONS=ALL_SESSIONS, RATS=ay, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

#---- AY3 -----

LAST_SESSION = 5
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
aY3 = [52,53,55,56,58,59,60,61,63,64,65,66,67,68,69,71,72,74]

dr.exp_dataframe(PREFIX= 'AY3', ALL_SESSIONS=ALL_SESSIONS, RATS=aY3, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

#---- AY4 -----

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
aY4 = [52,55,56,58,59,60,61,63,64,65,66,68,69,71,72,74]

dr.exp_dataframe_timeout(PREFIX= 'AY4', ALL_SESSIONS=ALL_SESSIONS, RATS=aY4, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

#---- AY5 -----

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
aY5 = [52,55,56,58,59,60,61,63,64,65,66,68,69,71,72,74]

dr.exp_dataframe_timeout(PREFIX= 'AY5', ALL_SESSIONS=ALL_SESSIONS, RATS=aY5, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)
