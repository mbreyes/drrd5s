
from datetime import datetime
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIXES = ['BG1']
DATA_PATH = os.path.realpath("../../data/raw/BG/")+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'
ABOVE = 5 # in seconds

# ---- BG1 -----

LAST_SESSION = 14
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bg = [135,136,137,138,139,140]

dr.exp_dataframe(PREFIX= 'BG1', ALL_SESSIONS=ALL_SESSIONS, RATS=bg, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

#---- BG2 -----

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)

#dr.exp_dataframe(PREFIX= 'BG2', ALL_SESSIONS=ALL_SESSIONS, RATS=bg, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)


