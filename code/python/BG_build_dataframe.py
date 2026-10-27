#%%
from datetime import datetime
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIXES = ['BG1', 'BG2', 'BG3', 'BG4', 'BG5']
DATA_PATH = os.path.realpath("../../data/raw/BG/")+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'
ABOVE = 5 # in seconds


#%%
# ---- BG1 -----

LAST_SESSION = 20
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bg = [135,136,137,138,139,140]

dr.exp_dataframe(PREFIX= 'BG1', ALL_SESSIONS=ALL_SESSIONS, RATS=bg, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)


#%%
#---- BG2 -----

LAST_SESSION = 8
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bg2 = [135,136,137,139,140]

dr.exp_dataframe(PREFIX= 'BG2', ALL_SESSIONS=ALL_SESSIONS, RATS=bg2, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)



# %%

# ---- BG3 -----

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bg3 = [135,136,137,139,140]

dr.exp_dataframe(PREFIX= 'BG3', ALL_SESSIONS=ALL_SESSIONS, RATS=bg3, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

# %%

# ------ BG4 -----

LAST_SESSION = 7
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bg4 = [135,136,137,139,140]

dr.exp_dataframe(PREFIX= 'BG4', ALL_SESSIONS=ALL_SESSIONS, RATS=bg4, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

# %%

# ------- BG5 -----

LAST_SESSION = 3
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
bg5 = [135,136,137,139,140]

dr.exp_dataframe(PREFIX= 'BG5', ALL_SESSIONS=ALL_SESSIONS, RATS=bg5, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

# %%
