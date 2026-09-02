
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

#---- AY2 -----
# Each rat did a different number of AY2 sessions (58 only has 4, 72 only has 2,
# the rest have 6) -- exp_dataframe already skips missing session files per rat,
# so a single ALL_SESSIONS upper bound covering the longest rat is enough.

LAST_SESSION = 6
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)
ay2 = [51,54,57,58,62,70,72,73]

dr.exp_dataframe(PREFIX= 'AY2', ALL_SESSIONS=ALL_SESSIONS, RATS=ay2, Data_Path=DATA_PATH, Output_Path=OUTPUT_PATH)

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

#---- Join AY1 + AY2, treating AY2 sessions as a continuation of AY1 ----
# AY1.csv is the already-reviewed/finalized dataframe; AY2_tentative.csv is
# whatever the exp_dataframe call above just produced -- review it before
# trusting AY12.csv. Each rat's AY2 session numbers (which restart at 1, like
# every other phase) are shifted by that same rat's own last AY1 session, so
# rats with a different number of AY1/AY2 sessions are all handled correctly.

ay1 = pd.read_csv(OUTPUT_PATH + 'AY1.csv').astype({'rat': int, 'session': int})
ay2 = pd.read_csv(OUTPUT_PATH + 'AY2_tentative.csv').astype({'rat': int, 'session': int})

last_ay1_session = ay1.groupby('rat')['session'].max()

ay2 = ay2.copy()
ay2['session'] = ay2['rat'].map(last_ay1_session) + ay2['session']

if ay2['session'].isna().any():
    missing_rats = sorted(ay2.loc[ay2['session'].isna(), 'rat'].unique())
    raise ValueError(f'AY2 rat(s) with no matching AY1 session data: {missing_rats}')

ay12 = pd.concat([ay1, ay2], ignore_index=True)
ay12 = ay12.astype({'session': int})
ay12.to_csv(OUTPUT_PATH + 'AY12.csv', index=False)
