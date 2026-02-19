
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')
sns.set(style='ticks')

PREFIX = 'AW1'
DATA_PATH = os.path.realpath('../../data/raw/AW/')+'/'
OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'

rats = [39,40,41,42,43,44,45,46,47,48,49,50] 
LAST_SESSION = 19
ALL_SESSIONS = np.arange(1,LAST_SESSION+1)

dr.exp_dataframe(PREFIX, rats, DATA_PATH, OUTPUT_PATH, ALL_SESSIONS)


