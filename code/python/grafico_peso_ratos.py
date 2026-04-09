#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:16:54 2023

@author: Marcelo B. Reyes - UFABC
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# first day the rat was trained
FIRST_DAY = datetime.date(2026,3,19)

# number of day in the graph
N_DAYS = 60


R117= [345,340,330,330]
R118= [340,340,330,325]
R119= [330,330,325,320]
R120= [335,330,325,320]
R121= [315,310,305,305]
R122= [330,330,325,320]


# all rats
ALL_RATS = [R117,R118,R119,R120,R121,R122]

# range of weights to plot
#MIN_WEIGHT = 290
#MAX_WEIGHT = MIN_WEIGHT + 150


def build_days_for_x_tick_lables(fd, N= 60):
    l = []                          # initiates list with all labels for x axis
    for i in range(0, N+1, 7):         # loop for all days
        if i % 7 == 0:              # every 7 days repeats the weekday
            fd = FIRST_DAY + datetime.timedelta(days=i)
            l.append(str(fd.day)+'/'+str(fd.month))            # adds to list
        # else:
        #     l.append('')            # otherwise, ommits the label
    return(l)

def build_graph(data, rat, N_DAYS, MIN_WEIGHT, should_save= False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # plotting weights
    per_90 = data[0]*.90
    per_85 = data[0]*.85
    plt.axhline(y = per_90,color = 'r', linestyle = '--')
    plt.axhline(y = per_85, color = 'r', linestyle = '--')
    plt.xlim(-1,N_DAYS)
    plt.ylim(MIN_WEIGHT,MAX_WEIGHT)
    plt.xlabel('dia')
    plt.ylabel('massa (g)')
    plt.title(f'Pesagem diária rato {rat}')
    
    # Major ticks for x axis
    y_major_ticks = np.arange(MIN_WEIGHT, MAX_WEIGHT+1, 10)
    y_minor_ticks = np.arange(MIN_WEIGHT, MAX_WEIGHT+1,  1)
    
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)
    
    # Major ticks for x axis
    x_major_ticks = np.arange(0, N_DAYS+1, 7)
    x_minor_ticks = np.arange(0, N_DAYS+1, 1)
    
    labels = build_days_for_x_tick_lables(FIRST_DAY, N= N_DAYS)
    
    ax.set_xticks(x_major_ticks, labels= labels, fontsize=10)
    ax.set_xticks(x_minor_ticks, minor=True)
    
    # And a corresponding grid
    ax.grid(which='both')
    
    # adding vertical lines for saturdays and sundays
    for i in range(N_DAYS):
        day = FIRST_DAY + datetime.timedelta(days=i)
        if day.weekday() in [5,6]:
            plt.axvline(x=i, alpha=0.8,color='r', lw= 0.5, ls='--')
    
    
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=1)
    
    plt.plot(data,'ko', ms=3)
    plt.savefig(os.path.realpath(f'../../output/tmp/pesos/peso_rato{rat}_temp.pdf'))
    
   # if should_save:
    #    plt.savefig(f'rat{rat}.pdf')
    plt.show()
    

# -------- main ----------

for rat,data in enumerate(ALL_RATS):
    MIN_WEIGHT = round(data[0]*.80/5)*5
    MAX_WEIGHT = round((data[0]+8)/5)*5
    build_graph(data, rat+117, N_DAYS, MIN_WEIGHT, should_save=True)
    
    
    
