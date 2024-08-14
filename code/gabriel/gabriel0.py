#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:09:05 2024

@author: mbreyes
"""

import pandas as pd
import matplotlib as plt

DATA_PATH = '../'

    # Vale lembra que em last_sesion deve ser colocado o valor seguinte
    # Exemplo: se a última sessão registrada foi a 8, então last_session = 9

def extrai_dados(rat,last_session):
    for i in range(1, last_session):
        if i==1:
            rato1 = pd.read_excel(f"/home/gabriel/python/{rat}.xlsx", sheet_name= f'session {i}')
            rato1 = rato1.drop(range(15), axis=0)
            rato1 = rato1.rename(columns={'date':'trial','Unnamed: 4': f'outcome{i}'})
            rat = rato1[['trial', f'outcome{i}']]
        else:
            rato1 = pd.read_excel(f"/home/gabriel/python/{rat}.xlsx", sheet_name= f'session {i}')
            rato1 = rato1.drop(range(15), axis=0)
            rato1 = rato1.rename(columns={'date':'trial','Unnamed: 4': f'outcome{i}'})
            rat[f'outcome{i}'] = rato1[f'outcome{i}']
    return rat
    
rato1_df = extrai_dados("AT001", 9)
