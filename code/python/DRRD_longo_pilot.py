#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ailacamara
"""
#%%
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import model_fitting as mf

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIX = 'BG1'
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_longo_pilot/')+'/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Limite superior (s) do eixo de tempo nos ajustes da dupla gaussiana em escala log
T_MAX_LOG = 15

#%%
#---- LOADING DATA ----

data = pd.read_csv(DATA_PATH + f'{PREFIX}_tentative.csv')
data = data.astype({'rat':int,'session':int,'reinforced':int})

ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order
COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

df = data.copy()

sessions = sorted(df.session.unique())

#%%
# ------ Taxa de acerto (proporção de respostas reforçadas) ao longo das sessões, por animal ------

mf.proportion_correct_responses_by_rat(df, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------ Duração média das respostas ao longo das sessões, por animal ------

mf.session_metric_by_rat(df, metric='mean_duration', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------ Número de tentativas por animal para atingir os critérios 1s, 2s, 3s, 4s e 5s ------

bg_trials_to_criteria = mf.trials_reach_criteria_by_rat(df, criteria=[1, 2, 3, 4, 5], \
                                                          PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

mf.sessions_reach_criteria_by_rat(df, criteria=[1, 2, 3, 4, 5], \
                                  PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)


#%%
# ------- Fit Double Gaussian to Log of Duration Distribution -------

# ------- MLE fitting, per rat and per individual session -------

bg_parameters = mf.fit_params_all_sessions(df, mf.MLE, log=True)
bg_parameters.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Parameters.csv', index=False)

#%%
# -------- Parâmetros da dupla gaussiana por animal (lineplot), um subplot por parâmetro --------

mf.plot_parameters_lineplot_by_rat(bg_parameters, order=sessions, \
                           PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# ------- MLE fitting em escala linear, per rat and per individual session -------

bg_parameters_lin = mf.fit_params_all_sessions(df, mf.MLE, log=False)
bg_parameters_lin.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_linear_Parameters.csv', index=False)

#%%
# -------- Parâmetros da dupla gaussiana (escala linear) por animal (lineplot) --------

mf.plot_parameters_lineplot_by_rat(bg_parameters_lin, order=sessions, log=False, \
                           PREFIX=f'{PREFIX}_linear', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- Ajuste da dupla gaussiana por animal, um subplot por sessão --------

for rat in sorted(df.rat.unique()):
    df_rat = df.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df_rat, log=True, t_max=T_MAX_LOG, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- Ajuste da dupla gaussiana (escala linear) por animal, um subplot por sessão --------

for rat in sorted(df.rat.unique()):
    df_rat = df.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df_rat, log=False, PREFIX=f'{PREFIX}_linear', OUTPUT_PATH=OUTPUT_PATH)

#%%
# ======================================================================
# ---- SEGUNDA FASE: BG2 ----
# ======================================================================

PREFIX2 = 'BG2'

data2 = pd.read_csv(DATA_PATH + f'{PREFIX2}_tentative.csv')
data2 = data2.astype({'rat':int,'session':int,'reinforced':int})

df2 = data2.copy()

sessions2 = sorted(df2.session.unique())

#%%
# ------ BG2: Taxa de acerto ao longo das sessões, por animal ------

mf.proportion_correct_responses_by_rat(df2, PREFIX=PREFIX2, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------ BG2: Duração média das respostas ao longo das sessões, por animal ------

mf.session_metric_by_rat(df2, metric='mean_duration', PREFIX=PREFIX2, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------- BG2: MLE fitting, per rat and per individual session -------

bg2_parameters = mf.fit_params_all_sessions(df2, mf.MLE, log=True)
bg2_parameters.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX2}_Parameters.csv', index=False)

#%%
# -------- BG2: Parâmetros da dupla gaussiana por animal (lineplot) --------

mf.plot_parameters_lineplot_by_rat(bg2_parameters, order=sessions2, \
                           PREFIX=PREFIX2, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# ------- BG2: MLE fitting em escala linear, per rat and per individual session -------

bg2_parameters_lin = mf.fit_params_all_sessions(df2, mf.MLE, log=False)
bg2_parameters_lin.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX2}_linear_Parameters.csv', index=False)

#%%
# -------- BG2: Parâmetros da dupla gaussiana (escala linear) por animal (lineplot) --------

mf.plot_parameters_lineplot_by_rat(bg2_parameters_lin, order=sessions2, log=False, \
                           PREFIX=f'{PREFIX2}_linear', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG2: Ajuste da dupla gaussiana por animal, um subplot por sessão --------

for rat in sorted(df2.rat.unique()):
    df2_rat = df2.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df2_rat, log=True, t_max=T_MAX_LOG, PREFIX=PREFIX2, OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- BG2: Ajuste da dupla gaussiana (escala linear) por animal, um subplot por sessão --------

for rat in sorted(df2.rat.unique()):
    df2_rat = df2.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df2_rat, log=False, PREFIX=f'{PREFIX2}_linear', OUTPUT_PATH=OUTPUT_PATH)

#%%
# ======================================================================
# ---- BG1 + BG2: Taxa de acerto por animal nas duas fases ----
# ======================================================================

mf.proportion_correct_responses_by_rat_phases({PREFIX: df, PREFIX2: df2}, \
                                              PREFIX=f'{PREFIX}_{PREFIX2}', OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- BG1 + BG2: Parâmetros da dupla gaussiana por animal nas duas fases --------

mf.plot_parameters_lineplot_by_rat_phases({PREFIX: bg_parameters, PREFIX2: bg2_parameters}, \
                                          PREFIX=f'{PREFIX}_{PREFIX2}', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG1 + BG2: Parâmetros da dupla gaussiana (escala linear) por animal nas duas fases --------

mf.plot_parameters_lineplot_by_rat_phases({PREFIX: bg_parameters_lin, PREFIX2: bg2_parameters_lin}, log=False, \
                                          PREFIX=f'{PREFIX}_{PREFIX2}_linear', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# ======================================================================
# ---- TERCEIRA FASE: BG3 (testes: salina / muscimol / nada) ----
# ======================================================================

PREFIX3 = 'BG3'

data3 = pd.read_csv(DATA_PATH + f'{PREFIX3}_tentative.csv')
data3 = data3.astype({'rat':int,'session':int,'reinforced':int})

df3 = data3.copy()

sessions3 = sorted(df3.session.unique())

# Tratamento de cada sessão do BG3
BG3_TREATMENTS = {1: 'Saline', 2: 'Muscimol', 3: 'None', 4: 'None', 5: 'Saline', 6: 'Muscimol'}

#%%
# ------ BG2 + BG3: baseline vs. testes, por animal ------
# Número total de respostas, proporção de respostas corretas,
# proporção de respostas prematuras (< 500 ms) em relação ao total de respostas emitidas
# e duração média das respostas

for metric in ['responses', 'proportion_correct', 'premature', 'mean_duration']:
    mf.session_metric_by_rat_phases_treatment({PREFIX2: df2, PREFIX3: df3}, metric=metric, \
                                              treatments={PREFIX3: BG3_TREATMENTS}, \
                                              phase_labels={PREFIX2: 'Baseline', PREFIX3: 'Tests'}, \
                                              PREFIX=f'{PREFIX2}_{PREFIX3}', OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------- BG3: MLE fitting, per rat and per individual session -------

bg3_parameters = mf.fit_params_all_sessions(df3, mf.MLE, log=True)
bg3_parameters.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX3}_Parameters.csv', index=False)

#%%
# ------- BG3: MLE fitting em escala linear, per rat and per individual session -------

bg3_parameters_lin = mf.fit_params_all_sessions(df3, mf.MLE, log=False)
bg3_parameters_lin.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX3}_linear_Parameters.csv', index=False)

#%%
# -------- BG2 + BG3: cada parâmetro da dupla gaussiana por animal, baseline vs. testes --------
# Um gráfico por parâmetro: as 8 sessões do BG2 (baseline) seguidas das sessões do BG3

mf.plot_parameters_by_rat_phases_treatment({PREFIX2: bg2_parameters, PREFIX3: bg3_parameters}, \
                                           treatments={PREFIX3: BG3_TREATMENTS}, \
                                           phase_labels={PREFIX2: 'Baseline', PREFIX3: 'Tests'}, \
                                           PREFIX=f'{PREFIX2}_{PREFIX3}', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG2 + BG3: cada parâmetro da dupla gaussiana (escala linear) por animal, baseline vs. testes --------

mf.plot_parameters_by_rat_phases_treatment({PREFIX2: bg2_parameters_lin, PREFIX3: bg3_parameters_lin}, \
                                           treatments={PREFIX3: BG3_TREATMENTS}, \
                                           phase_labels={PREFIX2: 'Baseline', PREFIX3: 'Tests'}, log=False, \
                                           PREFIX=f'{PREFIX2}_{PREFIX3}_linear', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG1 + BG2 + BG3: acumulado de todas as fases, cada parâmetro da dupla gaussiana por animal --------
# Um gráfico por parâmetro: sessões do BG1, do BG2 (baseline) e do BG3 em sequência,
# com as sessões de teste do BG3 sombreadas pela cor do tratamento

mf.plot_parameters_by_rat_phases_treatment({PREFIX: bg_parameters, PREFIX2: bg2_parameters, PREFIX3: bg3_parameters}, \
                                           treatments={PREFIX3: BG3_TREATMENTS}, \
                                           phase_labels={PREFIX2: 'Baseline', PREFIX3: 'Tests'}, \
                                           PREFIX=f'{PREFIX}_{PREFIX2}_{PREFIX3}', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG1 + BG2 + BG3: acumulado de todas as fases, parâmetros da dupla gaussiana (escala linear) --------

mf.plot_parameters_by_rat_phases_treatment({PREFIX: bg_parameters_lin, PREFIX2: bg2_parameters_lin, PREFIX3: bg3_parameters_lin}, \
                                           treatments={PREFIX3: BG3_TREATMENTS}, \
                                           phase_labels={PREFIX2: 'Baseline', PREFIX3: 'Tests'}, log=False, \
                                           PREFIX=f'{PREFIX}_{PREFIX2}_{PREFIX3}_linear', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG3: Ajuste da dupla gaussiana por animal, um subplot por sessão --------
# O primeiro subplot é o baseline: todas as sessões do BG2 do animal agrupadas em um único ajuste

for rat in sorted(df3.rat.unique()):
    df3_rat = df3.query('rat == @rat').reset_index(drop=True)
    df2_rat = df2.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df3_rat, log=True, t_max=T_MAX_LOG, baseline_data=df2_rat, \
                                         baseline_label=f'Baseline ({PREFIX2}, all sessions)', \
                                         PREFIX=PREFIX3, OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- BG3: Ajuste da dupla gaussiana (escala linear) por animal, um subplot por sessão --------
# O primeiro subplot é o baseline: todas as sessões do BG2 do animal agrupadas em um único ajuste

for rat in sorted(df3.rat.unique()):
    df3_rat = df3.query('rat == @rat').reset_index(drop=True)
    df2_rat = df2.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df3_rat, log=False, baseline_data=df2_rat, \
                                         baseline_label=f'Baseline ({PREFIX2}, all sessions)', \
                                         PREFIX=f'{PREFIX3}_linear', OUTPUT_PATH=OUTPUT_PATH)

#%%
# ======================================================================
# ---- QUARTA FASE: BG4 (novo treino: critério reduzido de 5s para 1.5s) ----
# ======================================================================

PREFIX4 = 'BG4'

data4 = pd.read_csv(DATA_PATH + f'{PREFIX4}_tentative.csv')
data4 = data4.astype({'rat':int,'session':int,'reinforced':int})

df4 = data4.copy()

sessions4 = sorted(df4.session.unique())

# Bins (histograma e curva) até a maior duração + UPPER_OFFSET s (mf.data_histogram_max),
# em vez do range fixo 0-30s, para o critério de 1.5s
UPPER_OFFSET = 2

#%%
# ------ BG4: Taxa de acerto ao longo das sessões, por animal ------

mf.proportion_correct_responses_by_rat(df4, PREFIX=PREFIX4, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------ BG4: Número total de respostas, proporção de respostas prematuras (< 500 ms)
# e duração média das respostas ao longo das sessões, por animal ------

for metric in ['responses', 'premature', 'mean_duration']:
    mf.session_metric_by_rat(df4, metric=metric, PREFIX=PREFIX4, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------- BG4: MLE fitting em escala log, per rat and per individual session -------

PREFIX4_LOG = f'{PREFIX4}_log'

bg4_parameters_log = mf.fit_params_all_sessions(df4, mf.MLE, log=True)
bg4_parameters_log.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX4_LOG}_Parameters.csv', index=False)

#%%
# -------- BG4 (log): Parâmetros da dupla gaussiana por animal (lineplot) --------

mf.plot_parameters_lineplot_by_rat(bg4_parameters_log, order=sessions4, log=True, \
                           PREFIX=PREFIX4_LOG, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG4 (log): Ajuste da dupla gaussiana por animal, um subplot por sessão --------

for rat in sorted(df4.rat.unique()):
    df4_rat = df4.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df4_rat, log=True, upper_offset=UPPER_OFFSET, \
                                         PREFIX=PREFIX4_LOG, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------- BG4: MLE fitting em escala linear, per rat and per individual session -------

PREFIX4_LIN = f'{PREFIX4}_linear'

bg4_parameters_lin = mf.fit_params_all_sessions(df4, mf.MLE, log=False)
bg4_parameters_lin.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX4_LIN}_Parameters.csv', index=False)

#%%
# -------- BG4 (linear): Parâmetros da dupla gaussiana por animal (lineplot) --------

mf.plot_parameters_lineplot_by_rat(bg4_parameters_lin, order=sessions4, log=False, \
                           PREFIX=PREFIX4_LIN, OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG4 (linear): Ajuste da dupla gaussiana por animal, um subplot por sessão --------

for rat in sorted(df4.rat.unique()):
    df4_rat = df4.query('rat == @rat').reset_index(drop=True)
    mf.plot_fitting_mle_sessions_per_rat(df4_rat, log=False, upper_offset=UPPER_OFFSET, \
                                         PREFIX=PREFIX4_LIN, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ======================================================================
# ---- QUINTA FASE: BG5 (testes: nada / salina / muscimol) ----
# ======================================================================

PREFIX5 = 'BG5'

data5 = pd.read_csv(DATA_PATH + f'{PREFIX5}_tentative.csv')
data5 = data5.astype({'rat':int,'session':int,'reinforced':int})

df5 = data5.copy()

sessions5 = sorted(df5.session.unique())

# Tratamento de cada sessão do BG5 (sessões ainda não realizadas já ficam sombreadas
# com a cor do tratamento, definida em mf.TREATMENT_COLORS)
BG5_TREATMENTS = {1: 'None', 2: 'Saline', 3: 'Muscimol'}

# Escalas do ajuste da dupla gaussiana: {sufixo do PREFIX: log}
SCALES = {'log': True, 'linear': False}

#%%
# ------ BG5: métricas por sessão, por animal ------
# Número total de respostas, proporção de respostas corretas,
# proporção de respostas prematuras (< 500 ms) e duração média das respostas

for metric in ['responses', 'proportion_correct', 'premature', 'mean_duration']:
    mf.session_metric_by_rat_phases_treatment({PREFIX5: df5}, metric=metric, \
                                              treatments={PREFIX5: BG5_TREATMENTS}, \
                                              phase_labels={PREFIX5: 'Tests'}, \
                                              PREFIX=PREFIX5, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ------- BG5: MLE fitting em escala log e linear, per rat and per individual session -------

bg5_parameters = {}
for scale, log in SCALES.items():
    bg5_parameters[scale] = mf.fit_params_all_sessions(df5, mf.MLE, log=log)
    bg5_parameters[scale].to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX5}_{scale}_Parameters.csv', index=False)

#%%
# -------- BG5: cada parâmetro da dupla gaussiana por animal, sessões sombreadas pelo tratamento --------

for scale, log in SCALES.items():
    mf.plot_parameters_by_rat_phases_treatment({PREFIX5: bg5_parameters[scale]}, \
                                               treatments={PREFIX5: BG5_TREATMENTS}, \
                                               phase_labels={PREFIX5: 'Tests'}, log=log, \
                                               PREFIX=f'{PREFIX5}_{scale}', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG5: Ajuste da dupla gaussiana por animal, um subplot por sessão (log e linear) --------

for scale, log in SCALES.items():
    for rat in sorted(df5.rat.unique()):
        df5_rat = df5.query('rat == @rat').reset_index(drop=True)
        mf.plot_fitting_mle_sessions_per_rat(df5_rat, log=log, upper_offset=UPPER_OFFSET, \
                                             session_labels=BG5_TREATMENTS, \
                                             PREFIX=f'{PREFIX5}_{scale}', OUTPUT_PATH=OUTPUT_PATH)

#%%
# ======================================================================
# ---- BG4 (baseline) + BG5 (testes) ----
# ======================================================================

#%%
# ------ BG4 + BG5: baseline vs. testes, por animal ------
# Número total de respostas, proporção de respostas corretas,
# proporção de respostas prematuras (< 500 ms) e duração média das respostas

for metric in ['responses', 'proportion_correct', 'premature', 'mean_duration']:
    mf.session_metric_by_rat_phases_treatment({PREFIX4: df4, PREFIX5: df5}, metric=metric, \
                                              treatments={PREFIX5: BG5_TREATMENTS}, \
                                              phase_labels={PREFIX4: 'Baseline', PREFIX5: 'Tests'}, \
                                              PREFIX=f'{PREFIX4}_{PREFIX5}', OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- BG4 + BG5: cada parâmetro da dupla gaussiana por animal, baseline vs. testes (log e linear) --------

bg4_parameters = {'log': bg4_parameters_log, 'linear': bg4_parameters_lin}

for scale, log in SCALES.items():
    mf.plot_parameters_by_rat_phases_treatment({PREFIX4: bg4_parameters[scale], PREFIX5: bg5_parameters[scale]}, \
                                               treatments={PREFIX5: BG5_TREATMENTS}, \
                                               phase_labels={PREFIX4: 'Baseline', PREFIX5: 'Tests'}, log=log, \
                                               PREFIX=f'{PREFIX4}_{PREFIX5}_{scale}', OUTPUT_PATH=OUTPUT_PATH, fit_type='MLE')

#%%
# -------- BG4 + BG5: Ajuste da dupla gaussiana por animal, um subplot por sessão (log e linear) --------
# O primeiro subplot é o baseline: todas as sessões do BG4 do animal agrupadas em um único ajuste;
# os demais são as sessões do BG5, separadas

for scale, log in SCALES.items():
    for rat in sorted(df5.rat.unique()):
        df5_rat = df5.query('rat == @rat').reset_index(drop=True)
        df4_rat = df4.query('rat == @rat').reset_index(drop=True)
        mf.plot_fitting_mle_sessions_per_rat(df5_rat, log=log, upper_offset=UPPER_OFFSET, \
                                             session_labels=BG5_TREATMENTS, baseline_data=df4_rat, \
                                             baseline_label=f'Baseline ({PREFIX4}, all sessions)', \
                                             PREFIX=f'{PREFIX4}_{PREFIX5}_{scale}', OUTPUT_PATH=OUTPUT_PATH)

# %%
