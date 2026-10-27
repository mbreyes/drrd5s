#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:43:12 2024

@author: mbreyes
"""
#%%
from mimetypes import init
import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import drrd_functions as drrd
import matplotlib.pyplot as plt
import os
import pingouin as pg
from scipy.optimize import curve_fit
from scipy.stats import shapiro
import model_fitting as mf
import cure_model as cm

PREFIX = 'DRRD_10s_Groups'

sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
plt.rcParams['figure.dpi'] = 500

sns.set_palette(['#5B9E95', '#C5AA97'])
sns.set_context("paper")

PREFIXES = ['AU','AW1','AY12']
DATA_PATH = os.path.realpath('../../data/processed/')+'/'
OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_10s_Groups/')+'/'
LAST_SESSION = 19
ANIMALS = [rat for rat in list(range(15,27)) + list(range(39,75))]

init_params = [
    0.5,  # gamma
    0.05,  # mu1
    0.,  # sigma1
    5.0,  # mu2
    0.5   # sigma2
]

bin_width = 0.1
n_bins = 60

#%%
#---- LOADING DATA ----
data = None

for prefix in PREFIXES:
    if data is None:
        data = pd.read_csv(DATA_PATH +f'{prefix}.csv')
    else:
        data = pd.concat([data, pd.read_csv(DATA_PATH +f'{prefix}.csv')], ignore_index=True)


ALL_GROUPS = sorted(data['group'].unique())  # fixed, canonical order
COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

data1 = data.astype({'rat':int,'session':int,'reinforced':int})
data2 = data1.query('rat in @ANIMALS').reset_index(drop=True)
data3 = data2.query('duration < 30').reset_index(drop=True)

df = data3.copy()

#%%

# ============================================================
# EXPERIMENTO AU
# ============================================================

PREFIX = 'AU DRRD'
RAT_AU = list(range(15,27))
df_au = df.query('rat in @RAT_AU and session <= @LAST_SESSION').reset_index(drop=True)
df_successful = df_au.query('criterion == 10').rat.unique()
df_au_un = df.query('rat in @RAT_AU').reset_index(drop=True)
#%%
# ----- Mean Response Duration ---------

mf.mean_response_duration(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Distribuicao da duracao media por rato, sessoes-chave -----

mf.boxplot_groups_key_sessions(df_au, key_sessions=[2, 10, 19], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Distribuicao da duracao media por rato, primeira sessao (Early vs Late) -----

mf.boxplot_first_session_early_late(df_au, percentage_trials=33, n_example_rats = 4,PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista (R): duracao media ~ grupo * (primeiro vs ultimo terco da 1a sessao) -----

mf.run_mixed_anova_first_session_early_late(df_au, percentage_trials=33, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Duracao por trial, ultima sessao (30 primeiras e 30 ultimas tentativas), por rato -----

mf.duration_by_trial_session(df_au, n_trials=30, session=19, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Pressupostos da ANOVA (R): normalidade (Shapiro-Wilk) e homogeneidade (Levene) -----

mf.run_anova_assumptions(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista (R): duracao media ~ grupo * sessao -----

mf.run_mixed_anova(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista nao-parametrica (nparLD): grupo x sessao, com post-hoc entre grupos -----

mf.run_nparld_anova(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Learned vs Did Not Learn classification + Fisher's exact test (R): grupo x aprendeu -----
# Usa df_au_un (todas as sessoes do rato, sem o corte em LAST_SESSION) para nao
# descartar sessoes pos-criterio que ocorram depois da sessao 19. resolve_original_group
# nao tem efeito aqui se nenhum rato do experimento AU trocar de grupo, mas e aplicado
# por consistencia com o experimento AY/AW.

df_au_learned_input = mf.resolve_original_group(df_au_un)
au_learned_matrix, au_learned_fisher = mf.learned_fisher_test(df_au_learned_input, crit=10, min_reinforced_rate=0.05, \
                                                           PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ----- Number of sessions to reach criteria -----

mf.sessions_reach_criteria(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Number of Trials to reach criteria ------

mf.trials_reach_criteria(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA de uma via (R): tentativas cumulativas ate atingir 10s, entre grupos -----

mf.run_oneway_anova_trials_to_criterion(df_au, crit=10, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Percentage of Rats that reached criteria --------


#mf.percentage_rats_criteria(df_au, crit = 5, criterion = '5s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)


#mf.percentage_rats_criteria(df_au, crit = 10, criterion = '10s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)


# ------ Boxplot of the last three sessions for each group --------

#mf.boxplot_groups(df_au, PREFIX = PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ------ Number of rats that achived maximum criteria ---------

n_rats = df_au.query('criterion ==10').groupby(['group']).rat.nunique().reset_index(name= 'n_rats')

# ------ Proportion of correct responses ------

#mf.proportion_correct_responses(df_au, PREFIX = PREFIX, OUTPUT_PATH=OUTPUT_PATH)

mf.proportion_correct_responses_by_criterion(df_au, crit = 10, PREFIX = PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ------ Criterion progression per session ------

#mf.criterion_progression(df_au, PREFIX = PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ------ Mixture cure model: reach criterion 10s while keeping reinforced rate > 5% ------

# au_cure_fit = cm.fit_criterion_cure_model(df_au_un, crit = 10, min_reinforced_rate = 0.05, verbose = True)
# print(au_cure_fit['b'])
# print(au_cure_fit['beta'])
# print(cm.cure_probability_by_group(au_cure_fit))

# cm.plot_population_survival(au_cure_fit, times = list(range(1,LAST_SESSION+1)), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
# cm.plot_cure_probability_bar(au_cure_fit, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# # ------ Kaplan-Meier curve (same event definition as the cure model) ------

# cm.plot_kaplan_meier(df_au_un, crit = 10, min_reinforced_rate = 0.05, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, \
#                       cure_fit_result = au_cure_fit, times = list(range(1,LAST_SESSION+1)))



#%%
# ------- Fit Doble Gaussian to Log of Duration Distribution --------

# ------- MLE and LSF fitting, per rat and per individual session (1 - 19) ----------

au_parameters = mf.fit_params_all_sessions(df_au_un, mf.MLE, log=True)
au_parameters.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Parameters.csv', index=False)

#au_parameters_lsf = mf.fit_params_all_sessions(df_au_un, mf.LSF, log=True)
#au_parameters_lsf.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Parameters_LSF.csv', index=False)

#%%
# -------- Plot Parameters for each group ----------

mf.plot_parameters_boxplot(au_parameters, order = list(range(1,20)), \
                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_parameters_lineplot(au_parameters, order = list(range(1,20)), \
 #                          PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#for param in ['mu1', 'mu2', 'sigma1', 'sigma2', 'gamma']:
 #   mf.plot_param_method_comparison_boxplot(au_parameters, au_parameters_lsf, param, order = list(range(1,20)), \
  #                         PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#mf.plot_gamma_vs_gamma2_boxplot(au_parameters, order = list(range(1,20)), \
 #                          PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#%%
# ------ Criterion-tracking index: i = (C - mu2) / sigma2, per rat/session ------

au_criterion_index = mf.calculate_criterion_tracking_index(au_parameters, df_au_un, log = True)

mf.plot_criterion_tracking_index_lineplot(au_criterion_index, order = list(range(1,20)), \
                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#%%
# -------- Histogram Fit (MLE), per group, first 3 vs last 3 sessions --------

#early = [1, 2, 3]
#late = [17, 18, 19]

# for group in df_au.group.unique():

#     df_group_early = df_au.loc[(df_au['group'] == group) & (df_au['session'].isin(early))].reset_index(drop=True)
#     mf.plot_fitting_mle_rats(df_group_early, session = str(early), log = True, \
#                               PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#     df_group_late = df_au.loc[(df_au['group'] == group) & (df_au['session'].isin(late))].reset_index(drop=True)
#     mf.plot_fitting_mle_rats(df_group_late, session = str(late), log = True, \
#                               PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- Histogram Fit (MLE, dados sem log), por grupo, sessao 19, por rato --------

for group in df_au.group.unique():

    df_group_1 = df_au.loc[(df_au['group'] == group) & (df_au['session'] == 1)].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_1, session = '1', log = False, ceiling = 7,\
                              PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
    df_group_2 = df_au.loc[(df_au['group'] == group) & (df_au['session'] == 2)].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_2, session = '2', log = False, ceiling = 10, \
                                  PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
        

#%%
# -------- Histograma da duracao (sem ajuste), por grupo, sessao 19, por rato --------

mf.plot_duration_histogram_rats(df_au, session=19, bin_width=0.2, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%


# ============================================================
# EXPERIMENTOS AW E AY (analisados em conjunto)
# ============================================================

PREFIX = 'AY and AW DRRD'
RAT_AY = list(range(39,75))
RAT_AY = [rat for rat in RAT_AY if rat != 50]
sessions = np.setdiff1d(np.arange(1,20),[5])

df_ay_un = df.query('rat in @RAT_AY and session != 5').reset_index(drop=True)
df_ay = df.query('rat in @RAT_AY and session != 5 and session <= @LAST_SESSION').reset_index(drop=True)


#%%
# ----- Mean Response Duration ---------

mf.mean_response_duration(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Distribuicao da duracao media por rato, sessoes-chave -----

mf.boxplot_groups_key_sessions(df_ay, key_sessions=[2, 12, 19], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Distribuicao da duracao media por rato, primeira sessao (Early vs Late) -----

mf.boxplot_first_session_early_late(df_ay, percentage_trials=33, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista (R): duracao media ~ grupo * (primeiro vs ultimo terco da 1a sessao) -----

mf.run_mixed_anova_first_session_early_late(df_ay, percentage_trials=33, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Duracao por trial, ultima sessao (30 primeiras e 30 ultimas tentativas), por rato -----

mf.duration_by_trial_session(df_ay, n_trials=30, session=19, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Pressupostos da ANOVA (R): normalidade (Shapiro-Wilk) e homogeneidade (Levene) -----

mf.run_anova_assumptions(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista (R): duracao media ~ grupo * sessao -----

mf.run_mixed_anova(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista nao-parametrica (nparLD): grupo x sessao, com post-hoc entre grupos -----

mf.run_nparld_anova(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Learned vs Did Not Learn classification + Fisher's exact test (R): grupo x aprendeu -----
# Usa df_ay_un (todas as sessoes do rato, sem o corte em LAST_SESSION) para nao
# descartar sessoes pos-criterio que ocorram depois da sessao 19. resolve_original_group
# remove as sessoes de ratos (54, 57, 62, 73) apos serem reatribuidos a outro grupo.

df_ay_learned_input = mf.resolve_original_group(df_ay_un)
ay_learned_matrix, ay_learned_fisher = mf.learned_fisher_test(df_ay_learned_input, crit=10, min_reinforced_rate=0.05, \
                                                           PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ----- Number of sessions to reach criteria -----

#mf.sessions_reach_criteria(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Number of Trials to reach criteria ------

mf.trials_reach_criteria(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA de uma via (R): tentativas cumulativas ate atingir 10s, entre grupos -----

mf.run_oneway_anova_trials_to_criterion(df_ay, crit=10, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Percentage of Rats that reached criteria --------


#mf.percentage_rats_criteria(df_ay, crit = 5, criterion = '5s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)


mf.percentage_rats_criteria(df_ay, crit = 10, criterion = '10s', PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#mf.proportion_correct_responses(df_ay, PREFIX = PREFIX, OUTPUT_PATH=OUTPUT_PATH)

mf.proportion_correct_responses_by_criterion(df_ay, crit = 10, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ------ Criterion progression per session ------

#mf.criterion_progression(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ------ Mixture cure model: reach criterion 10s while keeping reinforced rate > 5% ------

# ay_cure_fit = cm.fit_criterion_cure_model(df_ay_un, crit = 10, min_reinforced_rate = 0.05, verbose = True)
# print(ay_cure_fit['b'])
# print(ay_cure_fit['beta'])
# print(cm.cure_probability_by_group(ay_cure_fit))

# cm.plot_population_survival(ay_cure_fit, times = list(sessions), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)
# cm.plot_cure_probability_bar(ay_cure_fit, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

# ------ Kaplan-Meier curve (same event definition as the cure model) ------

# cm.plot_kaplan_meier(df_ay_un, crit = 10, min_reinforced_rate = 0.05, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, \
                    #   cure_fit_result = ay_cure_fit, times = list(sessions))

#%%

# ------ Boxplot of the last three sessions for each group --------

#mf.boxplot_groups(df_ay, PREFIX = PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ------ Number of rats that reached the maximum criterion --------

n_rats = df_ay.query('criterion ==10').groupby(['group']).rat.nunique().reset_index(name= 'n_rats')

#%%

# ------- Fit Doble Gaussian to Log of Duration Distribution --------

# ------- MLE and LSF fitting, per rat and per individual session ----------

ay_parameters = mf.fit_params_all_sessions(df_ay_un, mf.MLE, log=True)
ay_parameters.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Parameters.csv', index=False)

#ay_parameters_lsf = mf.fit_params_all_sessions(df_ay_un, mf.LSF, log=True)
#ay_parameters_lsf.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Parameters_LSF.csv', index=False)

#%%
# -------- Plot Parameters for each group -----------

mf.plot_parameters_boxplot(ay_parameters, order = list(sessions), \
                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_parameters_lineplot(ay_parameters, order = list(sessions), \
 #                          PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#for param in ['mu1', 'mu2', 'sigma1', 'sigma2', 'gamma']:
 #   mf.plot_param_method_comparison_boxplot(ay_parameters, ay_parameters_lsf, param, order = list(sessions), \
  #                         PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)

#mf.plot_gamma_vs_gamma2_boxplot(ay_parameters, order = list(sessions), \
 #                          PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH)


#%%
ay_parameters_successful = ay_parameters.query('rat in @df_successful').reset_index(drop=True)

#%%
# ------ Criterion-tracking index: i = (C - mu2) / sigma2, per rat/session ------

ay_criterion_index = mf.calculate_criterion_tracking_index(ay_parameters.query('session <= 19'), df_ay_un, log = True)

mf.plot_criterion_tracking_index_lineplot(ay_criterion_index, order = list(sessions), \
                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')


ay_criterion_index = mf.calculate_criterion_tracking_index(ay_parameters, df_ay_un, log = True)

mf.plot_criterion_tracking_index_lineplot(ay_criterion_index, order = list(sessions), \
                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#%%
# early = [1, 2, 3]
late = [18]

for group in df_ay.group.unique():

#     df_group_early = df_ay.loc[(df_ay['group'] == group) & (df_ay['session'].isin(early))].reset_index(drop=True)
#     mf.plot_fitting_mle_rats(df_group_early, session = str(early), log = True, \
#                               PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

    df_group_late = df_ay.loc[(df_ay['group'] == group) & (df_ay['session'].isin(late))].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_late, session = str(late), log = True, \
                               PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# -------- Histogram Fit (MLE, dados sem log), por grupo, sessao 1 e 2, por rato --------


for group in df_ay.group.unique():

    df_group_1 = df_ay.loc[(df_ay['group'] == group) & (df_ay['session'] == 1)].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_1, session = '1', log = False, ceiling = 10, \
                              PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
    df_group_2 = df_ay.loc[(df_ay['group'] == group) & (df_ay['session'] == 2)].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_2, session = '2', log = False, ceiling = 10, \
                              PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
        

#%%
# -------- Histograma da duracao (sem ajuste), por grupo, sessao 19, por rato --------

mf.plot_duration_histogram_rats(df_ay, session=19, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# %%


# ============================================================
# ANALISE DE CURVA DE APRENDIZADO E GRAFICOS DE Ainf / TAU
# ------------------------------------------------------------
# Bloco comentado, deslocado para o fim do arquivo. Ajuste da curva
# logistica de aprendizado (mu2 por sessao -> A0, Ainf, tau) e as
# comparacoes derivadas dela (tau, Ainf, sessoes-ate-o-criterio),
# para os experimentos AU e AY/AW. Depende das variaveis calculadas
# nas secoes acima (au_parameters, df_au, ay_parameters, df_ay,
# sessions, etc.) -- reative descomentando o bloco desejado.
# ============================================================

# ---------------- AU ----------------
# PREFIX = 'AU DRRD'

#au_parameters_successful = au_parameters.query('rat in @df_successful').reset_index(drop=True)

#au_tau = mf.plot_mu2_learning_curve_boxplot(au_parameters, order = list(range(1,20)), \
#                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_tau_boxplot(au_tau, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_ainf_boxplot(au_tau, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ Sessions needed to reach mu2 = ln(10), from the free-Ainf learning curve fit ------

#_, au_sessions_to_criterion_excluded = mf.plot_sessions_to_criterion_boxplot(au_tau, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ Same, but with A_infinito fixed at ln(11) (just above ln(10), so the curve still crosses it) ------

#_, au_tau_fixed_ainf11 = mf.fit_learning_curve_per_rat(au_parameters, ainf = mf.CRITERION_11S_LOG_ASYMPTOTE)

#_, au_sessions_to_criterion_excluded_fixed11 = mf.plot_sessions_to_criterion_boxplot(au_tau_fixed_ainf11, \
 #                          label = 'FixedAinf_ln11', PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ mu2 learning curve, split by whether the rat reached the max criterion ------

#au_tau_by_criterion = mf.plot_mu2_learning_curve_boxplot_by_criterion(au_parameters, df_au, crit = 10, \
  #                         order = list(range(1,20)), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_tau_boxplot_by_criterion(au_tau_by_criterion, crit = 10, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_ainf_boxplot_by_criterion(au_tau_by_criterion, crit = 10, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ mu2 learning curve, A_infinito fixed at the 10s-criterion log asymptote vs. freely fit ------

#_, au_tau_fixed_ainf = mf.plot_mu2_learning_curve_boxplot_free_vs_fixed(au_parameters, order = list(range(1,20)), \
#                           PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_tau_free_vs_fixed_ainf_boxplot(au_tau, au_tau_fixed_ainf, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#au_tau_successful = au_tau_by_criterion.query('criterion_subgroup == "Reached 10s"').reset_index(drop=True)

#mf.plot_tau_all_vs_successful_boxplot(au_tau, au_tau_successful, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')


# ---------------- AY e AW ----------------
# PREFIX = 'AY and AW DRRD'

#ay_tau = mf.plot_mu2_learning_curve_boxplot(ay_parameters, order = list(sessions), \
                          # PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_tau_boxplot(ay_tau, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_ainf_boxplot(ay_tau, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ Sessions needed to reach mu2 = ln(10), from the free-Ainf learning curve fit ------

#ay_sessions_to_criterion_excluded = mf.plot_sessions_to_criterion_boxplot(ay_tau, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ Same, but with A_infinito fixed at ln(11) (just above ln(10), so the curve still crosses it) ------

#ay_tau_fixed_ainf11 = mf.fit_learning_curve_per_rat(ay_parameters, ainf = mf.CRITERION_11S_LOG_ASYMPTOTE)

#ay_sessions_to_criterion_excluded_fixed11 = mf.plot_sessions_to_criterion_boxplot(ay_tau_fixed_ainf11, \
                          # label = 'FixedAinf_ln11', PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ mu2 learning curve, split by whether the rat reached the max criterion ------

#ay_tau_by_criterion = mf.plot_mu2_learning_curve_boxplot_by_criterion(ay_parameters, df_ay, crit = 10, \
                           #order = list(sessions), PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_tau_boxplot_by_criterion(ay_tau_by_criterion, crit = 10, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_ainf_boxplot_by_criterion(ay_tau_by_criterion, crit = 10, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# ------ mu2 learning curve, A_infinito fixed at the 10s-criterion log asymptote vs. freely fit ------

#ay_tau_fixed_ainf = mf.plot_mu2_learning_curve_boxplot_free_vs_fixed(ay_parameters, order = list(sessions), \
                           #PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#mf.plot_tau_free_vs_fixed_ainf_boxplot(ay_tau, ay_tau_fixed_ainf, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

#ay_tau_successful = ay_tau_by_criterion.query('criterion_subgroup == "Reached 10s"').reset_index(drop=True)

#mf.plot_tau_all_vs_successful_boxplot(ay_tau, ay_tau_successful, PREFIX = PREFIX, OUTPUT_PATH = OUTPUT_PATH, fit_type = 'MLE')

# %%
