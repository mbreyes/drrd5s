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

# ---- R (rpy2): usado para ANOVA mista (parametrica e nparLD, nao-parametrica) ----
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

rstatix = importr('rstatix')
nparld = importr('nparLD')

PREFIX = 'DRRD_10s_Groups'


def _session_as_ordered_factor(df, col='session'):
    """
    Converte `col` (tipicamente 'session') em categorico ordenado numericamente
    antes de enviar para o R. Sem isso, a conversao direta para string faz o R
    ordenar os niveis do fator alfabeticamente (ex.: '1','10','11',...,'2','3'),
    o que bagunca a ordem de sessoes de dois digitos nas tabelas de post-hoc.
    """
    df = df.copy()
    categories = [str(v) for v in sorted(df[col].unique())]
    df[col] = pd.Categorical(df[col].astype(str), categories=categories, ordered=True)
    return df

plt.style.use('ggplot')
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
#---- ANOVA ASSUMPTIONS: NORMALITY (Shapiro-Wilk) + HOMOGENEITY (Levene) (R via rpy2) ----

def run_anova_assumptions(data, PREFIX, OUTPUT_PATH):
    """
    Checagem de pressupostos da ANOVA mista (grupo x sessao) sobre a
    duracao media por rato/sessao, via rstatix (R, rpy2):
      - Shapiro-Wilk: normalidade da duracao media em cada celula grupo x sessao.
      - Levene: homogeneidade de variancia entre grupos, em cada sessao.

    (O teste de Mauchly, para esfericidade do fator sessao, e reportado por
    run_mixed_anova — ja e calculado internamente pelo rstatix::anova_test.)

    Salva em OUTPUT_PATH:
      - {PREFIX}_Shapiro_Normality_MeanDuration.csv
      - {PREFIX}_Levene_Homogeneity_MeanDuration.csv
    """
    df_mean = data.groupby(['rat', 'session', 'group']).duration.mean().reset_index(name='mean_duration')

    df_mean_r = df_mean.astype({'rat': str, 'group': str})
    df_mean_r = _session_as_ordered_factor(df_mean_r)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_mean_r)
    ro.globalenv['df_assump_r'] = r_df

    ro.r('df_assump_r$group <- as.factor(df_assump_r$group)')
    ro.r('shapiro_tab <- as.data.frame(df_assump_r %>% dplyr::group_by(group, session) %>% rstatix::shapiro_test(mean_duration))')
    ro.r('levene_tab <- as.data.frame(df_assump_r %>% dplyr::group_by(session) %>% rstatix::levene_test(mean_duration ~ group))')

    with localconverter(ro.default_converter + pandas2ri.converter):
        shapiro_df = ro.conversion.rpy2py(ro.r('shapiro_tab')).reset_index(drop=True)
        levene_df = ro.conversion.rpy2py(ro.r('levene_tab')).reset_index(drop=True)

    shapiro_df.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Shapiro_Normality_MeanDuration.csv', index=False)
    levene_df.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Levene_Homogeneity_MeanDuration.csv', index=False)

    n_shapiro_violations = (shapiro_df['p'] < 0.05).sum()
    n_levene_violations = (levene_df['p'] < 0.05).sum()

    print(f'{PREFIX}: Shapiro-Wilk normality (group x session cells) on mean duration '
          f'-- {n_shapiro_violations}/{len(shapiro_df)} cells with p < .05')
    print(shapiro_df)
    print(f'{PREFIX}: Levene homogeneity of variance (per session) on mean duration '
          f'-- {n_levene_violations}/{len(levene_df)} sessions with p < .05')
    print(levene_df)

    return shapiro_df, levene_df

#%%
#---- MIXED ANOVA (R via rpy2) ----

def run_mixed_anova(data, PREFIX, OUTPUT_PATH):
    """
    ANOVA mista de dois fatores: grupo (entre-sujeitos) x sessao
    (intra-sujeito, medida repetida em cada rato), sobre a duracao media por
    rato/sessao, via rstatix::anova_test (R, rpy2). Reporta tambem o teste
    de Mauchly para esfericidade do fator sessao (ja calculado internamente
    pelo anova_test, usado para decidir a correcao de Greenhouse-Geisser).

    Salva em OUTPUT_PATH:
      - {PREFIX}_Mean_Duration_by_Session_Group.csv (duracao media por rato/sessao/grupo, entrada da ANOVA)
      - {PREFIX}_Mixed_ANOVA_MeanDuration.csv (tabela da ANOVA: efeitos de grupo, sessao e interacao)
      - {PREFIX}_Mauchly_Sphericity_MeanDuration.csv (teste de Mauchly para o fator sessao)
    """
    df_mean = data.groupby(['rat', 'session', 'group']).duration.mean().reset_index(name='mean_duration')
    df_mean.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Mean_Duration_by_Session_Group.csv', index=False)

    df_mean_r = df_mean.astype({'rat': str, 'group': str})
    df_mean_r = _session_as_ordered_factor(df_mean_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_mean_r)

    anova_result = rstatix.anova_test(data=r_df, dv='mean_duration', wid='rat', within='session', between='group')
    anova_table = rstatix.get_anova_table(anova_result)

    ro.globalenv['anova_result_r'] = anova_result
    ro.r('mauchly_tab <- as.data.frame(anova_result_r[["Mauchly\'s Test for Sphericity"]])')

    with localconverter(ro.default_converter + pandas2ri.converter):
        anova_df = ro.conversion.rpy2py(anova_table)
        mauchly_df = ro.conversion.rpy2py(ro.r('mauchly_tab')).reset_index(drop=True)

    anova_df.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Mixed_ANOVA_MeanDuration.csv', index=False)
    mauchly_df.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Mauchly_Sphericity_MeanDuration.csv', index=False)

    print(f'{PREFIX}: Mixed ANOVA (group x session) on mean duration')
    print(anova_df)
    print(f'{PREFIX}: Mauchly Test for Sphericity (session factor) on mean duration')
    print(mauchly_df)
    return anova_df

#%%
#---- NPARLD: NONPARAMETRIC LONGITUDINAL/MIXED ANOVA (R via rpy2) ----

def run_nparld_anova(data, PREFIX, OUTPUT_PATH):
    """
    Alternativa nao-parametrica a ANOVA mista: analise nao-parametrica de
    dados longitudinais (nparLD::nparLD) sobre a duracao media por
    rato/sessao, com grupo como fator entre-sujeitos (whole-plot) e sessao
    como fator intra-sujeito (sub-plot). Reporta o Wald-Type Statistic
    (WTS) e o ANOVA-Type Statistic (ATS, mais robusto em amostras
    pequenas/moderadas) para grupo, sessao e interacao, alem de um post-hoc
    de comparacoes multiplas (Tukey) entre os grupos — o que o teste de
    Friedman nao permite (ele exige o mesmo sujeito em cada nivel, e por
    isso so compararia sessoes dentro de cada grupo).

    Salva em OUTPUT_PATH:
      - {PREFIX}_nparLD_ANOVA_MeanDuration.csv (WTS e ATS: grupo, sessao, interacao)
      - {PREFIX}_nparLD_PostHoc_Group_MeanDuration.csv (comparacoes par a par entre grupos, Tukey)
    """
    df_mean = data.groupby(['rat', 'session', 'group']).duration.mean().reset_index(name='mean_duration')

    df_mean_r = df_mean.astype({'rat': str, 'group': str})
    df_mean_r = _session_as_ordered_factor(df_mean_r)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_mean_r)
    ro.globalenv['df_nparld_r'] = r_df

    ro.r('df_nparld_r$rat <- as.factor(df_nparld_r$rat)')
    ro.r('df_nparld_r$group <- as.factor(df_nparld_r$group)')
    group_levels = list(ro.r('levels(df_nparld_r$group)'))

    ro.r('''
    fit_nparld <- nparLD::nparLD(mean_duration ~ group * session, data = df_nparld_r,
                                  subject = "rat", contrast = list("group", "Tukey"),
                                  Factor.Information = FALSE)
    wts_df <- as.data.frame(fit_nparld$WTS)
    wts_df$statistic_type <- "WTS"
    wts_df$term <- rownames(wts_df)
    ats_df <- as.data.frame(fit_nparld$ATS)
    ats_df$statistic_type <- "ATS"
    ats_df$term <- rownames(ats_df)
    anova_tab <- rbind(wts_df[, c("term","statistic_type","Statistic","df","p-value")],
                        ats_df[, c("term","statistic_type","Statistic","df","p-value")])
    mctp_df <- fit_nparld$MCTP$Local.Results
    mctp_df$contrast <- rownames(mctp_df)
    ''')

    with localconverter(ro.default_converter + pandas2ri.converter):
        anova_df = ro.conversion.rpy2py(ro.r('anova_tab')).reset_index(drop=True)
        posthoc_df = ro.conversion.rpy2py(ro.r('mctp_df')).reset_index(drop=True)

    # decodifica os indices numericos do contraste ("2 - 1") para os nomes reais dos grupos
    level_map = {str(i + 1): lvl for i, lvl in enumerate(group_levels)}
    def _decode_contrast(c):
        a, b = [t.strip() for t in c.split('-')]
        return f'{level_map.get(a, a)} - {level_map.get(b, b)}'
    posthoc_df['contrast'] = posthoc_df['contrast'].apply(_decode_contrast)

    anova_df.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_nparLD_ANOVA_MeanDuration.csv', index=False)
    posthoc_df.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_nparLD_PostHoc_Group_MeanDuration.csv', index=False)

    print(f'{PREFIX}: nparLD (nonparametric longitudinal ANOVA: group x session) on mean duration')
    print(anova_df)
    print(f'{PREFIX}: nparLD post-hoc pairwise group comparisons (Tukey) on mean duration')
    print(posthoc_df)

    return anova_df, posthoc_df

#%%
#---- LEARNED CLASSIFICATION + FISHER'S EXACT TEST (group x learned), R via rpy2 ----

def resolve_original_group(data):
    """
    Alguns ratos (54, 57, 62 e 73, no experimento AY12) sao reatribuidos a
    um outro grupo (LINEAR) apos a sessao ~21-22, dentro do mesmo arquivo de
    dados -- provavelmente uma extensao/re-treino do rato apos o fim do seu
    protocolo original. Para manter uma unica atribuicao de grupo por rato,
    mantemos apenas as sessoes do grupo original (o primeiro cronologicamente)
    de cada rato, descartando as sessoes posteriores a troca de grupo.
    """
    original_group = data.sort_values('session').groupby('rat').group.first().rename('original_group')
    data = data.merge(original_group, on='rat')
    data = data.query('group == original_group').drop(columns='original_group').reset_index(drop=True)
    return data


def classify_learned(data, crit=10, min_reinforced_rate=0.05):
    """
    Classifica cada rato como 'aprendeu' ou 'nao aprendeu', em duas etapas:

      1. 'reached': ratos que atingiram o criterio maximo (`criterion == crit`
         em pelo menos um trial), obtidos via `data.query('criterion == @crit').rat.unique()`.
         A primeira sessao em que isso ocorre (`first_session_reached`) marca
         o inicio da fase "pos-criterio" de cada rato.
      2. Por rato/sessao, taxa de recompensa (media de `reinforced`) rotulada
         como recompensada (> min_reinforced_rate) ou nao recompensada,
         considerando apenas as sessoes a partir de `first_session_reached`
         (inclusive) -- i.e., as sessoes apos o rato atingir o criterio de 10s.

    'aprendeu' = ratos 'reached' cujas sessoes pos-criterio foram todas
    recompensadas (nenhuma sessao, a partir de `first_session_reached`, com
    taxa de recompensa <= min_reinforced_rate). Os demais (nao atingiram o
    criterio, ou atingiram mas tiveram ao menos uma sessao pos-criterio nao
    recompensada) sao classificados como 'nao aprendeu'.

    Retorna um DataFrame: rat, group, reached, all_sessions_rewarded, learned.
    """
    first_session_reached = data.query('criterion == @crit').groupby('rat').session.min() \
        .reset_index(name='first_session_reached')
    reached_rats = first_session_reached['rat'].unique()

    session_rate = data.groupby(['rat', 'session']).reinforced.mean().reset_index(name='reinforced_rate')
    session_rate['rewarded_session'] = session_rate['reinforced_rate'] > min_reinforced_rate

    session_rate_post = session_rate.merge(first_session_reached, on='rat', how='inner')
    session_rate_post = session_rate_post.query('session >= first_session_reached')
    all_sessions_rewarded = session_rate_post.groupby('rat')['rewarded_session'].all() \
        .reset_index(name='all_sessions_rewarded')

    classification = data[['rat', 'group']].drop_duplicates().reset_index(drop=True)
    classification['reached'] = classification['rat'].isin(reached_rats)
    classification = classification.merge(all_sessions_rewarded, on='rat', how='left')
    classification['all_sessions_rewarded'] = classification['all_sessions_rewarded'].fillna(False)
    classification['learned'] = classification['reached'] & classification['all_sessions_rewarded']

    return classification


def learned_fisher_test(data, crit=10, min_reinforced_rate=0.05, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH):
    """
    Monta a matriz grupo x (aprendeu / nao aprendeu), a partir de
    classify_learned, e testa a associacao entre grupo e aprendizado com o
    teste exato de Fisher (stats::fisher.test, R via rpy2), calculando o
    p-valor por dois metodos separados:
      1. Exato -- algoritmo de rede de Mehta-Patel (extensao de Freeman-Halton
         para tabelas r x c quando ha mais de dois grupos).
      2. Simulacao de Monte Carlo (simulate.p.value = TRUE, B = 100000),
         usada como alternativa/comparacao ao exato.

    Salva em OUTPUT_PATH:
      - {PREFIX}_Learned_Classification.csv (classificacao individual por rato)
      - {PREFIX}_Learned_Matrix_Group.csv (matriz grupo x aprendeu/nao aprendeu)
      - {PREFIX}_Learned_FisherExact.csv (tabela com p-valor de cada metodo,
        e odds ratio quando a matriz e 2x2)
    """
    classification = classify_learned(data, crit=crit, min_reinforced_rate=min_reinforced_rate)
    classification.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Learned_Classification.csv', index=False)

    matrix = pd.crosstab(classification['group'], classification['learned'])
    matrix = matrix.rename(columns={True: 'learned', False: 'did_not_learn'})
    for col in ['learned', 'did_not_learn']:
        if col not in matrix.columns:
            matrix[col] = 0
    matrix = matrix[['learned', 'did_not_learn']]
    matrix.index.name = 'group'
    matrix.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Learned_Matrix_Group.csv')

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_matrix = ro.conversion.py2rpy(matrix)
    ro.globalenv['learned_matrix_r'] = r_matrix

    ro.r('''
        set.seed(42)
        fisher_exact_res <- tryCatch(
            fisher.test(as.matrix(learned_matrix_r), workspace = 2e7),
            error = function(e) NULL
        )
        fisher_mc_res <- fisher.test(as.matrix(learned_matrix_r), simulate.p.value = TRUE, B = 100000)
    ''')
    exact_failed = bool(ro.r('is.null(fisher_exact_res)')[0])
    fisher_exact_p = np.nan if exact_failed else ro.r('fisher_exact_res$p.value')[0]
    fisher_mc_p = ro.r('fisher_mc_res$p.value')[0]

    is_2x2 = matrix.shape == (2, 2)
    if is_2x2:
        odds_source = 'fisher_mc_res' if exact_failed else 'fisher_exact_res'
        fisher_odds_ratio = ro.r(f'{odds_source}$estimate')[0]
    else:
        fisher_odds_ratio = np.nan

    fisher_summary = pd.DataFrame({
        'method': ['exact (network / Freeman-Halton)', 'Monte Carlo simulation (B=100000)'],
        'p_value': [fisher_exact_p, fisher_mc_p],
        'odds_ratio': [fisher_odds_ratio, fisher_odds_ratio],
    })
    fisher_summary.to_csv(os.path.realpath(f'{OUTPUT_PATH}')+'/'+f'{PREFIX}_Learned_FisherExact.csv', index=False)

    print(f'{PREFIX}: Learned vs Did Not Learn, by group')
    print(matrix)
    print(fisher_summary)

    return matrix, fisher_summary

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

# ----- Duracao por trial, ultima sessao (30 primeiras e 30 ultimas tentativas), por rato -----

mf.duration_by_trial_last_session(df_au, n_trials=30, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Pressupostos da ANOVA (R): normalidade (Shapiro-Wilk) e homogeneidade (Levene) -----

run_anova_assumptions(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista (R): duracao media ~ grupo * sessao -----

run_mixed_anova(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista nao-parametrica (nparLD): grupo x sessao, com post-hoc entre grupos -----

run_nparld_anova(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Learned vs Did Not Learn classification + Fisher's exact test (R): grupo x aprendeu -----
# Usa df_au_un (todas as sessoes do rato, sem o corte em LAST_SESSION) para nao
# descartar sessoes pos-criterio que ocorram depois da sessao 19. resolve_original_group
# nao tem efeito aqui se nenhum rato do experimento AU trocar de grupo, mas e aplicado
# por consistencia com o experimento AY/AW.

df_au_learned_input = resolve_original_group(df_au_un)
au_learned_matrix, au_learned_fisher = learned_fisher_test(df_au_learned_input, crit=10, min_reinforced_rate=0.05, \
                                                           PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ----- Number of sessions to reach criteria -----

mf.sessions_reach_criteria(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Number of Trials to reach criteria ------

mf.trials_reach_criteria(df_au, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

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

# ----- Duracao por trial, ultima sessao (30 primeiras e 30 ultimas tentativas), por rato -----

mf.duration_by_trial_last_session(df_ay, n_trials=30, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Pressupostos da ANOVA (R): normalidade (Shapiro-Wilk) e homogeneidade (Levene) -----

run_anova_assumptions(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista (R): duracao media ~ grupo * sessao -----

run_mixed_anova(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- ANOVA mista nao-parametrica (nparLD): grupo x sessao, com post-hoc entre grupos -----

run_nparld_anova(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Learned vs Did Not Learn classification + Fisher's exact test (R): grupo x aprendeu -----
# Usa df_ay_un (todas as sessoes do rato, sem o corte em LAST_SESSION) para nao
# descartar sessoes pos-criterio que ocorram depois da sessao 19. resolve_original_group
# remove as sessoes de ratos (54, 57, 62, 73) apos serem reatribuidos a outro grupo.

df_ay_learned_input = resolve_original_group(df_ay_un)
ay_learned_matrix, ay_learned_fisher = learned_fisher_test(df_ay_learned_input, crit=10, min_reinforced_rate=0.05, \
                                                           PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

#%%
# ----- Number of sessions to reach criteria -----

#mf.sessions_reach_criteria(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

# ----- Number of Trials to reach criteria ------

#mf.trials_reach_criteria(df_ay, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

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
early = [1, 2, 3]
late = [17, 18, 19]

for group in df_ay.group.unique():

    df_group_early = df_ay.loc[(df_ay['group'] == group) & (df_ay['session'].isin(early))].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_early, session = str(early), log = True, \
                              PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

    df_group_late = df_ay.loc[(df_ay['group'] == group) & (df_ay['session'].isin(late))].reset_index(drop=True)
    mf.plot_fitting_mle_rats(df_group_late, session = str(late), log = True, \
                              PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

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
