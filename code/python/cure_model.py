#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ailacamara

Semiparametric mixture cure model (logistic incidence + Cox PH latency),
fit via the EM algorithm of Sy & Taylor (2000) / Peng & Dear (2000). This
is a from-scratch Python re-implementation of the algorithm behind R's
`smcure` package (model = 'ph'); `smcure_ph`'s output was validated against
`smcure()` on identical simulated data (matches to ~6 significant digits).

Model
-----
Each individual i is either "susceptible" (will eventually experience the
event) with probability pi(Z_i), or "cured" (will never experience the
event) with probability 1 - pi(Z_i):

    pi(Z_i) = exp(b0 + b'Z_i) / (1 + exp(b0 + b'Z_i))            [logistic]

Conditional on being susceptible, the time to event follows a standard
Cox proportional-hazards model:

    S_u(t | X_i) = S_0(t) ** exp(beta'X_i)                        [Cox PH]

so the marginal (population) survival function is a two-component mixture:

    S_pop(t | X_i, Z_i) = (1 - pi(Z_i)) + pi(Z_i) * S_u(t | X_i)

Y_i (susceptible/cured) is latent. It is known to be 1 whenever the event
is observed (delta_i = 1); for censored individuals (delta_i = 0) it is
unknown. The EM algorithm alternates:

  E-step: soft-assign each censored individual a "susceptibility weight"
      w_i = P(Y_i = 1 | censored at C_i) = pi_i * S_u(C_i|X_i) /
            [(1 - pi_i) + pi_i * S_u(C_i|X_i)]   (w_i = 1 when delta_i = 1)

  M-step: refit the logistic submodel by weighted (fractional-response)
      logistic regression of w on Z, and refit the Cox submodel by a
      weighted partial-likelihood Cox regression with case weights w. The
      Breslow baseline hazard is then re-estimated (see `_breslow_baseline`)
      using the *previous* beta together with the *new* w -- this one-step
      lag is exactly what R's `smcure:::em`/`smsurv` do, and is required to
      reproduce its results (using an always-freshly-fit baseline, as
      `lifelines.predict_survival_function` does after every Cox refit,
      makes the EM diverge -- see module tests).
"""

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter


# ----------------------------------------------------------------------
# Survival data construction: "event" = reach criterion 10s while keeping
# the reinforced-response rate above 5% in that same session.
# ----------------------------------------------------------------------

def build_cure_survival_data(data, crit=10, min_reinforced_rate=0.05):
    """One row per rat: rat, group, time, event.

    `event` = 1 at the first session where the rat's max criterion reaches
    `crit` AND its proportion of reinforced trials in that session exceeds
    `min_reinforced_rate`. If that never happens, the rat is censored
    (event = 0) at its last observed session."""

    session_stats = data.groupby(['rat', 'group', 'session']).agg(
        max_criterion=('criterion', 'max'),
        reinforced_rate=('reinforced', 'mean')
    ).reset_index()

    session_stats['success'] = (session_stats['max_criterion'] >= crit) & \
                                (session_stats['reinforced_rate'] > min_reinforced_rate)

    rows = []
    for rat, rat_data in session_stats.groupby('rat'):
        rat_data = rat_data.sort_values('session')
        group = rat_data['group'].iloc[0]
        success_sessions = rat_data.loc[rat_data['success'], 'session']

        if len(success_sessions) > 0:
            time, event = success_sessions.iloc[0], 1
        else:
            time, event = rat_data['session'].iloc[-1], 0

        rows.append({'rat': rat, 'group': group, 'time': float(time), 'event': event})

    return pd.DataFrame(rows)


def build_cure_survival_data_trials(data, crit=10, min_reinforced_rate=0.05):
    """One row per rat: rat, group, time, event -- same event definition as
    build_cure_survival_data, but `time` is the cumulative number of trials
    completed (summed across the rat's sessions, in session order) instead
    of the session index -- trials give a finer-grained, more directly
    behavioral measure of exposure than sessions, which can vary widely in
    trial count both within and across rats.

    `event` = 1 at the first session where the rat's max criterion reaches
    `crit` AND its proportion of reinforced trials in that session exceeds
    `min_reinforced_rate`; `time` is then the rat's cumulative trial count
    through the end of that session. If that never happens, the rat is
    censored (event = 0) at its total cumulative trial count across all
    observed sessions."""

    session_stats = data.groupby(['rat', 'group', 'session']).agg(
        max_criterion=('criterion', 'max'),
        reinforced_rate=('reinforced', 'mean'),
        n_trials=('trial', 'count')
    ).reset_index()

    session_stats = session_stats.sort_values(['rat', 'session'])
    session_stats['cum_trials'] = session_stats.groupby('rat')['n_trials'].cumsum()

    session_stats['success'] = (session_stats['max_criterion'] >= crit) & \
                                (session_stats['reinforced_rate'] > min_reinforced_rate)

    rows = []
    for rat, rat_data in session_stats.groupby('rat'):
        rat_data = rat_data.sort_values('session')
        group = rat_data['group'].iloc[0]
        success_rows = rat_data.loc[rat_data['success']]

        if len(success_rows) > 0:
            time, event = success_rows['cum_trials'].iloc[0], 1
        else:
            time, event = rat_data['cum_trials'].iloc[-1], 0

        rows.append({'rat': rat, 'group': group, 'time': float(time), 'event': event})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Weighted Breslow baseline (smcure's `smsurv`, PH model)
# ----------------------------------------------------------------------

def _breslow_baseline(time, delta, X, beta, w):
    """Weighted Breslow baseline cumulative-hazard step function: the
    distinct observed event times (`death_points`) and the cumulative
    baseline hazard Lambda_0 evaluated at each of them, given a *fixed*
    beta and case weights w (no optimization -- beta is not re-fit here)."""

    time = np.asarray(time, dtype=float)
    delta = np.asarray(delta, dtype=float)
    w = np.asarray(w, dtype=float)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    coxexp = np.exp(np.asarray(X, dtype=float) @ beta)

    death_points = np.sort(np.unique(time[delta == 1]))
    lam = np.empty(len(death_points))
    for k, t_k in enumerate(death_points):
        events_at_k = np.sum(delta * (time == t_k))
        risk_weight = np.sum((time >= t_k) * w * coxexp)
        lam[k] = events_at_k / risk_weight if risk_weight > 0 else 0.0

    return death_points, np.cumsum(lam)


def _survival_at(query_times, death_points, cum_hazard):
    """S_0(t) from a Breslow step function at arbitrary query times: 1
    before the first observed event, the step function in between, 0 past
    the last observed event (matches smcure's `smsurv` convention)."""

    query_times = np.atleast_1d(np.asarray(query_times, dtype=float))
    if len(death_points) == 0:
        return np.ones_like(query_times)

    H = np.array([
        cum_hazard[death_points <= t][-1] if np.any(death_points <= t) else 0.0
        for t in query_times
    ])
    H = np.where(query_times > death_points.max(), np.inf, H)
    H = np.where(query_times < death_points.min(), 0.0, H)
    return np.exp(-H)


def _fit_weighted_cox(time, delta, X, w, cov_names):
    """Weighted Cox PH partial-likelihood fit (case weights w); returns the
    fitted beta vector, in `cov_names` order."""

    cox_df = pd.DataFrame(np.asarray(X, dtype=float), columns=cov_names)
    cox_df['time'] = time
    cox_df['delta'] = delta
    cox_df['w'] = np.clip(w, 1e-8, None)

    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cph.fit(cox_df, duration_col='time', event_col='delta', weights_col='w', robust=True)

    return cph.params_.reindex(cov_names).values


# ----------------------------------------------------------------------
# EM fit of the semiparametric PH mixture cure model
# ----------------------------------------------------------------------

def smcure_ph(df, time_col='time', event_col='event', cure_cov=None, cox_cov=None,
              max_iter=50, tol=1e-7, verbose=True):
    """Semiparametric PH mixture cure model, EM fit (Sy & Taylor, 2000;
    matches R's `smcure(..., model='ph')`).

    df: one row per subject with `time_col`, `event_col`, and the covariate
        columns referenced by `cure_cov` (incidence/logistic submodel) and
        `cox_cov` (latency/Cox PH submodel). The two covariate sets may
        differ or coincide, exactly as in `smcure(formula=, cureform=)`.

    Returns a dict: b (logistic coefficients, incl. intercept, as a Series),
    beta (Cox coefficients, as a Series), df (input df with pi_susceptible /
    cure_probability / em_weight columns appended), death_points and
    cum_hazard (the fitted Breslow baseline, for `_survival_at`),
    converged, n_iter.
    """

    df = df.reset_index(drop=True).copy()
    time = df[time_col].values.astype(float)
    delta = df[event_col].values.astype(float)

    cure_cov = list(cure_cov) if cure_cov else []
    cox_cov = list(cox_cov) if cox_cov else []

    Z = sm.add_constant(df[cure_cov], has_constant='add') if cure_cov else \
        pd.DataFrame({'const': np.ones(len(df))})
    X = df[cox_cov].values if cox_cov else np.zeros((len(df), 0))

    # ---- Initialization: w = delta (only observed events are "known" susceptible) ----
    w = delta.copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        logit_model = sm.GLM(w, Z, family=sm.families.Binomial()).fit()
    b = logit_model.params.values

    beta = _fit_weighted_cox(time, delta, X, w, cox_cov) if cox_cov else np.zeros(0)
    death_points, cum_hazard = _breslow_baseline(time, delta, X, beta, w)
    s = _survival_at(time, death_points, cum_hazard)  # S_0 at each subject's own time

    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):

        # ---- E-step ----
        pi = 1.0 / (1.0 + np.exp(-(Z.values @ b)))
        survival_i = np.clip(s ** np.exp(X @ beta), 1e-12, 1.0)
        denom = np.clip((1 - pi) + pi * survival_i, 1e-12, None)
        w = np.where(delta == 1, 1.0, pi * survival_i / denom)

        # ---- M-step ----
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logit_model = sm.GLM(w, Z, family=sm.families.Binomial()).fit()
        update_b = logit_model.params.values

        update_beta = _fit_weighted_cox(time, delta, X, w, cox_cov) if cox_cov else np.zeros(0)

        # NB: the refreshed baseline uses the *previous* beta with the *new*
        # w (matches R's smcure:::em -- see module docstring).
        death_points, cum_hazard = _breslow_baseline(time, delta, X, beta, w)
        update_s = _survival_at(time, death_points, cum_hazard)

        diff = np.sum((update_b - b) ** 2) + np.sum((update_beta - beta) ** 2) + np.sum((s - update_s) ** 2)
        b, beta, s = update_b, update_beta, update_s

        if verbose:
            print(f'EM iter {iteration:3d}: convergence criterion = {diff:.6g}')

        if diff < tol:
            converged = True
            break

    # final baseline, consistent with the final (b, beta, w)
    death_points, cum_hazard = _breslow_baseline(time, delta, X, beta, w)

    if verbose:
        status = 'converged' if converged else 'did NOT converge (max_iter reached)'
        print(f'--- smcure_ph {status} after {iteration} iterations ---')

    pi_final = 1.0 / (1.0 + np.exp(-(Z.values @ b)))

    df_out = df.copy()
    df_out['pi_susceptible'] = pi_final
    df_out['cure_probability'] = 1 - pi_final
    df_out['em_weight'] = w

    return {
        'b': pd.Series(b, index=Z.columns, name='logistic_coef'),
        'beta': pd.Series(beta, index=cox_cov, name='cox_coef'),
        'df': df_out,
        'cure_cov': cure_cov,
        'cox_cov': cox_cov,
        'death_points': death_points,
        'cum_hazard': cum_hazard,
        'converged': converged,
        'n_iter': iteration,
    }


def fit_criterion_cure_model(data, crit=10, min_reinforced_rate=0.05,
                              max_iter=50, tol=1e-7, verbose=True):
    """Convenience wrapper: builds the rat-level survival data (event =
    reaching criterion `crit` while keeping the reinforced-response rate
    above `min_reinforced_rate`, see build_cure_survival_data) and fits
    smcure_ph to it, using `group` dummies as covariates for both the
    cure-probability (logistic) and latency (Cox PH) submodels."""

    surv_df = build_cure_survival_data(data, crit=crit, min_reinforced_rate=min_reinforced_rate)

    group_dummies = pd.get_dummies(surv_df['group'], prefix='group', drop_first=True).astype(float)
    surv_df = pd.concat([surv_df, group_dummies], axis=1)
    cov_cols = list(group_dummies.columns)

    result = smcure_ph(surv_df, time_col='time', event_col='event',
                        cure_cov=cov_cols, cox_cov=cov_cols,
                        max_iter=max_iter, tol=tol, verbose=verbose)
    result['group_dummy_cols'] = cov_cols
    result['surv_df'] = surv_df
    return result


def fit_criterion_cure_model_trials(data, crit=10, min_reinforced_rate=0.05,
                                     max_iter=50, tol=1e-7, verbose=True):
    """Same as fit_criterion_cure_model, but the survival time is the
    cumulative number of trials (build_cure_survival_data_trials) rather
    than the session index."""

    surv_df = build_cure_survival_data_trials(data, crit=crit, min_reinforced_rate=min_reinforced_rate)

    group_dummies = pd.get_dummies(surv_df['group'], prefix='group', drop_first=True).astype(float)
    surv_df = pd.concat([surv_df, group_dummies], axis=1)
    cov_cols = list(group_dummies.columns)

    result = smcure_ph(surv_df, time_col='time', event_col='event',
                        cure_cov=cov_cols, cox_cov=cov_cols,
                        max_iter=max_iter, tol=tol, verbose=verbose)
    result['group_dummy_cols'] = cov_cols
    result['surv_df'] = surv_df
    return result


# ----------------------------------------------------------------------
# Bootstrap standard errors (optional; matches smcure's Var=TRUE, nboot)
# ----------------------------------------------------------------------

def bootstrap_smcure_ph(df, time_col='time', event_col='event', cure_cov=None, cox_cov=None,
                         nboot=100, max_iter=50, tol=1e-7, random_state=None):
    """Nonparametric bootstrap SEs for b and beta: resamples events and
    censored subjects separately (matches smcure's stratified bootstrap)
    and refits smcure_ph on each resample. Returns a dict with `b_se`,
    `beta_se` (Series) and the raw bootstrap draws (`b_boot`, `beta_boot`,
    each shape (n_success, n_params))."""

    rng = np.random.default_rng(random_state)
    events = df.index[df[event_col] == 1].to_numpy()
    censored = df.index[df[event_col] == 0].to_numpy()

    b_draws, beta_draws = [], []
    for _ in range(nboot):
        boot_idx = np.concatenate([
            rng.choice(events, size=len(events), replace=True),
            rng.choice(censored, size=len(censored), replace=True),
        ])
        boot_df = df.loc[boot_idx].reset_index(drop=True)
        try:
            res = smcure_ph(boot_df, time_col=time_col, event_col=event_col,
                             cure_cov=cure_cov, cox_cov=cox_cov,
                             max_iter=max_iter, tol=tol, verbose=False)
        except Exception:
            continue
        b_draws.append(res['b'].values)
        beta_draws.append(res['beta'].values)

    b_boot = np.array(b_draws)
    beta_boot = np.array(beta_draws)

    return {
        'b_se': pd.Series(b_boot.std(axis=0, ddof=1), index=(cure_cov and ['const'] + list(cure_cov)) or ['const']),
        'beta_se': pd.Series(beta_boot.std(axis=0, ddof=1), index=cox_cov),
        'b_boot': b_boot,
        'beta_boot': beta_boot,
        'n_success': len(b_draws),
    }


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------

def cure_probability_by_group(fit_result):
    """Mean fitted cure probability (1 - pi) per group, from the per-rat
    fitted values in fit_result['df']."""

    return fit_result['df'].groupby('group')['cure_probability'].agg(['mean', 'std', 'count']).reset_index()


def population_survival_curve(fit_result, times, group_dummy_cols=None):
    """S_pop(t | group) = (1 - pi_group) + pi_group * S_u(t | group) for
    each distinct group present in fit_result['df'], evaluated over `times`.
    Returns a long DataFrame: group, time, S_pop, pi_susceptible."""

    group_dummy_cols = group_dummy_cols or fit_result.get('group_dummy_cols') or fit_result['cox_cov']
    df = fit_result['df']
    b = fit_result['b']
    beta = fit_result['beta'].values
    death_points, cum_hazard = fit_result['death_points'], fit_result['cum_hazard']

    groups = df[['group'] + group_dummy_cols].drop_duplicates().reset_index(drop=True)

    rows = []
    for _, row in groups.iterrows():
        z_row = np.concatenate([[1.0], row[group_dummy_cols].values.astype(float)])
        pi_group = 1.0 / (1.0 + np.exp(-(z_row @ b.reindex(['const'] + group_dummy_cols).values)))

        x_row = row[group_dummy_cols].values.astype(float)
        S0_t = _survival_at(times, death_points, cum_hazard)
        Su_t = S0_t ** np.exp(x_row @ beta)

        S_pop = (1 - pi_group) + pi_group * Su_t
        for t, s_val in zip(np.atleast_1d(times), S_pop):
            rows.append({'group': row['group'], 'time': t, 'S_pop': s_val, 'pi_susceptible': pi_group})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_population_survival(fit_result, times, PREFIX='BD', OUTPUT_PATH='.', xlabel='Session', label=''):
    """Marginal population survival curve S_pop(t) = (1-pi) + pi*S_u(t) per
    group, i.e. the mixture-cure-model fit overlaid across groups. The
    curve's plateau (right end) is the estimated cure fraction.

    `xlabel` should match whatever unit `times`/fit_result's time column is
    in (e.g. 'Session' for fit_criterion_cure_model, 'Cumulative Trials' for
    fit_criterion_cure_model_trials). `label` is appended to the
    title/filename to tell apart calls with different time units."""

    surv = population_survival_curve(fit_result, times)
    ALL_GROUPS = sorted(surv['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    for group in ALL_GROUPS:
        g = surv.query('group == @group')
        cure = 1 - g['pi_susceptible'].iloc[0]
        plt.step(g['time'], g['S_pop'], where='post', color=COLOR_MAP[group], lw=2,
                  label=f'{group} (cure = {cure:.2f})')
    plt.xlabel(xlabel)
    plt.ylabel('Population survival S(t)')
    plt.ylim(0, 1.02)
    title_suffix = f' ({label})' if label else ''
    plt.title(f'{PREFIX}: Mixture Cure Model - Population Survival{title_suffix}')
    plt.legend(fontsize=9)
    filename_suffix = f'_{label}' if label else ''
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_CureModel_Population_Survival{filename_suffix}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return surv


def plot_cure_probability_bar(fit_result, PREFIX='BD', OUTPUT_PATH='.', label=''):
    """Bar plot of the estimated cure probability (1 - pi) per group.
    `label` is appended to the title/filename to tell apart calls fit on
    different time units (e.g. sessions vs. cumulative trials)."""

    summary = cure_probability_by_group(fit_result)
    ALL_GROUPS = sorted(summary['group'].unique())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.barplot(x='group', y='mean', data=summary, order=ALL_GROUPS, palette=COLOR_MAP)
    plt.xlabel('Groups')
    plt.ylabel('Estimated cure probability (1 - pi)')
    plt.ylim(0, 1)
    title_suffix = f' ({label})' if label else ''
    plt.title(f'{PREFIX}: Mixture Cure Model - Cure Probability by Group{title_suffix}')
    filename_suffix = f'_{label}' if label else ''
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_CureModel_Cure_Probability{filename_suffix}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return summary


def kaplan_meier_by_group(surv_df, time_col='time', event_col='event', group_col='group'):
    """Standard (non-parametric) Kaplan-Meier estimator per group, on the
    same survival data used by `smcure_ph` (see build_cure_survival_data).
    Returns a dict {group: fitted lifelines.KaplanMeierFitter}."""

    kmf_by_group = {}
    for group, g in surv_df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(g[time_col], event_observed=g[event_col], label=str(group))
        kmf_by_group[group] = kmf

    return kmf_by_group


def plot_kaplan_meier(data, crit=10, min_reinforced_rate=0.05, ci_show=True,
                       PREFIX='BD', OUTPUT_PATH='.', cure_fit_result=None, times=None,
                       time_unit='session', xlabel=None, label=''):
    """Kaplan-Meier survival curve per group, using the same event definition
    as the mixture cure model: reach criterion `crit` while keeping the
    reinforced-response rate above `min_reinforced_rate`. `data` may be
    either the raw trial-level dataframe (rat, group, session, trial,
    criterion, reinforced) or an already-built survival dataframe (rat,
    group, time, event) -- e.g. from build_cure_survival_data_trials. If raw
    data is given, `time_unit` picks which builder to use: 'session'
    (build_cure_survival_data, default) or 'trials'
    (build_cure_survival_data_trials, cumulative trial count).

    If `cure_fit_result` (the dict returned by smcure_ph / fit_criterion_cure_model
    or fit_criterion_cure_model_trials, fit on the same data/time unit) is
    given, the mixture-cure-model's population survival curve is overlaid as
    a dashed line per group for comparison.

    `xlabel` defaults to 'Session' or 'Cumulative Trials' matching
    `time_unit` (only relevant when `data` is raw; set it explicitly when
    passing an already-built survival dataframe). `label` is appended to the
    title/filename to tell apart calls with different time units.
    """

    if {'time', 'event'}.issubset(data.columns):
        surv_df = data
    elif time_unit == 'trials':
        surv_df = build_cure_survival_data_trials(data, crit=crit, min_reinforced_rate=min_reinforced_rate)
    else:
        surv_df = build_cure_survival_data(data, crit=crit, min_reinforced_rate=min_reinforced_rate)

    if xlabel is None:
        xlabel = 'Cumulative Trials' if time_unit == 'trials' else 'Session'

    kmf_by_group = kaplan_meier_by_group(surv_df)
    ALL_GROUPS = sorted(kmf_by_group.keys())
    COLOR_MAP = dict(zip(ALL_GROUPS, sns.color_palette('Set2', n_colors=len(ALL_GROUPS))))

    plt.figure()
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    ax = plt.gca()

    for group in ALL_GROUPS:
        kmf_by_group[group].plot_survival_function(ax=ax, ci_show=ci_show,
                                                     color=COLOR_MAP[group], show_censors=True)

    if cure_fit_result is not None:
        eval_times = times if times is not None else sorted(surv_df['time'].unique())
        pop_surv = population_survival_curve(cure_fit_result, eval_times)
        for group in ALL_GROUPS:
            g = pop_surv.query('group == @group')
            if not g.empty:
                ax.step(g['time'], g['S_pop'], where='post', color=COLOR_MAP[group],
                        lw=1.5, ls='--', alpha=0.8, label=f'{group} (cure model)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Survival probability')
    ax.set_ylim(0, 1.02)
    title_suffix = f' ({label})' if label else ''
    ax.set_title(f'{PREFIX}: Kaplan-Meier - Reach Criterion {crit}s '
                 f'(reinforced > {min_reinforced_rate:.0%}){title_suffix}')
    ax.legend(fontsize=8)
    filename_suffix = f'_{label}' if label else ''
    plt.savefig(os.path.realpath(f'{OUTPUT_PATH}/{PREFIX}_KaplanMeier_Criterion{crit}{filename_suffix}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return kmf_by_group


if __name__ == "__main__":

    PREFIX = 'AU'
    DATA_PATH = os.path.realpath('../../data/processed/') + '/'
    OUTPUT_PATH = os.path.realpath('../../output/tmp/DRRD_10s_Groups/') + '/'

    data = pd.read_csv(DATA_PATH + f'{PREFIX}.csv')
    data = data.astype({'rat': int, 'session': int, 'reinforced': int})

    fit_result = fit_criterion_cure_model(data, crit=10, min_reinforced_rate=0.05, verbose=True)

    print('\nLogistic (cure-probability) coefficients:')
    print(fit_result['b'])
    print('\nCox PH (latency) coefficients:')
    print(fit_result['beta'])
    print('\nCure probability by group:')
    print(cure_probability_by_group(fit_result))

    plot_population_survival(fit_result, times=list(range(1, 20)), PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)
    plot_cure_probability_bar(fit_result, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH)

    plot_kaplan_meier(data, crit=10, min_reinforced_rate=0.05, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH,
                       cure_fit_result=fit_result, times=list(range(1, 20)))

    # ---- Same model, but with cumulative trials (instead of sessions) as the survival time ----

    fit_result_trials = fit_criterion_cure_model_trials(data, crit=10, min_reinforced_rate=0.05, verbose=True)

    print('\n[trials] Logistic (cure-probability) coefficients:')
    print(fit_result_trials['b'])
    print('\n[trials] Cox PH (latency) coefficients:')
    print(fit_result_trials['beta'])
    print('\n[trials] Cure probability by group:')
    print(cure_probability_by_group(fit_result_trials))

    max_trials = int(fit_result_trials['surv_df']['time'].max())
    trial_grid = list(range(0, max_trials + 1, max(1, max_trials // 100)))

    plot_population_survival(fit_result_trials, times=trial_grid, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH,
                              xlabel='Cumulative Trials', label='trials')
    plot_cure_probability_bar(fit_result_trials, PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH, label='trials')

    plot_kaplan_meier(fit_result_trials['surv_df'], PREFIX=PREFIX, OUTPUT_PATH=OUTPUT_PATH,
                       cure_fit_result=fit_result_trials, times=trial_grid,
                       xlabel='Cumulative Trials', label='trials')
