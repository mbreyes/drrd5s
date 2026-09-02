"""
Analise de poder por simulacao - aquisicao de resposta sustentada (hold 10s)
sob 5 regras de incremento de criterio.

Alvo do poder:
  (1) Omnibus: diferenca entre as 5 regras na VELOCIDADE de aprendizagem.
  (2) Contraste de dose C1: tendencia dose-resposta entre as 3 exponenciais
      (10%, 20%, 40%), usando o passo em log centrado como pesos.

Modelo gerador (GLMM logistico de crescimento):
  Para o rato i (grupo g), sessao s (s = 0..S-1, centrada):
     logit(pi_is) = b0_i + b1_i * s_c
     b0_i ~ N(beta0, tau0^2)                      (intercepto aleatorio)
     b1_i ~ N(beta1_g, tau1^2)                    (inclinacao aleatoria; media do grupo = velocidade)
  Em cada sessao: k_is ~ Binomial(n_trials, pi_is)  (peso da gaussiana longa observado)

Analise (duas etapas, aproximacao robusta ao GLMM):
  - ajusta a inclinacao logit por animal (regressao dos logits empiricos ~ sessao);
  - testa no nivel do animal:
       omnibus  -> ANOVA de uma via (5 grupos);
       dose C1  -> contraste linear entre as 3 exponenciais (df erro = N-5).
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20260716)

# ---------- parametros fixos do desenho ----------
S = 20                      # sessoes por animal
n_trials = 40               # tentativas por sessao (para estimar o peso pi)
beta0 = 0.0                 # logit(pi) medio no centro do treino (~0.5)
tau0 = 1.0                  # DP entre animais do intercepto (logit)
beta1_base = 0.15           # inclinacao media (logit/sessao) = velocidade base
tau1 = 0.10                 # DP entre animais da inclinacao (nuisance-chave)

s = np.arange(S)
s_c = s - s.mean()          # sessao centrada
Sxx = np.sum(s_c**2)        # p/ regressao da inclinacao

# passos em log centrados (pesos de C1) para Exp10, Exp20, Exp40
logsteps = np.log([1.10, 1.20, 1.40])
c_exp = logsteps - logsteps.mean()          # [-0.109, -0.022, +0.132]
# vetor de contraste C1 nos 5 grupos (Exp10,Exp20,Exp40,Linear,Perc)
C1 = np.array([c_exp[0], c_exp[1], c_exp[2], 0.0, 0.0])

# ---------- cenarios de tamanho de efeito ----------
# 'a' escala a dose: media da inclinacao do grupo exp = beta1_base + a * passo_log_centrado.
# Gap Exp40-Exp10 = a * (0.132 - (-0.109)) = a * 0.241 (logit/sessao).
scenarios = {
    "Conservador (a=0.4)": 0.4,   # gap ~0.10 logit/sessao ; ~1.0 tau1
    "Realista (a=0.7)":    0.7,   # gap ~0.17 logit/sessao ; ~1.7 tau1
    "Otimista (a=1.1)":    1.1,   # gap ~0.27 logit/sessao ; ~2.7 tau1
}

n_per_group_grid = [6, 8, 10, 12, 15, 20]
n_groups = 5
R = 1500                    # replicacoes por celula
alpha = 0.05

def group_mean_slopes(a):
    # Exp10,Exp20,Exp40 seguem a dose; Linear e Percentil = base (neutros p/ C1)
    exp_slopes = beta1_base + a * c_exp
    return np.array([exp_slopes[0], exp_slopes[1], exp_slopes[2],
                     beta1_base, beta1_base])

def simulate_animal_slopes(nper, mean_slopes):
    """Retorna (N,) inclinacao logit estimada por animal e (N,) rotulo de grupo."""
    N = nper * n_groups
    grp = np.repeat(np.arange(n_groups), nper)
    b0 = rng.normal(beta0, tau0, size=N)
    b1 = rng.normal(mean_slopes[grp], tau1)
    # trajetoria: logit(pi) por sessao
    lin = b0[:, None] + b1[:, None] * s_c[None, :]      # (N,S)
    pi = 1.0 / (1.0 + np.exp(-lin))
    k = rng.binomial(n_trials, pi)                       # (N,S) contagens de "longo"
    # logit empirico com correcao de Haldane (evita 0/inf)
    p_hat = (k + 0.5) / (n_trials + 1.0)
    y = np.log(p_hat / (1.0 - p_hat))                    # (N,S)
    # inclinacao OLS por animal
    y_c = y - y.mean(axis=1, keepdims=True)
    slope = (y_c * s_c[None, :]).sum(axis=1) / Sxx       # (N,)
    return slope, grp

def test_power(nper, a):
    om_rej = 0
    c1_rej = 0
    dfg = n_groups
    N = nper * n_groups
    df_err = N - n_groups
    mean_slopes = group_mean_slopes(a)
    tcrit = stats.t.ppf(1 - alpha/2, df_err)
    for _ in range(R):
        slope, grp = simulate_animal_slopes(nper, mean_slopes)
        groups = [slope[grp == g] for g in range(n_groups)]
        # omnibus ANOVA
        F, p_om = stats.f_oneway(*groups)
        if p_om < alpha:
            om_rej += 1
        # contraste C1 (linear dose entre exponenciais)
        means = np.array([gg.mean() for gg in groups])
        # MSE agrupado
        ss_within = sum(((gg - gg.mean())**2).sum() for gg in groups)
        mse = ss_within / df_err
        L = np.sum(C1 * means)
        se = np.sqrt(mse * np.sum(C1**2 / nper))
        t = L / se
        if abs(t) > tcrit:
            c1_rej += 1
    return om_rej / R, c1_rej / R

# ---------- executar ----------
results = {}  # scenario -> dict(n -> (pow_omnibus, pow_c1))
print(f"{'Cenario':<22}{'n/grupo':>8}{'N total':>9}{'Poder omnibus':>15}{'Poder dose C1':>15}")
print("-" * 69)
for sc_name, a in scenarios.items():
    results[sc_name] = {}
    for nper in n_per_group_grid:
        po, pc = test_power(nper, a)
        results[sc_name][nper] = (po, pc)
        print(f"{sc_name:<22}{nper:>8}{nper*n_groups:>9}{po:>15.3f}{pc:>15.3f}")
    print("-" * 69)

import json
with open("power_results.json", "w") as f:
    json.dump({k: {str(n): v for n, v in d.items()} for k, d in results.items()}, f, indent=2)
print("\nSalvo power_results.json")