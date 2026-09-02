#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# DRRD 10s -- dados AW e AY apenas
#
# 1) Ajuste de gaussiana dupla por maxima verossimilhanca (MLE), por rato e
#    por sessao, sobre log(duration) -- mesma parametrizacao usada em
#    code/python/model_fitting.py (MLE(..., log = TRUE)).
# 2) Curva de Kaplan-Meier (nao-parametrica) para o evento "atingiu o
#    criterio de 10s", sobreposta a curva de sobrevivencia populacional do
#    modelo de cura misto, como checagem visual/numerica de ajuste.
# 3) Modelo de cura misto (smcure, model = "ph"): componente de incidencia
#    (logistica) ~ grupo, componente de latencia (Cox PH) ~ grupo. Evento =
#    atingir o criterio de 10s mantendo taxa de reforco > 5% na sessao
#    (mesma definicao de code/python/cure_model.py).
# 4) Teste bootstrap (estratificado por evento/censura, como o proprio
#    smcure faz) com comparacao pareada da probabilidade de cura predita
#    entre grupos.
#
# @author: ailacamara

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(survival)
  library(smcure)
})

set.seed(123)

# install.packages("smcure")  # se indisponivel no CRAN, instale do arquivo:
# install.packages("https://cran.r-project.org/src/contrib/Archive/smcure/smcure_2.0.tar.gz",
#                   repos = NULL, type = "source")

# ============================================================
# CONSTANTES / CAMINHOS (mesma estrutura de code/python/*_data_analysis.py)
# ============================================================

PREFIX       <- "AW and AY DRRD"
DATA_PATH    <- normalizePath(file.path("Downloads", "DRRD","drrd5s", "data", "processed"))
OUTPUT_PATH  <- normalizePath(file.path("Downloads", "DRRD","drrd5s", "output", "tmp", "DRRD_10s_R"))
PREFIXES     <- c("AW1", "AY12")   # OBS do usuario: só dados de AW e AY
RATS         <- setdiff(39:74, 50)   # rato 50 excluido
CRIT         <- 10
MIN_REINFORCED_RATE <- 0.05

# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

#' Le e concatena os CSVs de AW1/AY12, filtrando ratos e removendo a sessao 5
#' (mesmo filtro aplicado em Drrd_10s_groups_data_analysis.py para esse
#' intervalo de ratos) e duracoes >= 30s (outliers de trial, mesmo corte do
#' pipeline em Python).
load_drrd_aw_ay_data <- function(prefixes = PREFIXES, data_path = DATA_PATH,
                                  rats = RATS) {
  data <- purrr::map_dfr(prefixes, function(p) {
    read.csv(file.path(data_path, paste0(p, ".csv")))
  })

  data %>%
    mutate(rat = as.integer(rat),
           session = as.integer(session),
           reinforced = as.integer(reinforced)) %>%
    filter(rat %in% rats, session != 5, duration < 30) %>%
    arrange(rat, session, trial)
}

# ============================================================
# 2. GAUSSIANA DUPLA -- AJUSTE POR MAXIMA VEROSSIMILHANCA
#    (mesma parametrizacao de model_fitting.MLE(..., log = TRUE))
# ============================================================

.gaussian_pdf <- function(x, mu, sigma) {
  exp(-0.5 * ((x - mu) / sigma)^2) / (sigma * sqrt(2 * pi))
}

# Estimativa inicial "crua": divide os dados pela mediana e usa media/dp de
# cada metade (mesma heuristica de estimate_init_params em model_fitting.py).
.estimate_init_params <- function(x) {
  med <- median(x)
  lower <- x[x <= med]
  upper <- x[x > med]

  mu1 <- mean(lower)
  mu2 <- mean(upper)
  sigma1 <- if (length(lower) > 1) sd(lower) else 0.5
  sigma2 <- if (length(upper) > 1) sd(upper) else 0.5

  c(gamma = 0.5, mu1 = mu1, sigma1 = sigma1, delta_mu = max(mu2 - mu1, 1e-3), sigma2 = sigma2)
}

.negative_log_likelihood <- function(params, x) {
  gamma <- params[1]; mu1 <- params[2]; sigma1 <- params[3]
  delta_mu <- params[4]; sigma2 <- params[5]
  mu2 <- mu1 + delta_mu

  p1 <- (1 - gamma) * .gaussian_pdf(x, mu1, sigma1)
  p2 <- gamma * .gaussian_pdf(x, mu2, sigma2)
  probs <- pmax(p1 + p2, 1e-9)

  -sum(log(probs))
}

#' Ajuste de gaussiana dupla (MLE) sobre log(duration), para um unico vetor
#' de duracoes (um rato em uma sessao). gamma = peso da gaussiana 2 (respostas
#' longas); mu1/sigma1, mu2/sigma2 = media/dp de cada componente (escala log).
fit_double_gaussian_mle <- function(duration, min_n = 5) {
  x <- log(duration[duration > 0])
  if (length(x) < min_n) {
    stop(sprintf("dados insuficientes (n = %d) para ajustar a gaussiana dupla", length(x)))
  }

  init <- .estimate_init_params(x)

  fit <- optim(
    par = init,
    fn = .negative_log_likelihood,
    x = x,
    method = "L-BFGS-B",
    lower = c(0, -Inf, 1e-6, 0, 1e-6),
    upper = c(1, Inf, Inf, Inf, Inf)
  )

  gamma <- fit$par[["gamma"]]
  mu1 <- fit$par[["mu1"]]
  sigma1 <- fit$par[["sigma1"]]
  mu2 <- mu1 + fit$par[["delta_mu"]]
  sigma2 <- fit$par[["sigma2"]]

  tibble(gamma = gamma, mu1 = mu1, sigma1 = sigma1, mu2 = mu2, sigma2 = sigma2,
         n_trials = length(x), convergence = fit$convergence)
}

#' Aplica fit_double_gaussian_mle a cada combinacao rato x sessao. Grupos com
#' poucos trials (< min_n) sao pulados (com aviso) em vez de derrubar o script.
fit_double_gaussian_all_sessions <- function(data, min_n = 5) {
  safe_fit <- purrr::possibly(function(x) fit_double_gaussian_mle(x, min_n = min_n),
                               otherwise = NULL)

  data %>%
    group_by(rat, group, session) %>%
    group_modify(~ {
      res <- safe_fit(.x$duration)
      if (is.null(res)) {
        warning(sprintf("rato %s, sessao %s: ajuste pulado (dados insuficientes)",
                         .y$rat, .y$session))
        return(tibble())
      }
      res
    }) %>%
    ungroup()
}

# ============================================================
# 3. DADOS DE SOBREVIVENCIA -- EVENTO = ATINGIR CRITERIO DE 10s
#    (mesma definicao de build_cure_survival_data em cure_model.py)
# ============================================================

#' Uma linha por rato: rat, group, time (primeira sessao em que o criterio
#' maximo atingiu `crit` E a taxa de reforco superou `min_reinforced_rate`),
#' event = 1. Se isso nunca ocorre, o rato e censurado (event = 0) na ultima
#' sessao observada.
build_cure_survival_data <- function(data, crit = CRIT, min_reinforced_rate = MIN_REINFORCED_RATE) {
  session_stats <- data %>%
    group_by(rat, group, session) %>%
    summarise(max_criterion = max(criterion),
              reinforced_rate = mean(reinforced),
              .groups = "drop") %>%
    mutate(success = max_criterion >= crit & reinforced_rate > min_reinforced_rate)

  session_stats %>%
    arrange(rat, session) %>%
    group_by(rat, group) %>%
    summarise(
      time = if (any(success)) min(session[success]) else max(session),
      event = as.integer(any(success)),
      .groups = "drop"
    ) %>%
    mutate(time = as.numeric(time))
}

# ============================================================
# 4. KAPLAN-MEIER (nao-parametrico)
# ============================================================

#' Curva de Kaplan-Meier por grupo, em formato longo (para ggplot2 e para
#' comparar com a curva do modelo de cura).
kaplan_meier_by_group_df <- function(surv_df) {
  km_fit <- survival::survfit(Surv(time, event) ~ group, data = surv_df)
  s <- summary(km_fit)

  tibble(
    time = s$time,
    surv = s$surv,
    lower = s$lower,
    upper = s$upper,
    group = sub("^group=", "", s$strata)
  )
}

# ============================================================
# 5. MODELO DE CURA MISTO (smcure, model = "ph")
#    Incidencia (logistica) ~ grupo | Latencia (Cox PH) ~ grupo
#
#    Convencao do smcure: os coeficientes `b` modelam a probabilidade de o
#    animal ser SUSCETIVEL (i.e. vir a aprender), pi = plogis(Z %*% b) --
#    ver smcure:::em. Na notacao do usuario, pi = P(cura) = P(nunca
#    aprender); portanto aqui `p_susceptible` = 1 - pi(usuario), e
#    `cure_probability` (= 1 - p_susceptible) e exatamente o "pi" pedido:
#    a probabilidade de o animal nunca aprender.
# ============================================================

`%||%` <- function(a, b) if (is.null(a)) b else a

#' smcure monta a matriz de desenho lendo as colunas cruas de `data` (nao usa
#' model.matrix/contrasts internamente -- ver smcure()), entao um fator com
#' mais de 2 niveis passado direto quebra a multiplicacao de matrizes dentro
#' de em(). E preciso criar as variaveis dummy (0/1) do grupo manualmente e
#' referencia-las pelo nome na formula/cureform.
build_group_dummies <- function(df, group_levels) {
  design <- model.matrix(~ group, data = df %>% mutate(group = factor(group, levels = group_levels)))
  dummy_cols <- colnames(design)[-1]
  dummy_df <- as.data.frame(design[, -1, drop = FALSE])
  colnames(dummy_df) <- dummy_cols
  list(dummy_df = dummy_df, dummy_cols = dummy_cols)
}

#' Ajusta o modelo de cura misto (incidencia logistica + latencia Cox PH),
#' com "group" (dummy-codificado) como fator de risco categorico em ambos os
#' componentes.
fit_group_cure_model <- function(surv_df, group_levels = NULL, Var = TRUE, nboot = 100) {
  group_levels <- group_levels %||% sort(unique(surv_df$group))
  dummies <- build_group_dummies(surv_df, group_levels)
  surv_df <- dplyr::bind_cols(surv_df, dummies$dummy_df)

  rhs <- paste(dummies$dummy_cols, collapse = " + ")
  fml <- as.formula(paste("Surv(time, event) ~", rhs))
  cureform <- as.formula(paste("~", rhs))

  invisible(utils::capture.output(
    fit <- smcure::smcure(fml, cureform = cureform, data = surv_df,
                           model = "ph", Var = Var, nboot = nboot)
  ))
  fit$dummy_cols <- dummies$dummy_cols
  fit$group_levels <- group_levels
  fit
}

#' Probabilidade de cura (nunca aprender) predita por grupo, a partir dos
#' coeficientes de incidencia (fit$b) do modelo de cura misto.
predict_group_cure_probabilities <- function(fit, group_levels) {
  newdata <- data.frame(group = factor(group_levels, levels = group_levels))
  Z <- model.matrix(~ group, data = newdata)

  p_susceptible <- as.numeric(plogis(Z %*% fit$b))

  tibble(group = group_levels,
         p_susceptible = p_susceptible,
         cure_probability = 1 - p_susceptible)
}

#' Reconstroi a funcao de sobrevivencia basal S0(t) (Cox PH, degrau) a partir
#' dos pares (fit$Time, fit$s) devolvidos pelo smcure -- fit$s ja e S0
#' avaliada no tempo de cada sujeito (ver smcure:::smsurv / smcure:::em), e
#' como o risco basal so muda nos pontos de evento (que estao entre os
#' proprios fit$Time), o "ultimo valor conhecido <= t" reproduz exatamente a
#' funcao degrau em qualquer t de consulta.
baseline_survival_smcure <- function(fit) {
  ord <- order(fit$Time)
  time_sorted <- fit$Time[ord]
  s_sorted <- fit$s[ord]

  function(times) {
    vapply(times, function(t) {
      idx <- which(time_sorted <= t)
      if (length(idx) == 0) return(1)
      s_sorted[max(idx)]
    }, numeric(1))
  }
}

#' Curva de sobrevivencia populacional do modelo de cura,
#' S_pop(t | grupo) = (1 - pi) + pi * S0(t)^exp(beta * grupo), avaliada em
#' `times` para cada grupo -- para sobrepor a curva de Kaplan-Meier.
population_survival_curve_smcure <- function(fit, group_levels, times) {
  S0 <- baseline_survival_smcure(fit)

  purrr::map_dfr(group_levels, function(g) {
    newdata <- data.frame(group = factor(g, levels = group_levels))
    design <- model.matrix(~ group, data = newdata)
    x_row <- design[, -1, drop = FALSE]

    p_susceptible <- as.numeric(plogis(design %*% fit$b))
    Su <- S0(times) ^ as.numeric(exp(x_row %*% fit$beta))
    S_pop <- (1 - p_susceptible) + p_susceptible * Su

    tibble(group = g, time = times, S_pop = S_pop,
           p_susceptible = p_susceptible, cure_probability = 1 - p_susceptible)
  })
}

# ============================================================
# 6. BOOTSTRAP -- COMPARACAO PAREADA DA PROBABILIDADE DE CURA ENTRE GRUPOS
#    (reamostragem estratificada por evento/censura, como o proprio smcure
#    faz internamente para os erros-padrao -- ver smcure() com Var = TRUE)
# ============================================================

#' Reamostra `surv_df` (estratificado por evento/censura) `nboot` vezes,
#' reajusta o modelo de cura em cada reamostra (Var = FALSE, para nao
#' aninhar o bootstrap interno do smcure) e devolve a probabilidade de cura
#' predita por grupo em cada reamostra bem-sucedida.
bootstrap_group_cure_probabilities <- function(surv_df, group_levels, nboot = 500, seed = 123) {
  set.seed(seed)

  events <- which(surv_df$event == 1)
  censored <- which(surv_df$event == 0)

  boot_list <- vector("list", nboot)

  for (b in seq_len(nboot)) {
    boot_idx <- c(sample(events, length(events), replace = TRUE),
                  sample(censored, length(censored), replace = TRUE))
    boot_df <- surv_df[boot_idx, ]

    fit_b <- tryCatch(
      fit_group_cure_model(boot_df, group_levels = group_levels, Var = FALSE),
      error = function(e) NULL
    )
    if (is.null(fit_b)) next

    preds <- tryCatch(predict_group_cure_probabilities(fit_b, group_levels),
                       error = function(e) NULL)
    if (is.null(preds)) next

    boot_list[[b]] <- preds$cure_probability
  }

  boot_list <- purrr::compact(boot_list)
  boot_mat <- do.call(rbind, boot_list)
  colnames(boot_mat) <- group_levels

  list(boot_mat = boot_mat, n_success = nrow(boot_mat), n_requested = nboot)
}

#' Comparacao pareada (todos os pares de grupos) da diferenca de
#' probabilidade de cura predita, a partir da distribuicao bootstrap:
#' IC 95% percentil e p-valor bicaudal (2 * proporcao de sinal invertido).
pairwise_bootstrap_comparison <- function(boot_result) {
  boot_mat <- boot_result$boot_mat
  groups <- colnames(boot_mat)
  pairs <- combn(groups, 2, simplify = FALSE)

  purrr::map_dfr(pairs, function(pr) {
    diff_boot <- boot_mat[, pr[1]] - boot_mat[, pr[2]]
    ci <- quantile(diff_boot, c(0.025, 0.975))
    p_value <- 2 * min(mean(diff_boot > 0), mean(diff_boot < 0))

    tibble(group_a = pr[1], group_b = pr[2],
           diff_mean = mean(diff_boot),
           ci_low = unname(ci[1]), ci_high = unname(ci[2]),
           p_value = pmin(p_value, 1))
  })
}

# ============================================================
# EXECUCAO
# ============================================================

if (sys.nframe() == 0) {

  # ---- 1. Carrega dados (AW1 + AY12 apenas) ----

  data <- load_drrd_aw_ay_data()
  ALL_GROUPS <- sort(unique(data$group))

  # ---- 2. Gaussiana dupla por rato x sessao ----

  cat("Ajustando gaussiana dupla (MLE) por rato x sessao...\n")
  df_parameters <- fit_double_gaussian_all_sessions(data)

  write.csv(df_parameters,
            file.path(OUTPUT_PATH, paste0(PREFIX, "_DoubleGaussian_Parameters_R.csv")),
            row.names = FALSE)

  p_mu2 <- ggplot(df_parameters, aes(x = session, y = mu2, color = group, fill = group)) +
    stat_summary(fun = mean, geom = "line", linewidth = 1) +
    stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.2, color = NA) +
    labs(title = paste(PREFIX, ": mu2 (gaussiana dupla, MLE) por sessao"),
         x = "Sessao", y = expression(mu[2] ~ "(log-duracao)")) +
    theme_minimal()
  ggsave(file.path(OUTPUT_PATH, paste0(PREFIX, "_Lineplot_mu2_MLE_R.png")),
         p_mu2, width = 7, height = 5, dpi = 300)

  # ---- 3. Dados de sobrevivencia (evento = atingir criterio 10s) ----

  surv_df <- build_cure_survival_data(data, crit = CRIT, min_reinforced_rate = MIN_REINFORCED_RATE)

  # ---- 4. Kaplan-Meier (nao-parametrico) ----

  cat("Ajustando Kaplan-Meier...\n")
  km_df <- kaplan_meier_by_group_df(surv_df)

  # ---- 5. Modelo de cura misto (smcure) ----

  cat("Ajustando modelo de cura misto (smcure, incidencia logistica + latencia Cox PH)...\n")
  cure_fit <- fit_group_cure_model(surv_df, group_levels = ALL_GROUPS, Var = TRUE, nboot = 100)

  cat("\nCoeficientes de incidencia (probabilidade de ser suscetivel, i.e. de aprender):\n")
  print(data.frame(term = cure_fit$bnm, estimate = unname(cure_fit$b),
                    p_value = unname(cure_fit$b_pvalue)), row.names = FALSE)

  cat("\nCoeficientes de latencia (Cox PH, tempo ate atingir o criterio | suscetivel):\n")
  print(data.frame(term = cure_fit$betanm, estimate = unname(cure_fit$beta),
                    p_value = unname(cure_fit$beta_pvalue)), row.names = FALSE)

  group_cure_probs <- predict_group_cure_probabilities(cure_fit, ALL_GROUPS)
  cat("\nProbabilidade de cura (nunca atingir o criterio de 10s) por grupo:\n")
  print(group_cure_probs)

  # ---- 6. Kaplan-Meier x modelo de cura: checagem de ajuste ----

  eval_times <- sort(unique(surv_df$time))
  pop_surv <- population_survival_curve_smcure(cure_fit, ALL_GROUPS, eval_times)

  COLOR_MAP <- setNames(scales::hue_pal()(length(ALL_GROUPS)), ALL_GROUPS)

  p_km <- ggplot() +
    geom_step(data = km_df, aes(x = time, y = surv, color = group), linewidth = 1) +
    geom_ribbon(data = km_df, aes(x = time, ymin = lower, ymax = upper, fill = group),
                alpha = 0.15, color = NA, na.rm = TRUE) +
    geom_step(data = pop_surv, aes(x = time, y = S_pop, color = group),
              linetype = "dashed", linewidth = 1) +
    coord_cartesian(ylim = c(0, 1.02)) +
    labs(title = paste0(PREFIX, ": Kaplan-Meier (solido) vs. Modelo de Cura (tracejado)"),
         subtitle = paste0("Evento = atingir criterio ", CRIT, "s (reforco > ",
                            scales::percent(MIN_REINFORCED_RATE), ")"),
         x = "Sessao", y = "Probabilidade de sobrevivencia (nao atingiu o criterio)") +
    theme_minimal()
  ggsave(file.path(OUTPUT_PATH, paste0(PREFIX, "_KaplanMeier_vs_CureModel_R.png")),
         p_km, width = 8, height = 6, dpi = 300)

  # Checagem numerica simples: platô da KM (sobrevivencia na ultima sessao
  # observada de cada grupo) vs. fracao de cura estimada pelo modelo --
  # quanto mais proximos, melhor o modelo de cura explica o platô observado.
  km_plateau <- km_df %>%
    group_by(group) %>%
    filter(time == max(time)) %>%
    summarise(km_plateau_survival = surv, .groups = "drop")

  goodness_of_fit <- km_plateau %>%
    left_join(group_cure_probs, by = "group") %>%
    mutate(abs_diff = abs(km_plateau_survival - cure_probability))

  cat("\nChecagem de ajuste -- platô observado (KM) vs. fracao de cura predita (modelo):\n")
  print(goodness_of_fit)

  # ---- 7. Bootstrap -- comparacao pareada entre grupos ----

  cat("\nBootstrap (reamostragem estratificada evento/censura) da probabilidade de cura por grupo...\n")
  boot_result <- bootstrap_group_cure_probabilities(surv_df, group_levels = ALL_GROUPS, nboot = 500)
  cat(sprintf("Reamostras bem-sucedidas: %d / %d\n", boot_result$n_success, boot_result$n_requested))

  pairwise_results <- pairwise_bootstrap_comparison(boot_result)
  cat("\nComparacao pareada (bootstrap) da probabilidade de cura entre grupos:\n")
  print(pairwise_results)

  write.csv(pairwise_results,
            file.path(OUTPUT_PATH, paste0(PREFIX, "_CureModel_Bootstrap_Pairwise_R.csv")),
            row.names = FALSE)

  boot_long <- as_tibble(boot_result$boot_mat) %>%
    pivot_longer(everything(), names_to = "group", values_to = "cure_probability")

  p_boot <- ggplot(boot_long, aes(x = group, y = cure_probability, fill = group)) +
    geom_violin(alpha = 0.5, trim = FALSE) +
    geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.7) +
    labs(title = paste0(PREFIX, ": Probabilidade de cura por grupo (bootstrap, n = ",
                         boot_result$n_success, ")"),
         x = "Grupo", y = "Probabilidade de cura (nunca atingir 10s)") +
    theme_minimal() +
    theme(legend.position = "none")
  ggsave(file.path(OUTPUT_PATH, paste0(PREFIX, "_CureModel_Bootstrap_Violin_R.png")),
         p_boot, width = 7, height = 5, dpi = 300)

  cat("\nConcluido. Resultados salvos em:", OUTPUT_PATH, "\n")
}

