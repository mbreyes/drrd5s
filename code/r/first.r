# Definindo os parâmetros
set.seed(123) # Para reprodutibilidade - fixa a sequência aleatória
# e ajuda a reproduzir os mesmos resultados em análises
# estatísticas.
mu <- 80 # Média populacional
sigma <- 20 # Desvio padrão populacional (sqrt(400))
n <- 50 # Tamanho da amostra
alpha <- 0.05 # Nível de significância (para IC de 95%)
z <- qnorm(1- alpha/2) # Valor crítico para 95% de confiança
# Gerando uma amostra aleatória única para manter consistência nos cálculos
# e gráficos
amostra <- rnorm(n, mean = mu, sd = sigma)
media_amostral <- mean(amostra)
erro_padrao <- sigma / sqrt(n)
# Calculando o intervalo de confiança
IC_inferior <- media_amostral- z * erro_padrao
IC_superior <- media_amostral + z * erro_padrao

