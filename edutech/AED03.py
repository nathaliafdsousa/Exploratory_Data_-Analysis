import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
# Estilo dos gráficos e formato dos números (4 casas decimais)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)

caminho = ('edutech/edutech_performance.csv')

try:
    df_raw = pd.read_csv(caminho)
    print("Dimensão: ", df_raw.shape)
    print(df_raw.head())
except FileNotFoundError:
  print('Arquivo não encontrado')

# Visão geral: tipos,nulos e estatística básica

df_raw.info()
print('Valores nulos por colunas')
print(df_raw.isna().sum())
df_raw.describe()

"""1.1 SENEAMENTO DE DADOS"""

# 1) IDs duplicados
print('IDs duplicadps: ', df_raw['id_aluno'].duplicated().sum())

# 2) Maiores valores de horas de estudo
print('\nMaiores valores de estudo:')
print(df_raw.sort_values(by='horas_estudo_semana', ascending=False).head())

# 3) Notas fora da escala permitada (0 a 10)
print('\nNotas fora da escala')
print(df_raw[(df_raw['nota_exame_final'] < 0) | (df_raw['nota_exame_final'] > 10)])

# 4) Constância entre aprovado  e a regra da nota >= 6
notas = np.where(df_raw['nota_exame_final'] >= 6.0, 'Sim', 'Não')
print('\nCOnstância de aprovados')
print((df_raw['aprovado'] == notas).value_counts().to_string())

# 5) Faixa da taxa de conclusão (deveria estar entre 0 a 100)
print("\nTaxa de conclusão — mín/máx:", round(df_raw["taxa_conclusao_pct"].min(), 2), "/", round(df_raw["taxa_conclusao_pct"].max(), 2))
print("\nValores > 100:", (df_raw["taxa_conclusao_pct"] > 100).sum(), "| Valores < 0:", (df_raw["taxa_conclusao_pct"] < 0).sum(), "| Valores == 100 (saturados):", (df_raw["taxa_conclusao_pct"] == 100).sum())



df = df_raw.copy()   # base limpa

# (a) Outlier de horas: cerca extrema Q3 + 3*IQR ---
q1_h, q3_h = df["horas_estudo_semana"].quantile([0.25, 0.75])
iqr_h = q3_h - q1_h
limite_extremo = q3_h + 3 * iqr_h
print(f"Limite extremo de horas: {limite_extremo:.2f}")

mask_erro_horas = df["horas_estudo_semana"] > limite_extremo
print("Registros marcados como erro:", mask_erro_horas.sum())
df.loc[mask_erro_horas, "horas_estudo_semana"] = np.nan

# (b) Imputação dos ausentes (originais + o outlier) pela MEDIANA
mediana_horas = df["horas_estudo_semana"].median()
n_nulos = df["horas_estudo_semana"].isna().sum()
df["horas_estudo_semana"] = df["horas_estudo_semana"].fillna(mediana_horas)
print(f"{n_nulos} valores imputados com a mediana = {mediana_horas:.2f} h")

# (c) Remove notas fora da escala 0-10
antes = len(df)
df = df[df["nota_exame_final"].between(0, 10)].reset_index(drop=True)
print(f"Linhas removidas por nota inválida: {antes - len(df)}")

# (d) Normalização da taxa de conclusão para [0, 1]
df["taxa_conclusao_norm"] = df["taxa_conclusao_pct"] / 100
print("Faixa normalizada:", round(df["taxa_conclusao_norm"].min(), 3), "a", round(df["taxa_conclusao_norm"].max(), 3))

print("\nNulos restantes:", df.isna().sum().sum(), "| Dimensão final:", df.shape)

"""1.2 PORCENTAGEM E VARIAÇÃO PORCENTUAL

"""



# Ordenação da base por horas de estudo (decrescente)
df_ordenado = df.sort_values("horas_estudo_semana", ascending=False)
print(df_ordenado.head())

# Média de horas por nível de ensino
media_nivel = df.groupby("nivel_ensino")["horas_estudo_semana"].mean()
print("Média de horas por nível:\n", media_nivel.to_string())

# Variação percentual: Pós-Graduação em relação à Graduação
grad, pos = media_nivel["Graduação"], media_nivel["Pós-Graduação"]
variacao_pct = (pos - grad) / grad * 100
print(f"\nVariação % (Pós vs Graduação): {variacao_pct:.2f}%")

# Proporção de aprovados por campus (em %)
prop_aprov_campus = (df.groupby("regiao_campus")["aprovado"]
                       .apply(lambda s: (s == "Sim").mean() * 100)
                       .sort_values(ascending=False))
print("\nAprovados por campus (%):\n", prop_aprov_campus.round(2).to_string())

"""2.1 MEDIDAS DE TENDÊNCIA CENTRAL E DISPERSÃO"""



def resumo_descritivo(x):
    """Retorna todas as medidas pedidas no item 2.1 para uma série numérica."""
    q1, q2, q3 = x.quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    return pd.Series({
        "Média": x.mean(),
        "Mediana": x.median(),
        "Moda (arred. 1 casa)": x.round(1).mode().iloc[0],   # variável contínua: arredonda para achar moda
        "Média aparada 5%": stats.trim_mean(x, 0.05),
        "Variância (amostral)": x.var(ddof=1),
        "Desvio padrão": x.std(ddof=1),
        "Amplitude total": x.max() - x.min(),
        "IQR": iqr,
        "CV (%)": x.std(ddof=1) / x.mean() * 100,
        "Assimetria de Bowley": (q3 + q1 - 2 * q2) / (q3 - q1),
        "Curtose (excesso)": stats.kurtosis(x),               # 0 = normal; >0 caudas pesadas
    })

nota = df["nota_exame_final"]
tabela_21 = resumo_descritivo(nota).to_frame("nota_exame_final (limpa)")
tabela_21

"""2.2 SENSIBILIDADE A OUTLINERS"""

# Compara a base ORIGINAL (com a nota -2,5) com a base LIMPA
nota_raw = df_raw["nota_exame_final"]

comparacao = pd.concat([
    resumo_descritivo(nota_raw).rename("Com outlier (-2,5)"),
    resumo_descritivo(nota).rename("Sem outlier"),
], axis=1)
print(comparacao)

# Diferença Média - Mediana
print(f"\n|Média - Mediana| COM outlier: {abs(nota_raw.mean() - nota_raw.median()):.4f}")
print(f"|Média - Mediana| SEM outlier: {abs(nota.mean() - nota.median()):.4f}")

# Demonstração matemática: contribuição do outlier para a soma dos quadrados
#   s^2 = SQ / (n-1), onde SQ = soma (x_i - média)^2
outlier = nota_raw.min()
SQ_total = ((nota_raw - nota_raw.mean()) ** 2).sum()
SQ_outlier = (outlier - nota_raw.mean()) ** 2
print(f"\nSQ total = {SQ_total:.2f} | SQ do outlier = {SQ_outlier:.2f}")
print(f"Um único ponto ({outlier}) responde por {SQ_outlier / SQ_total * 100:.1f}% da soma dos quadrados")
print(f"DP com outlier = {nota_raw.std(ddof=1):.4f} | DP sem outlier = {nota.std(ddof=1):.4f} "
      f"(variação de {(nota_raw.std(ddof=1) / nota.std(ddof=1) - 1) * 100:.1f}%)")

horas_raw = df_raw["horas_estudo_semana"].dropna()
print(f"\nHoras — DP original = {horas_raw.std(ddof=1):.3f} | DP limpo = {df['horas_estudo_semana'].std(ddof=1):.3f}")

"""2.3 VISUALIZAÇÃO UNIVARIADA"""

# Histograma + KDE das horas de estudo
plt.figure(figsize=(8, 4.5))
sns.histplot(df["horas_estudo_semana"], bins=20, kde=True, color="steelblue")
plt.axvline(df["horas_estudo_semana"].mean(), color="red", ls="--", label="Média")
plt.axvline(df["horas_estudo_semana"].median(), color="green", ls="--", label="Mediana")
plt.title("Distribuição das horas de estudo semanais")
plt.xlabel("Horas por semana"); plt.ylabel("Frequência")
plt.legend(); plt.show()

# Gráfico de barras: distribuição do nível de ensino
ordem_nivel = ["Extensão", "Graduação", "Pós-Graduação"]   # ordem assumida (ordinal)
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="nivel_ensino", order=ordem_nivel, hue="nivel_ensino",
                   palette="Blues", legend=False)
for c in ax.containers:
    ax.bar_label(c)   # escreve a contagem em cima de cada barra
plt.title("Alunos por nível de ensino"); plt.xlabel(""); plt.ylabel("Quantidade")
plt.show()
print((df["nivel_ensino"].value_counts(normalize=True) * 100).round(1).to_string())

# Boxplot da nota + cercas de Tukey (regra do IQR)
q1, q3 = nota.quantile([0.25, 0.75])
iqr = q3 - q1
lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1:.3f}  Q3={q3:.3f}  IQR={iqr:.3f}")
print(f"Cerca inferior = {lim_inf:.3f} | Cerca superior = {lim_sup:.3f}")

atipicos = df[(nota < lim_inf) | (nota > lim_sup)]
print(f"Atípicos pela regra do IQR: {len(atipicos)}")
print(atipicos[["id_aluno", "nivel_ensino", "regiao_campus", "nota_exame_final"]])

plt.figure(figsize=(8, 3.5))
sns.boxplot(x=nota, color="lightcoral")
plt.axvline(lim_inf, color="gray", ls=":"); plt.axvline(lim_sup, color="gray", ls=":")
plt.title("Boxplot da nota do exame final (cercas de Tukey tracejadas)")
plt.xlabel("Nota"); plt.show()

"""3.1 PROBABILIDADE CONDICIONAL E TEOREMA DE BAYES"""

n = len(df)
aprov = df["aprovado"] == "Sim"
pos = df["nivel_ensino"] == "Pós-Graduação"

# Probabilidades marginais
P_A = aprov.mean()          # P(Aprovado)
P_Pos = pos.mean()          # P(Pós-Graduação)

# Verossimilhança: P(Pós | Aprovado) = (Pós e Aprovado) / Aprovados
P_Pos_dado_A = (pos & aprov).sum() / aprov.sum()

# Teorema de Bayes: P(A | Pós) = P(Pós | A) * P(A) / P(Pós)
P_A_dado_Pos_bayes = P_Pos_dado_A * P_A / P_Pos

# Conferência: cálculo direto, dentro do grupo Pós-Graduação
P_A_dado_Pos_direto = aprov[pos].mean()

print(f"P(Aprovado)             = {P_A:.4f}")
print(f"P(Pós-Graduação)        = {P_Pos:.4f}")
print(f"P(Pós | Aprovado)       = {P_Pos_dado_A:.4f}")
print(f"P(Aprovado | Pós) Bayes = {P_A_dado_Pos_bayes:.4f}")
print(f"P(Aprovado | Pós) direto= {P_A_dado_Pos_direto:.4f}   <- deve ser igual ao de Bayes")

"""3.2 DISTRIBUIÇÃO BINOMIAL"""

# B(10, p): p = taxa histórica de aprovação da base
p = P_A
prob_7 = stats.binom.pmf(7, 10, p)   # P(X = 7)
print(f"p = {p:.4f}")
print(f"P(exatamente 7 aprovados em 10) = {prob_7:.4f} ({prob_7 * 100:.2f}%)")
print(f"Esperança = {10 * p:.2f} | Desvio padrão = {np.sqrt(10 * p * (1 - p)):.2f}")

# Gráfico da distribuição completa, destacando k = 7
k = np.arange(0, 11)
plt.figure(figsize=(7, 4))
plt.bar(k, stats.binom.pmf(k, 10, p), color=["tomato" if i == 7 else "steelblue" for i in k])
plt.title("Binomial B(10, p) — destaque para k = 7")
plt.xlabel("Número de aprovados"); plt.ylabel("Probabilidade"); plt.show()

"""3.3 DISTRIBUIÇÃO NORMAL E Z-SCORE"""

mu, sigma = nota.mean(), nota.std(ddof=1)

# Padronização (Z-score): Z = (X - mu) / sigma
df["nota_z"] = (nota - mu) / sigma
z_85 = (8.5 - mu) / sigma

# P(X > 8.5) teórica: função de sobrevivência = 1 - CDF
p_teorica = stats.norm.sf(z_85)
p_empirica = (nota > 8.5).mean()   # proporção observada, para comparar

print(f"mu = {mu:.3f} | sigma = {sigma:.3f}")
print(f"Z(8,5) = {z_85:.3f}")
print(f"P(nota > 8,5) teórica (Normal) = {p_teorica:.4f}")
print(f"P(nota > 8,5) empírica         = {p_empirica:.4f}")

# Histograma com a curva normal ajustada, para avaliar o pressuposto
x = np.linspace(nota.min(), 10.5, 300)
plt.figure(figsize=(8, 4))
sns.histplot(nota, stat="density", bins=20, color="lightsteelblue")
plt.plot(x, stats.norm.pdf(x, mu, sigma), color="red", label="Normal ajustada")
plt.axvline(8.5, color="black", ls="--", label="8,5")
plt.title("Nota final vs. curva Normal"); plt.legend(); plt.show()

"""4.1 Qui-quadrado de independência (`nivel_ensino` × `aprovado`)

"""

# Tabela de contingência (frequências observadas) com totais
tabela = pd.crosstab(df["nivel_ensino"], df["aprovado"])
print(pd.crosstab(df["nivel_ensino"], df["aprovado"], margins=True, margins_name="Total"))
print((pd.crosstab(df["nivel_ensino"], df["aprovado"], normalize="index") * 100).round(1))

# Teste Qui-quadrado
chi2, p_valor, gl, esperado = stats.chi2_contingency(tabela)
print(f"\nχ² = {chi2:.4f} | gl = {gl} | p-valor = {p_valor:.4f}")
print("Menor frequência esperada:", esperado.min().round(2), "(pressuposto: >= 5)")

# Cramér's V: força da associação (0 = nenhuma, 1 = perfeita)
n_total = tabela.values.sum()
v_cramer = np.sqrt(chi2 / (n_total * (min(tabela.shape) - 1)))
print(f"V de Cramér = {v_cramer:.4f}")

alfa = 0.05
print("\nDecisão:", "Rejeita H0 (há associação)" if p_valor < alfa else "Não rejeita H0 (sem evidência de associação)")

""" 4.2 Nota por campus e teste t (Barra Funda × EAD)"""

# Boxplots paralelos da nota por campus
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x="regiao_campus", y="nota_exame_final", hue="regiao_campus",
            palette="Set2", legend=False)
plt.title("Nota do exame final por campus"); plt.xlabel(""); plt.ylabel("Nota"); plt.show()
print(df.groupby("regiao_campus")["nota_exame_final"].agg(["count", "mean", "median", "std"]))

# Separa as duas amostras
bf = df.loc[df["regiao_campus"] == "Barra Funda", "nota_exame_final"]
ead = df.loc[df["regiao_campus"] == "EAD", "nota_exame_final"]

# Levene: as variâncias são iguais? (define qual versão do teste t usar)
lev_stat, lev_p = stats.levene(bf, ead)
print(f"Levene: p = {lev_p:.4f} ->", "variâncias iguais" if lev_p >= 0.05 else "variâncias diferentes")

# Teste t de Student (equal_var=True) e versão de Welch (equal_var=False)
t_stat, p_t = stats.ttest_ind(bf, ead, equal_var=(lev_p >= 0.05))
print(f"\nBarra Funda: média = {bf.mean():.3f} | EAD: média = {ead.mean():.3f}")
print(f"t = {t_stat:.4f} | p-valor = {p_t:.4f}")
print("Decisão (α = 0,05):", "diferença significativa" if p_t < 0.05 else "sem diferença significativa")

""" 4.3 Correlação: horas de estudo × nota"""

# Dispersão com reta de tendência
plt.figure(figsize=(7, 5))
sns.regplot(data=df, x="horas_estudo_semana", y="nota_exame_final",
            scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Horas de estudo × Nota do exame final")
plt.xlabel("Horas de estudo por semana"); plt.ylabel("Nota"); plt.show()

# Pearson (relação linear) e Spearman (relação monotônica, baseada em postos)
r, p_r = stats.pearsonr(df["horas_estudo_semana"], df["nota_exame_final"])
rho, p_rho = stats.spearmanr(df["horas_estudo_semana"], df["nota_exame_final"])
print(f"Pearson  r   = {r:.4f} (p = {p_r:.2e})")
print(f"Spearman rho = {rho:.4f} (p = {p_rho:.2e})")

""" 5.1 Regressão linear simples (OLS)"""

# Modelo: nota = b0 + b1 * horas + erro
modelo1 = smf.ols("nota_exame_final ~ horas_estudo_semana", data=df).fit()
print(modelo1.summary())

b0, b1 = modelo1.params["Intercept"], modelo1.params["horas_estudo_semana"]
print(f"\nEquação ajustada: nota = {b0:.4f} + {b1:.4f} * horas")
print(f"R² = {modelo1.rsquared:.4f} -> {modelo1.rsquared * 100:.1f}% da variação da nota é explicada pelas horas")

# Diagnóstico dos resíduos
resid = modelo1.resid
ajustados = modelo1.fittedvalues

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
sns.histplot(resid, kde=True, ax=axs[0], color="steelblue")           # normalidade (forma)
axs[0].set_title("Histograma dos resíduos")
sm.qqplot(resid, line="s", ax=axs[1])                                  # normalidade (Q-Q)
axs[1].set_title("Q-Q plot dos resíduos")
axs[2].scatter(ajustados, resid, alpha=0.5)                            # homocedasticidade
axs[2].axhline(0, color="red", ls="--")
axs[2].set_title("Resíduos × Valores ajustados")
axs[2].set_xlabel("Valor ajustado"); axs[2].set_ylabel("Resíduo")
plt.tight_layout(); plt.show()

# Testes formais
sh_stat, sh_p = stats.shapiro(resid)                                   # H0: resíduos normais
bp_stat, bp_p, _, _ = het_breuschpagan(resid, modelo1.model.exog)      # H0: variância constante
print(f"Shapiro-Wilk:    p = {sh_p:.4f} ->", "normais" if sh_p >= 0.05 else "NÃO normais")
print(f"Breusch-Pagan:   p = {bp_p:.4f} ->", "homocedástico" if bp_p >= 0.05 else "heterocedástico")

# Alerta de extrapolação: previsão para X = 50 horas
previsao_50 = modelo1.predict(pd.DataFrame({"horas_estudo_semana": [50]}))[0]
print(f"Faixa de horas observada: {df['horas_estudo_semana'].min():.1f} a {df['horas_estudo_semana'].max():.1f}")
print(f"Previsão para 50 h: nota = {previsao_50:.2f}")
print("-> Acima de 10 (impossível) e muito além dos dados observados: EXTRAPOLAÇÃO NÃO CONFIÁVEL.")

""" 5.2 Regressão linear múltipla"""

# Modelo múltiplo: adiciona a taxa de conclusão (%)
modelo2 = smf.ols("nota_exame_final ~ horas_estudo_semana + taxa_conclusao_pct", data=df).fit()
print(modelo2.summary())

# Comparação dos dois modelos
comp = pd.DataFrame({
    "R²": [modelo1.rsquared, modelo2.rsquared],
    "R² ajustado": [modelo1.rsquared_adj, modelo2.rsquared_adj],
    "AIC": [modelo1.aic, modelo2.aic],
    "b1 (horas)": [modelo1.params["horas_estudo_semana"], modelo2.params["horas_estudo_semana"]],
}, index=["Simples", "Múltiplo"])
print(comp)

# Multicolinearidade: correlação entre os preditores e VIF
print("Correlação horas × taxa de conclusão:",
      round(df["horas_estudo_semana"].corr(df["taxa_conclusao_pct"]), 4))
X = sm.add_constant(df[["horas_estudo_semana", "taxa_conclusao_pct"]])
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
                index=X.columns[1:], name="VIF")
print("\nVIF (acima de 5-10 indica colinearidade preocupante):")
print(vif.to_string())

""" 5.3 Intervalo de confiança de 95% para a média das notas"""

n = len(nota)
media = nota.mean()
ep = stats.sem(nota)                     # erro padrão = s / raiz(n)
t_crit = stats.t.ppf(0.975, df=n - 1)    # t crítico bilateral a 95%

ic_manual = (media - t_crit * ep, media + t_crit * ep)
ic_scipy = stats.t.interval(0.95, df=n - 1, loc=media, scale=ep)   # conferência

print(f"Média = {media:.4f} | Erro padrão = {ep:.4f} | t crítico = {t_crit:.4f}")
print(f"IC 95% (manual): [{ic_manual[0]:.4f}, {ic_manual[1]:.4f}]")
print(f"IC 95% (scipy) : [{ic_scipy[0]:.4f}, {ic_scipy[1]:.4f}]")


