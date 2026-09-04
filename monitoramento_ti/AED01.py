
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('monitoramento_ti_infra.csv')
df.info()

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') # Converte a coluna data de texto em datetime

# Garante que as métricas sejam numéricas
colunas_numericas = [
    "latency_ms",
    "cpu_usage_pct",
    "memory_usage_gb"
]

for coluna in colunas_numericas:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

# Transforma valores infinitos em nulos
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.head()

# Limpeza de dados
df.isna().sum() # Verificando valores nulos

# Verificando Registros duplicados

registros_duplicados = df.duplicated().sum()
print(f"Número de registros duplicados: {registros_duplicados}")

# Verificando valores incosistentes
inconsistentes = df[
    (df["latency_ms"] < 0)
    | (~df["cpu_usage_pct"].between(0, 100))
    | (df["memory_usage_gb"] < 0)
]

print("Registros inconsistentes:", len(inconsistentes))

# Análise Estatísca Descritiva
colunas = [
    'latency_ms',
    'cpu_usage_pct',
    'memory_usage_gb'
]

df[colunas].describe().round(2)

# Histograma da latência

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="latency_ms",
    bins=20,
    kde=True,
    color="royalblue"
)

plt.title("Distribuição da latência dos servidores")
plt.xlabel("Latência (ms)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# Boxplot de CPU e memória

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=df,
    x="metric_type",
    y="cpu_usage_pct",
    ax=axes[0]
)

axes[0].set_title("Uso de CPU por tipo de serviço")
axes[0].set_xlabel("Tipo de serviço")
axes[0].set_ylabel("CPU (%)")

sns.boxplot(
    data=df,
    x="metric_type",
    y="memory_usage_gb",
    ax=axes[1]
)

axes[1].set_title("Uso de memória por tipo de serviço")
axes[1].set_xlabel("Tipo de serviço")
axes[1].set_ylabel("Memória (GB)")

plt.tight_layout()
plt.show()

# Correlação entre CPU e latência

correlacao = df[
    ["cpu_usage_pct", "latency_ms"]
].corr()

plt.figure(figsize=(6, 4))

sns.heatmap(
    correlacao,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlação entre CPU e latência")
plt.tight_layout()
plt.show()

print(correlacao)

