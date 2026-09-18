import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ml_experiment/ml_models_dataset_sujo.csv',sep=';')

# Visualizando as 5 primeiras linhas do dataset
print(df.head())

# Visualizando quantidade de registros, tipos dos dados e quantidade de valores preenchidos
print(df.info())

# Estatísticas de valores númericos
print(df.describe().T)

# Estatística de objetos
print(df.describe(include='object').T)

qt_nulos = df.isna().sum()
print(qt_nulos)

proporcao_nulos = df.isna().sum() / len(df) * 100
proporcao_nulos.round(2)

# Fazer uma cópia do dataset

df_limpo = df.copy()

# Calculando acurácias inválidas

acuracias_invalidas = (df_limpo['accuracy'] < 0) | (df_limpo['accuracy'] > 1)
acuracias_invalidas.sum()
print(f'Acurácias inválidas: {acuracias_invalidas.sum()}')

# Substituir acurácias inválidas se tiver valores inválidos por valores ausentes
df_limpo.loc[acuracias_invalidas, 'accuracy'] = np.nan

# Tratar taxas de aprendizado negativas ou iguais a zero
taxas_invalidas = (df_limpo['learning_rate'] <= 0)
taxas_invalidas.sum()
print(f'Taxas inválidas: {taxas_invalidas.sum()}')

# Subsstituindo valores inválidos
df_limpo.loc[taxas_invalidas, 'learning_rate'] = np.nan

# Identificar temprraturas corrompidas
temperaturas_invalidas = (df_limpo['gpu_temp_c'] >= 150)
temperaturas_invalidas.sum()
print(f'Temperaturas inválidas: {temperaturas_invalidas.sum()}')

# Substituir valores inválidos por valores ausentes
df_limpo.loc[temperaturas_invalidas, "gpu_temp_c"] = np.nan

# Conferindo a categoria hardware
df_limpo["hardware_type"].value_counts(dropna=False)

# Padronizar 'Nan' e 'Unknown' para Não informado
df_limpo["hardware_type"] = df_limpo["hardware_type"].replace(
    "Unknown", "Não informado"
)

df_limpo["hardware_type"] = df_limpo["hardware_type"].fillna(
    "Não informado"
)

df_limpo["hardware_type"].value_counts(dropna=False)

# Verificando valores nulos após as mudanças

print(f'Valores nulos: {df_limpo.isna().sum()}')

# Escolhemos a mediana porque tempos de treinamento muito elevados,podem influenciar a média. A mediana é menos sensível a esses extremos.
mediana_tempo = df_limpo["training_hours"].median()
print(f'Mediana: {mediana_tempo}')

# Preenchemos os valores ausentes mantendo as demais informações da linha.
df_limpo["training_hours"] = df_limpo["training_hours"].fillna(mediana_tempo)

# As acurácias inválidas já foram substituídas por NaN.Usamos a mediana das acurácias válidas como estimativa central.
mediana_accuracy = df_limpo["accuracy"].median()
print(f'Mediana: {mediana_accuracy}')
# Esse preenchimento preserva os registros, mas pode alterar a proporção de acurácias acima de 85%. Avaliaremos essa limitação nas probabilidades.
df_limpo["accuracy"] = df_limpo["accuracy"].fillna(mediana_accuracy)

# Usamos a mediana porque ela é menos influenciada por valores extremos. Os valores negativos existentes ainda precisam ser investigados:
# preencher os nulos não resolve outras possíveis inconsistências.
mediana_loss = df_limpo["loss"].median()
print(f'Mediana {mediana_loss}')
df_limpo["loss"] = df_limpo["loss"].fillna(mediana_loss)

# As leituras corrompidas identificadas já foram substituídas por NaN. Utilizamos a mediana das temperaturas restantes como estimativa.
mediana_temperatura = df_limpo["gpu_temp_c"].median()
print(f'Mediana {mediana_temperatura}')
# A mediana geral é uma simplificação, pois não considera diferenças entre hardwares. O preenchimento também não comprova a existência
# de GPU nos registros classificados como "CPU Only".
df_limpo["gpu_temp_c"] = df_limpo["gpu_temp_c"].fillna(mediana_temperatura)

# Escolhemos a moda para preencher com o tamanho mais frequente. Assim, utilizamos um tamanho presente nos registros, sem criar um valor intermediário que poderia surgir pelo uso da média.
# Se tiver empate entre modas, selecionamos a primeira.
moda_tamanho = df_limpo["dataset_size_gb"].mode().iloc[0]
print(f'Moda {moda_tamanho}')
df_limpo["dataset_size_gb"] = df_limpo["dataset_size_gb"].fillna(moda_tamanho)

# As taxas inválidas já foram substituídas por NaN. Escolhemos a moda para utilizar a configuração válida mais frequente,evitando criar uma taxa intermediária pela média dos valores.
# Se tiver empate entre modas, selecionamos a primeira.
moda_taxa = df_limpo["learning_rate"].mode().iloc[0]
print(f'Moda: {moda_taxa}')
df_limpo["learning_rate"] = df_limpo["learning_rate"].fillna(moda_taxa)

# Conferimos se ainda existem valores ausentes após o preenchimento. se tiver ainda será necessário investigar inconsistências e outliers.
print(f'Valores nulos {df_limpo.isna().sum()}')

# Selecionamos as métricas numéricas que serão analisadas.
colunas_numericas = [
    "training_hours",
    "accuracy",
    "loss",
    "dataset_size_gb",
    "learning_rate",
    "gpu_temp_c"
]
print(colunas_numericas)

# Calculamos média, mediana e desvio padrão na tabela original. Dataset sujo:
estatisticas_antes = df[colunas_numericas].agg(
    ["mean", "median", "std"]
).T

# O IQR mede a amplitude dos 50% centrais dos dados.
estatisticas_antes["IQR"] = (
    df[colunas_numericas].quantile(0.75)
    - df[colunas_numericas].quantile(0.25)
)

estatisticas_antes.round(4)

# Repetimos os cálculos no dataset limpo, como a correção de valores e o preenchimento afetaram os dados.
estatisticas_depois = df_limpo[colunas_numericas].agg(
    ["mean", "median", "std"]
).T

estatisticas_depois["IQR"] = (
    df_limpo[colunas_numericas].quantile(0.75)
    - df_limpo[colunas_numericas].quantile(0.25)
)

estatisticas_depois.round(4)

# Calculamos os quartis de cada coluna na tabela tratada.
Q1 = df_limpo[colunas_numericas].quantile(0.25)
Q3 = df_limpo[colunas_numericas].quantile(0.75)

IQR = Q3 - Q1

# A regra de 1,5 vezes o IQR identifica possíveis valores extremos. Esses limites são estatísticos, não regras de validade dos dados.
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Marcamos como True os valores abaixo ou acima dos limites.
outliers = (
    (df_limpo[colunas_numericas] < limite_inferior)
    | (df_limpo[colunas_numericas] > limite_superior)
)

# Resumimos os limites e a quantidade de valores sinalizados por coluna.
resumo_outliers = pd.DataFrame({
    "Limite inferior": limite_inferior,
    "Limite superior": limite_superior,
    "Quantidade": outliers.sum(),
    "Percentual": outliers.sum() / len(df_limpo) * 100
})

print(resumo_outliers.round(4))

# Exibimos os experimentos com tempos sinalizados pelo IQR, para examinar os valores e o hardware utilizado.
df_limpo.loc[
    outliers["training_hours"],
    ["model_id", "training_hours", "hardware_type"]
]

# Valores negativos de loss precisam de uma investigação específica.Sem saber qual função de perda foi usada, não podemos afirmar que todo valor negativo é necessariamente inválido.
df_limpo.loc[
    df_limpo["loss"] < 0,
    ["model_id", "accuracy", "loss"]
]

# Conferimos os nomes e as quantidades dos status.Isso evita usar um nome diferente do registrado no dataset.
df_limpo["deployment_status"].value_counts(dropna=False)

# Criamos uma coluna que identifica os modelos aprovados. True representa aprovado; False representa os demais status.
df_limpo["aprovado"] = df_limpo["deployment_status"] == "Approved"

# Somamos os aprovados e dividimos pelo total de registros.
probabilidade_geral = df_limpo["aprovado"].sum() / len(df_limpo)

print(f"Probabilidade geral de aprovação: {probabilidade_geral:.2%}")

# Agrupamos os registros por hardware e contamos os aprovados.
aprovados_por_hardware = df_limpo.groupby("hardware_type")["aprovado"].sum()

# Contamos o total de experimentos de cada hardware.
total_por_hardware = df_limpo.groupby("hardware_type").size()

# Cada quantidade de aprovados é dividida pelo total do mesmo hardware.
probabilidade_por_hardware = aprovados_por_hardware / total_por_hardware

# Apresentamos as contagens para mostrar a base de cada probabilidade.
tabela_probabilidades = pd.DataFrame({
    "Total de modelos": total_por_hardware,
    "Modelos aprovados": aprovados_por_hardware,
    "Aprovação (%)": probabilidade_por_hardware * 100
})

print(tabela_probabilidades.round(2))

# Ordenamos as probabilidades para facilitar a comparação visual.
probabilidades_ordenadas = probabilidade_por_hardware.sort_values(
    ascending=False
) * 100

# Criamos um gráfico exclusivo para a aprovação por hardware.
probabilidades_ordenadas.plot(
    kind="bar",
    figsize=(9, 5),
    color="steelblue"
)

plt.title("Probabilidade de aprovação no deploy por hardware")
plt.xlabel("Hardware")
plt.ylabel("Aprovação (%)")

# Mantemos a escala de 0% a 100% para representar as probabilidades.
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# Selecionamos os modelos que utilizaram A100 ou V100. isin() verifica se o valor pertence à lista informada.
gpu_alto_desempenho = df_limpo["hardware_type"].isin(
    ["NVIDIA A100", "NVIDIA V100"]
)

modelos_gpu = df_limpo.loc[gpu_alto_desempenho]

# Contamos os modelos com acurácia estritamente superior a 85%.
quantidade_acima_85 = (modelos_gpu["accuracy"] > 0.85).sum()

# O denominador inclui somente os modelos com A100 ou V100.
probabilidade_accuracy = quantidade_acima_85 / len(modelos_gpu)

print("Total de modelos com A100 ou V100:", len(modelos_gpu))
print("Quantidade com acurácia acima de 85%:", quantidade_acima_85)
print(f"Probabilidade: {probabilidade_accuracy:.2%}")

# Selecionamos acurácias observadas e válidas na tabela original.As comparações excluem também os valores ausentes.
accuracy_original_valida = (
    (df["accuracy"] >= 0)
    & (df["accuracy"] <= 1)
)

# Combinamos duas condições: hardware selecionado E acurácia válida.
modelos_gpu_observados = df_limpo.loc[
    gpu_alto_desempenho & accuracy_original_valida
]

probabilidade_observada = (
    (modelos_gpu_observados["accuracy"] > 0.85).sum()
    / len(modelos_gpu_observados)
)

print("Modelos com acurácia observada válida:", len(modelos_gpu_observados))
print(f"Com preenchimento: {probabilidade_accuracy:.2%}")
print(f"Somente acurácias observadas: {probabilidade_observada:.2%}")

# Analisamos a distribuição das acurácias após a limpeza.bins=20 divide os valores em 20 intervalos.
plt.figure(figsize=(8, 4))

sns.histplot(data=df_limpo, x="accuracy", bins=20)

plt.title("Distribuição da acurácia após a limpeza")
plt.xlabel("Acurácia")
plt.ylabel("Quantidade de modelos")
plt.tight_layout()
plt.show()

# Verificamos se os tempos se concentram em determinadas faixas e se existem treinamentos muito mais longos que os demais.
plt.figure(figsize=(8, 4))

sns.histplot(data=df_limpo, x="training_hours", bins=20)

plt.title("Distribuição do tempo de treinamento")
plt.xlabel("Tempo de treinamento (horas)")
plt.ylabel("Quantidade de modelos")
plt.tight_layout()
plt.show()

# Comparamos a acurácia entre hardwares, observando mediana, dispersão e possíveis outliers dentro de cada grupo.
plt.figure(figsize=(10, 5))

sns.boxplot(data=df_limpo, x="hardware_type", y="accuracy")

plt.title("Acurácia por hardware")
plt.xlabel("Hardware")
plt.ylabel("Acurácia")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# Comparamos a distribuição dos tempos de treinamento por hardware. Diferenças podem envolver também o tamanho dos dados e a tarefa executada.
plt.figure(figsize=(10, 5))

sns.boxplot(data=df_limpo, x="hardware_type", y="training_hours")

plt.title("Tempo de treinamento por hardware")
plt.xlabel("Hardware")
plt.ylabel("Tempo de treinamento (horas)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# Selecionamos somente as métricas numéricas.Usamos Pearson para investigar associações lineares entre elas.
correlacao = df_limpo[colunas_numericas].corr(method="pearson")

plt.figure(figsize=(9, 6))

# annot=True mostra os números; fmt=".2f" usa duas casas decimais.
# A escala fixa de -1 a 1 facilita interpretar as cores.
sns.heatmap(
    correlacao,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Correlação entre as métricas após a limpeza")
plt.tight_layout()
plt.show()

# Criamos uma tabela de indicadores a partir dos nulos originais. True indica uma ocorrência de problema no registro.
mascara_ruido = df.isna()

# Incluímos as acurácias fora do intervalo válido.
mascara_ruido["accuracy"] = (
    mascara_ruido["accuracy"]
    | (df["accuracy"] < 0)
    | (df["accuracy"] > 1)
)

# Incluímos taxas de aprendizado inválidas segundo o critério adotado.
mascara_ruido["learning_rate"] = (
    mascara_ruido["learning_rate"]
    | (df["learning_rate"] <= 0)
)

# Incluímos as temperaturas corrompidas descritas no enunciado.
mascara_ruido["gpu_temp_c"] = (
    mascara_ruido["gpu_temp_c"]
    | (df["gpu_temp_c"] >= 150)
)

# Unknown também representa falta de identificação do hardware.
mascara_ruido["hardware_type"] = (
    mascara_ruido["hardware_type"]
    | (df["hardware_type"] == "Unknown")
)

# Calculamos a porcentagem de problemas em cada coluna por hardware. A média de True/False equivale à proporção de True.
# Usamos os grupos de hardware padronizados, mantendo os índices originais.
ruido_por_hardware = (
    mascara_ruido.groupby(df_limpo["hardware_type"]).mean() * 100
)

plt.figure(figsize=(12, 5))

sns.heatmap(
    ruido_por_hardware,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "Registros com problema (%)"}
)

plt.title("Dados ausentes ou corrompidos por hardware")
plt.xlabel("Coluna")
plt.ylabel("Hardware")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Usamos a identificação das falhas originais, anterior ao preenchimento.
# O grupo True reúne temperaturas ausentes ou corrompidas.
# O grupo False reúne registros sem esses problemas detectados.
resumo_temperatura = df_limpo.groupby(
    mascara_ruido["gpu_temp_c"]
)["aprovado"].agg(["size", "sum", "mean"])

# Renomeamos as colunas para facilitar a interpretação.
resumo_temperatura.columns = [
    "Total de modelos",
    "Modelos aprovados",
    "Aprovação (%)"
]

# Transformamos a proporção de aprovados em porcentagem.
resumo_temperatura["Aprovação (%)"] *= 100

resumo_temperatura.index.name = "Temperatura ausente ou corrompida"

resumo_temperatura.round(2)
