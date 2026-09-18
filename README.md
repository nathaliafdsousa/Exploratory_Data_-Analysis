# 📊 Análise Exploratória de Dados
Repositório acadêmico destinado às atividades e aos projetos desenvolvidos na disciplina de **Análise Exploratória de Dados** que apresenta étodos e ferramentas para coletar, preparar, explorar, visualizar e interpretar dados. O objetivo é transformar dados brutos em informações úteis para identificar padrões, 


## 🎯 Objetivos de aprendizagem

- Compreender as etapas de uma análise de dados;
- Importar, limpar e preparar conjuntos de dados;
- Aplicar medidas de estatística descritiva;
- Criar visualizações adequadas;
- Identificar relações entre variáveis;
- Comunicar resultados por meio de storytelling.

## 🛠️ Tecnologias utilizadas

- **Python:** desenvolvimento das análises;
- **Pandas:** tratamento e exploração dos dados;
- **NumPy:** operações numéricas;
- **Matplotlib:** criação e personalização de gráficos;
- **Seaborn:** visualizações estatísticas;




## 🖥️ Exercício — Monitoramento de Infraestrutura de TI

### Objetivo

Realizar uma Análise Exploratória de Dados no arquivo `monitoramento_ti_infra.csv` para compreender o desempenho dos servidores e identificar possíveis problemas de infraestrutura.

### Etapas realizadas

- Verificação de valores nulos, duplicados e inconsistentes;
- Cálculo de média, mediana, desvio padrão, mínimo e máximo;
- Criação de histograma para a latência;
- Criação de boxplots para CPU e memória por serviço;
- Análise da correlação entre CPU e latência com heatmap;
- Elaboração de um storytelling com os principais resultados.

### 📈 Principais resultados

- O conjunto possui **500 registros**, sem valores nulos ou duplicados;
- A latência média foi de **49,75 ms**;
- O uso médio de CPU foi de **49,98%**;
- O consumo médio de memória foi de **17,20 GB**;
- A maior latência foi de **90,65 ms**, registrada no `SRV-01`;
- Não foram identificados outliers nos boxplots de CPU e memória;
- A correlação entre CPU e latência foi de aproximadamente **-0,002**, indicando ausência de relação linear relevante;
- O `SRV-03` apresentou a maior taxa de erro, com **15,38%**.

### 📝 Storytelling

Os servidores apresentaram latência média de 49,75 ms e não foram identificados outliers estatísticos no uso de CPU ou memória. O maior pico de latência ocorreu no servidor SRV-01, alcançando 90,65 ms.

A correlação entre CPU e latência foi praticamente zero, mostrando que os aumentos de CPU não acompanharam os aumentos de latência. O principal ponto de atenção foi o servidor SRV-03, que apresentou a maior taxa de erro. Recomenda-se investigar seus registros e criar alertas para acompanhar a disponibilidade dos serviços.

## 🚀 Como executar

Instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## 🤖 Exercício — Análise de Experimentos de Machine Learning

### Objetivo

Realizar uma Análise Exploratória de Dados no arquivo `ml_models_dataset_sujo.csv`, investigando a qualidade dos registros, o desempenho por hardware e as probabilidades de aprovação para deploy.

### Etapas realizadas

- Inspeção da estrutura e dos valores ausentes;
- Tratamento de acurácias inválidas, taxas de aprendizado não positivas e temperaturas corrompidas;
- Preenchimento de valores ausentes com mediana, moda ou categoria;
- Comparação das estatísticas antes e depois da limpeza;
- Identificação de outliers pelo método IQR;
- Cálculo de probabilidades gerais e condicionais;
- Criação de histogramas, boxplots, gráfico de probabilidades e mapas de calor.

### 📝 Análise dos resultados

O storytelling completo, com a interpretação dos resultados e as limitações da análise, está disponível no notebook dentro da pasta do código.
