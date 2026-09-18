## 📝 Storytelling — Análise de Experimentos de Machine Learning

### Efeito da limpeza

A análise dos 500 registros mostrou que a limpeza influenciou os resultados: a acurácia média passou de **86,37% para 85,14%**, e a temperatura média caiu de **73,81 °C para 71,53 °C**. Os outliers restantes foram mantidos, pois valores extremos não representam necessariamente erros.

### Hardware e aprovação no deploy

As medianas de acurácia foram semelhantes entre os hardwares. Sem informações de custos e tarefas equivalentes, não foi possível determinar o melhor custo-benefício.

A aprovação geral foi de **39,80%**, com as seguintes proporções por hardware:

| Hardware | Aprovação |
|---|---:|
| CPU Only | 44,29% |
| NVIDIA V100 | 39,49% |
| RTX 4090 | 39,13% |
| NVIDIA A100 | 38,89% |
| Não informado | 38,18% |

CPU Only apresentou a maior aprovação observada, mas isso não comprova superioridade: outras características dos experimentos podem influenciar o resultado.

### Acurácia e influência do preenchimento

Nas GPUs A100 e V100, a proporção de acurácias acima de 85% foi de **51,94% com preenchimento** e **49,82% considerando apenas valores observados válidos**. A diferença de **2,12 pontos percentuais** mostra que a imputação alterou essa probabilidade.

A correlação entre acurácia e loss foi de aproximadamente **−0,61**, indicando que maior acurácia tende a acompanhar menor loss.

### Falhas de temperatura e deploy

Registros com temperatura ausente ou corrompida tiveram aprovação de **42,50%**, contra **39,57%** nos demais. Não observamos menor aprovação associada a essas falhas, mas a comparação não demonstra causa e efeito nem mede o impacto da temperatura real.

### Conclusão

Validar os registros e justificar o tratamento é essencial antes de utilizar os dados em Machine Learning. O preenchimento preserva registros, mas pode alterar as conclusões.

Recomendamos melhorar a coleta de temperatura, padronizar a identificação dos hardwares e documentar a função de loss para investigar seus valores negativos. Também seria necessário coletar custos e comparar tarefas equivalentes para avaliar o custo-benefício dos equipamentos.