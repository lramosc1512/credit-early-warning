# Credit Early Warning

Português | [English](README.en.md)

Modelo de alerta de inadimplência em cartão de crédito. Pipeline completo de Machine Learning para prever se um cliente vai deixar de pagar a fatura no mês seguinte. Desenvolvido como projeto final da disciplina de Engenharia de Aprendizado de Máquina da pós-graduação em Engenharia de IA da UniCEUB (Brasília).

**[Abrir o notebook no Google Colab](https://colab.research.google.com/drive/1-i_1RuEuG3neY7lfPizUtB9SF5eduMdq?usp=sharing)**

## Problema

Classificação binária: a partir do perfil do cliente e dos últimos 6 meses de histórico de pagamento, prever se ele fica inadimplente no mês seguinte (1 = inadimplente, 0 = adimplente).

Metas mínimas definidas pela disciplina:

| Métrica | Meta |
|---|---|
| AUC-ROC | ≥ 0,75 |
| Recall | ≥ 0,60 |
| F1-Score | ≥ 0,65 |

## Dataset

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), UCI Machine Learning Repository (id=350), licença CC BY 4.0.

- 30.000 clientes, 23 atributos
- Classes desbalanceadas: 77,9% adimplentes, 22,1% inadimplentes
- Carregado direto com `ucimlrepo`, sem arquivos locais nem autenticação

## Pipeline

1. **Carregamento** com `ucimlrepo` e renomeação das colunas genéricas (`X1`...`X23`) usando os metadados do próprio dataset
2. **EDA**: distribuições, correlações, taxa de inadimplência por grupo, outliers
3. **Limpeza**: códigos fora do dicionário oficial em `EDUCATION` (0, 5, 6) e `MARRIAGE` (0) agrupados na categoria "outros" já existente
4. **Engenharia de atributos**: 4 variáveis derivadas
   - `utilizacao_credito`: fatura mais recente / limite de crédito
   - `media_atraso`: média dos 6 códigos de status de pagamento
   - `meses_em_atraso`: quantidade de meses com atraso
   - `proporcao_pagamento`: total pago / total faturado
5. **Separação**: 70/15/15 estratificado (treino/validação/teste), `StandardScaler` ajustado só no treino
6. **Modelagem**: Regressão Logística, Árvore de Decisão, Random Forest, SVM, e depois Gradient Boosting, SMOTE e XGBoost
7. **Ajuste**: `RandomizedSearchCV` otimizando F1, mais busca do threshold de decisão (maximizar F1 com Recall ≥ 0,60)
8. **Explicabilidade** com SHAP (TreeExplainer)
9. **Demonstração** com clientes sintéticos gerados pelo `faker`
10. **Proposta de monitoramento** com função de PSI (Population Stability Index) para detecção de drift

## Resultados (conjunto de validação)

Modelos base, com hiperparâmetros padrão:

| Modelo | AUC-ROC | F1 | Recall | Precision |
|---|---|---|---|---|
| Random Forest | 0,758 | 0,463 | 0,363 | 0,639 |
| Regressão Logística | 0,748 | 0,397 | 0,282 | 0,666 |
| SVM | 0,724 | 0,444 | 0,337 | 0,652 |
| Árvore de Decisão | 0,591 | 0,364 | 0,369 | 0,359 |

Evolução das técnicas testadas:

| Etapa | AUC-ROC | F1 | Recall |
|---|---|---|---|
| Random Forest base | 0,758 | 0,463 | 0,363 |
| + `class_weight="balanced"` | 0,755 | 0,432 | 0,327 |
| + ajuste de hiperparâmetros | 0,778 | 0,539 | 0,579 |
| + ajuste de threshold | 0,778 | 0,535 | 0,606 |
| **Gradient Boosting + threshold (final)** | **0,782** | **0,543** | **0,619** |
| Gradient Boosting + SMOTE + threshold | 0,772 | 0,530 | 0,603 |
| XGBoost + threshold | 0,756 | 0,507 | 0,610 |

**Modelo final:** Gradient Boosting com as variáveis derivadas e threshold de decisão em 0,2325.
AUC-ROC e Recall atingiram as metas. O F1 (0,543) ficou abaixo de 0,65.

### Por que o F1 não bateu a meta

Para chegar a F1 = 0,65 com Recall = 0,60, a Precision precisaria ficar em torno de 0,71. O modelo final chegou a 0,484. Balanceamento de classes, ajuste de hiperparâmetros, variáveis novas, boosting, SMOTE e XGBoost deixaram o F1 sempre entre 0,50 e 0,54, o que aponta para sobreposição entre as classes nos atributos disponíveis, e não para uma escolha de modelagem que se resolveria com mais ajuste.

## Explicabilidade (SHAP)

`PAY_0` (status de pagamento mais recente) é o atributo com mais peso. Duas das variáveis criadas no projeto ficaram em 2º (`meses_em_atraso`) e 4º lugar (`utilizacao_credito`). Limites de crédito mais altos puxam a previsão para menor risco.

<p align="center">
  <img src="images/shap_importancia.png" width="45%">
  <img src="images/shap_direcao_impacto.png" width="45%">
</p>

Explicação de uma previsão individual:

<p align="center">
  <img src="images/shap_waterfall.png" width="70%">
</p>

## Proposta de monitoramento

- Acompanhar AUC-ROC, F1 e Recall conforme chegarem os rótulos (pagou ou não pagou)
- Detectar drift com PSI nas variáveis principais (`PAY_0`, `meses_em_atraso`, `utilizacao_credito`)
- Retreinar quando o PSI passar de 0,25, o AUC-ROC cair mais de 0,05 em relação à validação, ou a cada trimestre
- Versionar modelo e dados de treino, e registrar cada previsão com data e hora

## Limitações e próximos passos

- Todas as métricas vêm do conjunto de validação, que também foi usado para escolher o modelo e o threshold, então tendem a estar um pouco otimistas. O conjunto de teste foi separado e normalizado, mas não foi usado numa avaliação final. Próximo passo: avaliar o modelo final uma única vez em `X_test`.
- Meta de F1 não atingida (ver acima). Caminhos possíveis: aprendizado sensível a custo, stacking, ou dados externos como histórico em birôs de crédito.
- A demonstração sintética sorteia cada variável de forma independente e uniforme, então os clientes gerados não seguem a distribuição conjunta dos dados originais.

## Como executar

**Colab (recomendado):** abra o link no topo e execute todas as células. A instalação das bibliotecas e o download dos dados acontecem automaticamente.

**Local:**

```bash
git clone https://github.com/lramosc1512/credit-early-warning.git
cd credit-early-warning
pip install -r requirements.txt
jupyter notebook previsao_inadimplencia_cartao_credito.ipynb
```

A célula do `RandomizedSearchCV` pode levar de 5 a 15 minutos, dependendo da máquina.

## Tecnologias

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, Matplotlib, seaborn, Faker, ucimlrepo

## Autor

Leonardo Ramos Coutinho
[LinkedIn](https://www.linkedin.com/in/leonardorcoutinho)
