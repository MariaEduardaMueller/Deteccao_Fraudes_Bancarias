# Deteccao_Fraudes_Bancarias
# Detecção de Anomalias em Transações com Python

Projeto desenvolvido durante o curso da DIO: **detecção de anomalias em transações em Python**.

O objetivo é desenvolver modelos capazes de identificar transações fraudulentas em um conjunto de dados altamente desbalanceado, explorando diferentes algoritmos, técnicas de tratamento do desbalanceamento e métricas de avaliação.

### Tecnologias e ferramentas utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib
- Jupyter Notebook
- Google Colab

<br>

## Dataset

O projeto utiliza um dataset de transações financeiras contendo **284.807 registros e 31 colunas**. Para conferir o dataset: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

A variável `Class` representa o resultado da transação:
- `0` → transação normal
- `1` → transação fraudulenta

O conjunto apresenta um forte desbalanceamento, com aproximadamente 0,17% de transações fraudulentas.

Os dados foram divididos em:

- 70% para treinamento
- 30% para teste

A divisão foi realizada utilizando `stratify` para preservar a proporção das classes.

<br>

## Modelos utilizados

Foram testados os seguintes modelos:

### Regressão Logística

Utilizada com `class_weight="balanced"` e `StandardScaler` dentro de um Pipeline.

### Random Forest

Configurado com balanceamento de classes através de `class_weight="balanced"`.

### XGBoost

Utilizado para classificação das transações, com análise de importância das variáveis.

### XGBoost + SMOTE

Foi aplicada a técnica **SMOTE** somente nos dados de treinamento para aumentar a representação da classe minoritária.

### GridSearchCV

Foi utilizado para buscar diferentes combinações de hiperparâmetros do XGBoost, utilizando **Recall** como métrica de otimização.

<br>
<br>


## Resultados

Os modelos apresentaram os seguintes resultados no conjunto de teste:

| Modelo | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0655 | 0.8784 | 0.1219 | 0.9678 |
| Random Forest | 0.8309 | 0.7635 | 0.7958 | 0.9474 |
| XGBoost | 0.9268 | 0.7703 | 0.8413 | 0.9751 |
| XGBoost GridSearch | 0.8359 | 0.7230 | 0.7754 | 0.9134 |

### XGBoost

O XGBoost apresentou:

- **Precision:** 92,68%
- **Recall:** 77,03%
- **F1-Score:** 84,13%
- **AUC:** 97,51%

Esses resultados mostram a capacidade do modelo de identificar uma parcela significativa das transações fraudulentas mantendo uma precisão elevada.

### Regressão Logística

A Regressão Logística apresentou **Recall de 87,84%**, mas Precision de apenas **6,55%**, indicando uma quantidade maior de falsos positivos.

Esse resultado demonstra a importância de analisar diferentes métricas em problemas de detecção de fraude, em vez de considerar apenas a acurácia.

## Tratamento do desbalanceamento

Como a quantidade de transações fraudulentas é muito menor que a de transações normais, foram exploradas técnicas para lidar com o desbalanceamento.

Foi utilizado **SMOTE** no conjunto de treinamento:

```text
Antes do SMOTE:
Classe 0: 199.020
Classe 1: 344

Depois do SMOTE:
Classe 0: 199.020
Classe 1: 199.020
