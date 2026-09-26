# Deteccao_Fraudes_Bancarias

# Detecção de Anomalias em Transações com Python

Projeto desenvolvido durante o curso da DIO **Detecção de Anomalias em Transações em Python** do bootcamp Bradesco - GenAI, Dados & Cyber

O objetivo é desenvolver modelos capazes de identificar transações fraudulentas em um conjunto de dados altamente desbalanceado, explorando diferentes algoritmos de classificação, técnicas de tratamento do desbalanceamento, ajuste de limiar de decisão e métodos de interpretabilidade.

Em problemas de detecção de fraude, o **Recall da classe fraudulenta** é uma métrica especialmente importante, pois representa a proporção das fraudes reais que foram identificadas pelo modelo.

---

## Tecnologias e ferramentas utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* SHAP
* Matplotlib
* Jupyter Notebook
* Google Colab

---

## Dataset

O projeto utiliza o dataset de transações de cartão de crédito disponibilizado pelo TensorFlow.

**Dataset:**
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

O conjunto possui:

* **284.807 registros**
* **31 colunas**
* 30 variáveis de entrada
* 1 variável de classificação (`Class`)

A variável `Class` representa o resultado da transação:

* `0` → transação normal
* `1` → transação fraudulenta

O dataset apresenta um forte desbalanceamento, com aproximadamente **0,17% de transações fraudulentas**.

Por esse motivo, a análise não considera apenas a acurácia. São utilizadas métricas como Precision, Recall, F1-Score, ROC-AUC e Precision-Recall.

<img width="1813" height="595" alt="image" src="https://github.com/user-attachments/assets/437d4a67-810f-4c32-9a39-b3efb13ffd68" />

---

## Análise exploratória

Inicialmente foram analisados:

* dimensões do dataset;
* tipos das variáveis;
* estatísticas descritivas;
* distribuição da variável `Class`;
* proporção de transações normais e fraudulentas;
* visualização da distribuição das classes.

A análise evidencia o forte desbalanceamento existente entre as classes.

Esse cenário torna inadequado utilizar somente a acurácia como critério de avaliação, já que um modelo poderia classificar grande parte das transações como normais e ainda obter uma acurácia elevada, mesmo deixando de identificar diversas fraudes.

---

## Pré-processamento

Foi criada uma nova variável a partir de `Amount`:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

A transformação logarítmica é utilizada para reduzir a assimetria da variável de valor da transação.

As variáveis são então separadas entre:

* `X` → variáveis preditoras;
* `y` → variável alvo (`Class`).

Os dados foram divididos em:

* **70% para treinamento**
* **30% para teste**

A divisão utiliza `stratify=y`, preservando a proporção das classes nos dois conjuntos.

O conjunto de teste mantém sua distribuição original e é utilizado para avaliar os modelos.

---

# Modelos utilizados

## Regressão Logística

A Regressão Logística foi utilizada como modelo baseline.

Foi utilizado:

* `StandardScaler`;
* `Pipeline`;
* `class_weight="balanced"`.

O modelo apresentou:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |    0,0655 |
| Recall    |    0,8784 |
| F1-Score  |    0,1219 |
| ROC-AUC   |    0,9678 |

O modelo apresentou **Recall elevado para a classe fraude**, identificando grande parte das transações fraudulentas. Porém, a Precision foi baixa, indicando uma quantidade maior de falsos positivos.

Esse resultado evidencia o trade-off existente entre identificar mais fraudes e evitar classificar transações legítimas como fraudulentas.

---

## Random Forest

Foi utilizado um Random Forest com:

* `n_estimators=100`;
* `max_depth=10`;
* `class_weight="balanced"`.

Resultado obtido no conjunto de teste:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |    0,8309 |
| Recall    |    0,7635 |
| F1-Score  |    0,7958 |
| ROC-AUC   |    0,9474 |

O Random Forest também foi utilizado para avaliar diferentes estratégias de tratamento do desbalanceamento.

---

## XGBoost

O XGBoost foi utilizado como um dos principais modelos de classificação.

Configuração inicial:

* `n_estimators=100`;
* `max_depth=5`;
* `scale_pos_weight=10`;
* `eval_metric="logloss"`.

Resultado obtido no conjunto de teste:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |    0,9268 |
| Recall    |    0,7703 |
| F1-Score  |    0,8413 |
| ROC-AUC   |    0,9751 |

Além das métricas de classificação, foram analisadas as importâncias das variáveis e utilizadas técnicas de interpretabilidade com SHAP.

<img width="547" height="436" alt="image" src="https://github.com/user-attachments/assets/9629fa4b-be5e-4fd7-b535-7f28a506fe62" />


<img width="777" height="470" alt="image" src="https://github.com/user-attachments/assets/e6e21fc8-e41f-4c52-b72c-bc6c25178634" />

---

## GridSearchCV

Também foi utilizado `GridSearchCV` para testar diferentes combinações de hiperparâmetros do XGBoost.

A busca utilizou **Recall como métrica de otimização** durante a validação cruzada.

O melhor conjunto de parâmetros encontrado foi:

```text
max_depth = 5
n_estimators = 100
```

Resultado no conjunto de teste:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |    0,8359 |
| Recall    |    0,7230 |
| F1-Score  |    0,7754 |
| ROC-AUC   |    0,9134 |

O resultado demonstra que a configuração selecionada durante a validação cruzada não necessariamente apresenta o maior Recall no conjunto de teste. Por isso, os modelos são analisados utilizando um conjunto de métricas, e não apenas o resultado do GridSearchCV.

---

# Tratamento do desbalanceamento

Como a classe fraudulenta representa uma parcela muito pequena das transações, foram exploradas diferentes estratégias para lidar com o desbalanceamento.

## Undersampling

No undersampling, parte das transações da classe majoritária é removida para aproximar a quantidade de exemplos das duas classes.

A técnica é aplicada **somente ao conjunto de treinamento**.

O conjunto de teste permanece com sua distribuição original.

O Random Forest é então treinado utilizando o conjunto de treinamento balanceado por undersampling e posteriormente avaliado no conjunto de teste original.

---

## SMOTE

Também foi utilizado o **SMOTE (Synthetic Minority Over-sampling Technique)**.

O SMOTE cria exemplos sintéticos da classe minoritária para aumentar sua representação durante o treinamento.

No projeto, o SMOTE é aplicado somente aos dados de treinamento:

```python
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
```

A avaliação continua sendo realizada sobre o conjunto de teste original.

Essa separação evita utilizar informações do conjunto de teste durante o processo de treinamento.

---

# Comparação dos modelos

Os principais modelos foram comparados utilizando:

* Precision;
* Recall;
* F1-Score;
* ROC-AUC.

| Modelo              | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | --------: | -----: | -------: | ------: |
| Logistic Regression |    0,0655 | 0,8784 |   0,1219 |  0,9678 |
| Random Forest       |    0,8309 | 0,7635 |   0,7958 |  0,9474 |
| XGBoost             |    0,9268 | 0,7703 |   0,8413 |  0,9751 |
| XGBoost GridSearch  |    0,8359 | 0,7230 |   0,7754 |  0,9134 |

Além desses modelos, o notebook apresenta uma comparação específica entre Random Forest com:

* undersampling;
* SMOTE;
* `class_weight="balanced"`.

Os resultados dessas estratégias são calculados diretamente no notebook.

---

# Curva ROC

Foram construídas curvas ROC para comparar o comportamento dos principais modelos.

A métrica **ROC-AUC** permite avaliar a capacidade do modelo de distinguir entre transações normais e fraudulentas considerando diferentes limiares de decisão.

No conjunto analisado, o XGBoost apresentou ROC-AUC de aproximadamente **0,975**.

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/ee8a8972-68ba-48b3-ab38-fb0ad33dff65" />

---

# Curva Precision-Recall

Também foram analisadas as curvas **Precision-Recall**.

Essa análise é especialmente relevante neste projeto devido ao forte desbalanceamento entre as classes.

A curva permite observar o trade-off entre:

* aumentar o Recall e identificar mais fraudes;
* manter uma Precision elevada e reduzir falsos positivos.

Além da curva, é calculada a **Average Precision (AP)** para os modelos avaliados.

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/e45d99e1-b2b6-4d8f-a258-94785a882119" />

---

# Ajuste do Threshold

Por padrão, classificadores utilizam um threshold de aproximadamente `0.50` para transformar a probabilidade prevista em uma classe.

Neste projeto, foram testados diferentes thresholds:

```text
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.50
0.60
0.70
0.80
0.90
```

Para cada threshold foram calculados:

* Precision;
* Recall;
* F1-Score.

Essa análise permite visualizar como a mudança do limiar modifica o comportamento do classificador.

Thresholds menores podem aumentar o Recall, mas também podem aumentar os falsos positivos.

Thresholds maiores podem aumentar a Precision, mas podem deixar mais fraudes sem identificação.

O notebook também identifica o threshold que apresenta o maior F1-Score dentro da análise realizada.

---

# Interpretabilidade com SHAP

Para compreender o comportamento do XGBoost, foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**.

Foram realizadas duas análises.

<img width="875" height="568" alt="image" src="https://github.com/user-attachments/assets/db915da5-85e8-4fa9-bba8-301077728187" />
<img width="868" height="497" alt="image" src="https://github.com/user-attachments/assets/60e70d9c-1950-4fbf-a6f5-692ded7d0881" />



## Importância global

O gráfico de importância SHAP permite observar quais variáveis possuem maior influência nas previsões do modelo considerando o conjunto analisado.

Também é utilizado o gráfico beeswarm para visualizar a distribuição das contribuições das características.

## Explicação individual

Além da análise global, foi utilizado um gráfico **waterfall** para uma transação específica.

Esse gráfico permite observar quais características contribuíram para aumentar ou diminuir a previsão de fraude para aquela transação.

As variáveis `V1` até `V28` são utilizadas como variáveis numéricas no dataset. Como o dataset não fornece uma interpretação de negócio individual para essas variáveis, não são atribuídos significados específicos a elas sem evidência adicional.

<img width="913" height="600" alt="image" src="https://github.com/user-attachments/assets/ef19ff34-2c53-4e04-9c5b-fc0cb6f5ec3a" />


---

# Principais aprendizados

O projeto permitiu observar alguns pontos importantes sobre detecção de fraude:

* problemas altamente desbalanceados não devem ser avaliados somente por accuracy;
* Recall é especialmente importante quando o objetivo é identificar o maior número possível de fraudes;
* aumentar Recall pode resultar em aumento de falsos positivos;
* técnicas como undersampling e SMOTE podem alterar o comportamento dos modelos;
* o conjunto de teste deve manter sua distribuição original;
* diferentes thresholds produzem diferentes relações entre Precision e Recall;
* ROC-AUC e Precision-Recall fornecem perspectivas complementares;
* SHAP permite interpretar o comportamento do modelo tanto globalmente quanto para previsões individuais.

---

# Evoluções em relação à abordagem inicial

Além do fluxo apresentado durante o desafio, o projeto foi ampliado com:

* análise exploratória dos dados;
* visualização da distribuição das classes;
* comparação entre Logistic Regression, Random Forest e XGBoost;
* aplicação efetiva de undersampling;
* aplicação efetiva de SMOTE;
* comparação entre diferentes estratégias de balanceamento;
* utilização de GridSearchCV;
* análise de Precision, Recall, F1-Score e ROC-AUC;
* construção das curvas ROC;
* construção das curvas Precision-Recall;
* análise de diferentes thresholds;
* avaliação do threshold ajustado;
* utilização de SHAP para interpretabilidade global;
* utilização de SHAP para explicar uma previsão individual.

---

# Como executar

Clone o repositório:

```bash
git clone https://github.com/SEU-USUARIO/Deteccao_Fraudes_Bancarias.git
```

Instale as dependências:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib jupyter
```

Execute o notebook:

```text
deteccao_anomalias_transacoes.ipynb
```
