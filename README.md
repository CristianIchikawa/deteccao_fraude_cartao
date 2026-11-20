# Detecção de fraude em cartão de crédito

Projeto de classificação binária em um cenário muito desbalanceado (99,83% de transações normais e 0,17% de fraudes). O foco é escolher as métricas certas para a classe de fraude, comparar modelos, ajustar o limiar de decisão e explicar o que levou o modelo a marcar uma transação como fraude.

## Como executar

O notebook foi feito para o Google Colab. A base é lida **diretamente pelo link**, então não é preciso baixar nada, e o arquivo `creditcard.csv` não está neste repositório (passa do limite de 100 MB do GitHub).

1. Abra `deteccao_fraude_cartao.ipynb` no Google Colab.
2. Se a biblioteca `shap` não estiver instalada, rode `!pip install -q shap` antes da primeira célula.
3. Execute as células em ordem.
4. Base de dados: <https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv>

```
deteccao-fraude-cartao/
├── deteccao_fraude_cartao.ipynb   # notebook com as saídas salvas (relatórios e curvas)
└── README.md
```

## 1. O problema e por que o desbalanceamento muda a avaliação

Na base, `Time` marca o momento, `Amount` o valor e `Class` a resposta (1 = fraude). As variáveis `V1` a `V28` foram transformadas por PCA para proteger a privacidade. Apenas 0,17% das transações são fraude (98 fraudes entre 56.962 transações do conjunto de teste).

Por isso a **acurácia engana**: um modelo que responde "não é fraude" para tudo acertaria cerca de 99,83% (a proporção da classe normal) e não encontraria nenhuma fraude. O mesmo aparece nos modelos treinados: a regressão logística tem 97,5% de acurácia, mas só 6% de precisão na classe de fraude. A avaliação usa as métricas da classe de fraude:

- **Recall:** das fraudes reais, quantas o modelo encontrou.
- **Precisão:** das transações marcadas como fraude, quantas eram fraude de verdade.
- **F1:** equilíbrio entre as duas.
- **Curva de precisão e recall:** mais informativa que a ROC em dados desbalanceados (veja a seção 3).

## 2. Preparação dos dados

- Criação de `Amount_log = np.log1p(Amount)`, para reduzir a assimetria do valor. `Amount` original foi removido, para não manter duas versões da mesma informação.
- Divisão treino/teste de 80/20 com `stratify=y` e `random_state=42`, mantendo a proporção de fraudes nos dois conjuntos.
- Padronização com `StandardScaler` **dentro de um `Pipeline`**, de modo que o scaler seja ajustado só com os dados de treino (evita vazamento).
- Desbalanceamento tratado com pesos de classe: `class_weight="balanced"` na regressão logística e no Random Forest e `scale_pos_weight` no XGBoost.

## 3. Comparação entre os modelos

Resultados no conjunto de teste (limiar 0,5), classe de fraude:

| Modelo | Recall | Precisão | F1 |
|---|---|---|---|
| Regressão logística (baseline) | 0,9184 | 0,0603 | 0,1131 |
| Random Forest | 0,7449 | 0,9605 | 0,8391 |
| XGBoost | 0,8367 | 0,7387 | 0,7847 |
| Random Forest ajustado (GridSearchCV por recall) | 0,8469 | 0,6336 | 0,7249 |

**O que os números mostram:**

- A regressão logística encontra quase todas as fraudes, mas com precisão de 6%: são cerca de 1.400 alarmes falsos para 90 fraudes encontradas, o que a torna inviável na prática.
- O Random Forest padrão tem o melhor equilíbrio (maior F1), com poucos falsos alarmes, mas deixa passar cerca de 1 em cada 4 fraudes.
- O XGBoost fica entre os dois: encontra mais fraudes que o Random Forest, com mais alarmes falsos.
- **Otimizar só o recall no GridSearchCV piorou o modelo.** A busca escolheu `max_depth=8` e `n_estimators=50`, árvores mais rasas e mais permissivas. O recall subiu cerca de 10 pontos percentuais, mas a precisão caiu cerca de 33 pontos e o F1 caiu de 0,839 para 0,725. Para selecionar hiperparâmetros em problemas desbalanceados, `f1` ou `average_precision` tendem a ser critérios mais estáveis que o recall isolado.

### Curvas ROC e de precisão e recall

| Modelo | AUC (ROC) | Average Precision (PR) |
|---|---|---|
| Regressão logística | 0,97 | 0,71 |
| Random Forest | 0,95 | 0,85 |
| XGBoost | 0,98 | 0,86 |

A ROC dá a impressão de que os três modelos são parecidos (AUC entre 0,95 e 0,98), e a regressão logística, que gera cerca de 1.400 alarmes falsos, chega a ter AUC igual à do XGBoost. A curva de precisão e recall separa melhor os modelos: a regressão logística fica bem abaixo (AP 0,71), e o Random Forest e o XGBoost ficam próximos (0,85 e 0,86). Isso reforça a escolha da curva de precisão e recall como a leitura principal neste problema.

## 4. Limiar de decisão e SHAP

### Limiar

Comparei o limiar padrão de 0,5 com 0,3 no Random Forest ajustado:

| Limiar | Recall | Precisão | F1 |
|---|---|---|---|
| 0,5 | 0,8469 | 0,6336 | 0,7249 |
| 0,3 | 0,8878 | 0,3425 | 0,4943 |

Baixar o limiar capturou cerca de 4 fraudes a mais (de ~83 para ~87 entre as 98), ao custo de cerca de 119 alarmes falsos a mais (de ~48 para ~167). São, em média, uns 30 alarmes falsos para cada fraude extra encontrada.

**Limiar escolhido: 0,5.** O ganho de recall do 0,3 é pequeno perto da queda de precisão e de F1. Em um sistema real, cada alarme falso significa um cliente legítimo bloqueado ou um caso para a equipe de análise revisar.

### Importância das variáveis e SHAP

- Importância das variáveis no Random Forest ajustado, em ordem: `V14` (0,27), `V4`, `V10`, `V17`, `V3` e `V12`. `V14` é a mais relevante com folga.
- O gráfico de resumo do SHAP foi gerado sobre o **XGBoost**, a partir de uma amostra aleatória de 2.000 transações do teste.

**O que o SHAP mostrou.** As variáveis mais influentes, em ordem, são `V14`, `V4`, `V12`, `V10` e `V11`. O eixo horizontal está em log-odds: valores positivos empurram a previsão para fraude e valores negativos, para normal.

- **`V14`** é a que mais pesa. Valores **baixos** empurram para fraude (alguns pontos passam de +4) e valores altos puxam para normal.
- **`V4`** age no sentido oposto: valores **altos** empurram para fraude e valores baixos puxam para normal.
- **`V12`, `V10` e `V3`** seguem o padrão de `V14`: valores baixos aumentam o risco de fraude.
- **`V11`** segue o padrão de `V4`: valores altos aumentam o risco.
- `Amount_log` e `Time` aparecem no meio do gráfico, com efeito pequeno perto das variáveis de PCA.

A maior parte dos pontos tem SHAP negativo, o que é esperado: a amostra tem poucas fraudes (cerca de 3 a 4 em 2.000 transações), e é nelas que estão os valores positivos.

**Por que a ordem difere da importância do Random Forest.** `V14` e `V4` aparecem no topo nos dois métodos, mas o restante da ordem muda (por exemplo, `V17` está entre as mais importantes no Random Forest e nem aparece entre as 20 primeiras do SHAP). Isso é esperado: são modelos diferentes e medidas diferentes (importância por redução de impureza versus contribuição para a previsão).

As variáveis `V1` a `V28` vêm de PCA e são anônimas, então o sinal é útil para o modelo, mas não tem significado direto de negócio.

## 5. O que mudei em relação ao que a Expert fez

- Levei o **Random Forest** para o `Pipeline`, com `GridSearchCV` e ajuste de limiar, em vez de manter só a regressão logística nessa etapa, como sugerido nas ideias de evolução do desafio.
- Incluí o **XGBoost** com `scale_pos_weight` e usei o SHAP sobre ele.
- Comparei o resultado do `GridSearchCV` por recall com o Random Forest padrão e documentei que otimizar só o recall piorou o F1.
- Medi o custo operacional de baixar o limiar em número de falsos alarmes por fraude extra encontrada.
- Comparei a ROC com a curva de precisão e recall usando AUC e Average Precision, para mostrar que a ROC esconde a diferença entre os modelos.

## Limitações

- O teste tem apenas 98 fraudes, então cada fraude vale cerca de 1 ponto percentual de recall. As diferenças entre modelos são ruidosas.
- O `GridSearchCV` foi reduzido (4 combinações, 3 dobras) por causa do tempo de execução no Colab.
- A comparação de limiares foi feita olhando o próprio conjunto de teste. O ideal é escolher o limiar por validação cruzada no treino e usar o teste só na avaliação final.
- A amostra do SHAP é aleatória e contém poucas fraudes, e o SHAP foi calculado no XGBoost, enquanto o limiar foi analisado no Random Forest ajustado.
