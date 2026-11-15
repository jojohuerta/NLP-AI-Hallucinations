# Resultados de la corrida de referencia

Este archivo resume **todo lo que devolvió cada experimento** en la corrida de
referencia: `--seed 42`, Python 3.12, GPU NVIDIA RTX 3080 Ti (12 GB). Los
números salen de los archivos de `results/` y de los `*_summary.json` generados
por los scripts.

> En una máquina distinta los valores pueden variar levemente por hardware,
> versión de `transformers` o actualizaciones de los datasets.

---

## Resumen general

| Experimento | Método | Métrica principal | Resultado |
|---|---|---|---|
| 1 | Bag-of-Words + LogReg sobre segmentos | F1 macro / MCC | **0,6406 / 0,3463** |
| 1 | TF-IDF + LogReg sobre segmentos | F1 macro / MCC | **0,6491 / 0,3303** |
| 1 | GPT-2, perplejidad dentro vs. fuera de spans | t-test apareado | **p = 0,985** (sin diferencia) |
| 2 | TF-IDF + LogReg, Mu-SHROOM vs. Phantom | F1 macro / MCC | **0,9833 / 0,9672** |
| 2 | DistilBERT fine-tuned | F1 macro / MCC | **1,0000 / 1,0000** |
| 2b | TF-IDF + LogReg, pares controlados | F1 macro / MCC | **0,5939 / 0,1995** |
| 2b | DistilBERT fine-tuned, pares controlados | F1 macro / MCC | **0,6157 / 0,2835** |
| 2b | Umbral de longitud (control) | F1 macro / MCC | **0,3333 / 0,0000** (piso de azar) |
| 3 | Entidades spaCy | F1 macro / MCC | **0,6528 / 0,4345** |
| 3 | LLM few-shot (Qwen2.5-3B) | F1 macro / MCC | **0,6753 / 0,3708** |
| 3 | Control de longitud | F1 macro / MCC | **0,8397 / 0,6822** |

El contraste central del TP: el Experimento 2 da un F1 casi perfecto, pero al
controlar fuente, tema y longitud (Experimento 2b) el rendimiento cae a ~0,60.
En el Experimento 3, un control trivial de longitud supera a los dos métodos
propuestos.

---

## Experimento 1 — Léxico y perplejidad

**Script:** `experiments/exp1_lexical_perplexity.py`

### Corpus

| Dato | Valor |
|---|---|
| Dataset | `Helsinki-NLP/mu-shroom`, config `en` |
| Splits | `validation` + `test` |
| Documentos | 204 |
| Documentos con al menos un span | 197 |
| Segmentos alucinados (Corpus A) | 609 |
| Segmentos veraces (Corpus B) | 599 |
| Segmentos usados en el análisis léxico | 797 |
| Documentos representados en el análisis léxico | 185 |

### Análisis léxico: Bag-of-Words vs. TF-IDF

Ambos modelos usan exactamente el mismo conjunto de segmentos y el mismo split
agrupado por documento (`GroupShuffleSplit` por `doc_id`, `test_size=0.25`),
`LogisticRegression` con `class_weight="balanced"` y la misma configuración de
vectorizador (unigramas + bigramas, `min_df=2`, `max_df=0.95`, stopwords en
inglés).

| Dato | Valor |
|---|---|
| Segmentos de train | 623 (138 documentos) |
| Segmentos de test | 174 (47 documentos) |
| Composición del test | 98 veraces (B) / 76 alucinados (A) |
| Vocabulario de evaluación | 692 |
| Features del ajuste completo | 892 |

**Métricas en test:**

| Métrica | Bag-of-Words | TF-IDF |
|---|---:|---:|
| F1 macro | **0,6406** | **0,6491** |
| MCC | **0,3463** | **0,3303** |
| Precisión · alucinación (A) | 0,5614 | 0,5728 |
| Recall · alucinación (A) | 0,8421 | 0,7763 |
| Precisión · veraz (B) | 0,8000 | 0,7606 |
| Recall · veraz (B) | 0,4898 | 0,5510 |

**Matriz de confusión en test:**

| Modelo | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Bag-of-Words | 48 | 50 | 12 | 64 |
| TF-IDF | 54 | 44 | 17 | 59 |

Los coeficientes negativos empujan hacia el Corpus B (veraz) y los positivos
hacia el Corpus A (alucinación).

**Top 15 de Bag-of-Words por |coeficiente|:**

| Palabra | Coeficiente | Dirección |
|---|---:|---|
| `located` | −1,5871 | B (veraz) |
| `population` | −1,4094 | B (veraz) |
| `born` | −1,3560 | B (veraz) |
| `released` | −1,3172 | B (veraz) |
| `called` | −1,2649 | B (veraz) |
| `won` | −1,2383 | B (veraz) |
| `total` | −1,1765 | B (veraz) |
| `yes` | −1,1381 | B (veraz) |
| `known` | −1,0369 | B (veraz) |
| `club` | −1,0198 | B (veraz) |
| `march` | 0,9840 | A (alucinación) |
| `question` | −0,9465 | B (veraz) |
| `olympics` | −0,9331 | B (veraz) |
| `emperor` | −0,9024 | B (veraz) |
| `small` | 0,8983 | A (alucinación) |

**Top 15 de TF-IDF por |coeficiente|:**

| Palabra | Coeficiente | Dirección |
|---|---:|---|
| `located` | −1,8824 | B (veraz) |
| `population` | −1,3279 | B (veraz) |
| `won` | −1,3223 | B (veraz) |
| `born` | −1,3071 | B (veraz) |
| `called` | −1,2004 | B (veraz) |
| `released` | −1,1978 | B (veraz) |
| `known` | −1,1665 | B (veraz) |
| `total` | −1,1410 | B (veraz) |
| `bc` | 1,0412 | A (alucinación) |
| `march` | 0,9776 | A (alucinación) |
| `species` | −0,9626 | B (veraz) |
| `yes` | −0,9508 | B (veraz) |
| `club` | −0,9303 | B (veraz) |
| `bmw` | −0,9140 | B (veraz) |
| `23` | 0,8669 | A (alucinación) |

### Perplejidad con GPT-2

| Dato | Valor |
|---|---|
| Modelo | `gpt2` |
| Dispositivo | `cuda` |
| Documentos procesados | 204 |
| Documentos evaluados | 167 |
| Documentos descartados por `min_tokens` | 37 |
| Documentos truncados | 0 |
| `max_length` | 1024 |
| Tokens del Corpus A | 3.272 |
| Tokens del Corpus B | 5.505 |
| NLL media A | 3,9083 |
| NLL media B | 3,9060 |
| Perplejidad media A | **49,8129** |
| Perplejidad media B | **49,6983** |

**Test apareado a nivel documento (167 documentos):**

| Estadístico | Valor |
|---|---|
| t | 0,0187 |
| p | **0,9851** |
| Grados de libertad | 166 |

**Lectura:** tanto BoW como TF-IDF alcanzan una señal léxica modesta (F1 ≈ 0,64–
0,65, MCC ≈ 0,33–0,35), con más recall que precisión sobre las alucinaciones. La
perplejidad de GPT-2, en cambio, no distingue tokens alucinados de veraces: las
medias son casi idénticas y el t-test apareado no rechaza la igualdad
(p ≈ 0,985).

### Archivos generados

```text
results/exp1/exp1_segments.csv
results/exp1/exp1_top_features_bow.csv
results/exp1/exp1_top_features_tfidf.csv
results/exp1/exp1_perplexity_per_document.csv
results/exp1/exp1_summary.json

graphs/exp1/exp1_lexical_metrics.html
graphs/exp1/exp1_top_features_bow.html
graphs/exp1/exp1_top_features_tfidf.html
graphs/exp1/exp1_perplexity_boxplot.html
graphs/exp1/exp1_perplexity_delta_histogram.html
```

---

## Experimento 2 — Clasificación binaria Mu-SHROOM vs. Phantom

**Script:** `experiments/exp2_binary_classification.py`

### Dataset construido

| Dato | Valor |
|---|---|
| Positivos disponibles (Mu-SHROOM con spans) | 197 |
| Positivos usados | 197 |
| Negativos únicos de Phantom (pool) | 1.415 |
| Negativos usados | 197 |
| Total | 394 (197 + 197) |
| Mínimo de palabras para un negativo | 3 |
| Train | 275 (137 negativos / 138 positivos) |
| Validación | 59 (30 negativos / 29 positivos) |
| Test | 60 (30 negativos / 30 positivos) |

**Longitud en palabras por clase:**

| Clase | Media | Mediana | Máximo |
|---|---:|---:|---:|
| Sin alucinación (Phantom) | 50,19 | 35 | 273 |
| Alucinación (Mu-SHROOM) | 40,20 | 18 | 249 |

### Baseline TF-IDF + LogisticRegression

| Métrica | Validación | Test |
|---|---:|---:|
| n | 59 | 60 |
| F1 macro | 0,9322 | **0,9833** |
| F1 alucinación | 0,9333 | 0,9836 |
| F1 sin alucinación | 0,9310 | 0,9831 |
| Precisión alucinación | 0,9032 | 0,9677 |
| Recall alucinación | 0,9655 | 1,0000 |
| Precisión sin alucinación | 0,9643 | 1,0000 |
| Recall sin alucinación | 0,9000 | 0,9667 |
| MCC | 0,8665 | **0,9672** |
| ROC-AUC | 0,9770 | 0,9944 |
| PR-AUC | 0,9758 | 0,9919 |

**Matriz de confusión en test:** TN = 29, FP = 1, FN = 0, TP = 30.

**Desagregado por fuente en test:**

| Fuente | n | Métrica | Valor |
|---|---:|---|---:|
| Mu-SHROOM (positivos) | 30 | Recall | 1,0000 |
| Phantom (negativos) | 30 | Especificidad | 0,9667 |

**Top features del baseline:**

| Feature | Coeficiente | Dirección |
|---|---:|---|
| `2024` | −1,3819 | Phantom (negativo) |
| `company` | −1,2476 | Phantom (negativo) |
| `located` | 1,1556 | Mu-SHROOM (positivo) |
| `million` | −1,0682 | Phantom (negativo) |
| `fund` | −1,0361 | Phantom (negativo) |
| `shares` | −0,9484 | Phantom (negativo) |
| `known` | 0,8859 | Mu-SHROOM (positivo) |
| `stock` | −0,8534 | Phantom (negativo) |
| `year` | −0,8198 | Phantom (negativo) |
| `risk` | −0,8139 | Phantom (negativo) |
| `named` | 0,8049 | Mu-SHROOM (positivo) |
| `financial` | −0,7857 | Phantom (negativo) |

Los términos separan **fuentes** (empresas/finanzas vs. Wikipedia), no
necesariamente alucinaciones.

### DistilBERT fine-tuned

| Métrica | Test |
|---|---:|
| F1 macro | **1,0000** |
| F1 alucinación | 1,0000 |
| F1 sin alucinación | 1,0000 |
| Precisión / recall alucinación | 1,0000 / 1,0000 |
| Precisión / recall sin alucinación | 1,0000 / 1,0000 |
| MCC | **1,0000** |
| ROC-AUC | 1,0000 |
| PR-AUC | 1,0000 |

**Matriz de confusión en test:** TN = 30, FP = 0, FN = 0, TP = 30.

**Entrenamiento:**

| Dato | Valor |
|---|---|
| Modelo | `distilbert-base-uncased` |
| `max_length` | 256 |
| Épocas | 3 |
| `train_loss` final | 0,4004 |
| `train_runtime` | 4,65 s |
| Mejor checkpoint | `checkpoint-36` (época 2) |
| Mejor métrica de validación (F1 macro) | 0,9661 |

**Truncamiento:**

| Dato | Valor |
|---|---|
| Tokens máximos | 501 |
| Percentil 95 | 185,05 |
| Truncados | 7 / 394 (1,78 %) |

### Comparación baseline vs. transformer

| Delta | Valor |
|---|---:|
| Δ F1 macro | +0,0167 |
| Δ MCC | +0,0328 |

### Advertencia

Todos los positivos vienen de Mu-SHROOM y todos los negativos de Phantom. El F1
casi perfecto puede reflejar separación de fuente, tema, estilo o longitud, no
detección genuina de alucinaciones. Por eso el Experimento 2b existe.

### Archivos generados

```text
results/exp2/exp2_balanced_dataset.csv
results/exp2/exp2_baseline_metrics.json
results/exp2/exp2_baseline_predictions.csv
results/exp2/exp2_baseline_predictions_validation.csv
results/exp2/exp2_baseline_top_features.csv
results/exp2/exp2_transformer_metrics.json
results/exp2/exp2_transformer_predictions.csv
results/exp2/exp2_transformer_training_log.csv
results/exp2/exp2_metrics_comparison.csv
results/exp2/exp2_summary.json
results/exp2/transformer_checkpoints/

graphs/exp2/exp2_class_balance.html
graphs/exp2/exp2_answer_length_distribution.html
graphs/exp2/exp2_baseline_top_features.html
graphs/exp2/exp2_confusion_baseline.html
graphs/exp2/exp2_confusion_transformer.html
graphs/exp2/exp2_transformer_training_curves.html
graphs/exp2/exp2_model_comparison.html
graphs/exp2/exp2_probability_distribution.html
```

---

## Experimento 2b — Phantom pareado con controles

**Script:** `experiments/exp2b_phantom_paired_control.py`

### Dataset construido

| Dato | Valor |
|---|---|
| Filas crudas de Phantom | 30.886 |
| Respuestas únicas (texto normalizado) | 3.985 |
| Queries únicas en los pares | 324 |
| Contextos únicos en los pares | 324 |
| Pares candidatos | 1.888 |
| Pares usados | **324** |
| Pares descartados por respuestas cortas | 1.189 |
| Pares descartados por duplicado (Jaccard ≥ 0,98) | 375 |
| Grupos sin ambas etiquetas | 209 |
| `target_words` | 40 |
| `max_jaccard` | 0,98 |
| Puntuación eliminada | Sí |
| Total de ejemplos | 648 (324 + 324) |
| Pares con prefijo idéntico | 0 (0 %) |

**Solapamiento dentro de cada par:**

| Estadístico | Jaccard |
|---|---:|
| Media | 0,6887 |
| Mediana | 0,7037 |
| Percentil 90 | 0,9394 |

**Longitud original (palabras):**

| Clase | Media | Mediana |
|---|---:|---:|
| Sin alucinación | 72,52 | 63 |
| Alucinación | 102,37 | 93 |

Después del recorte, ambas clases tienen exactamente **40 palabras** (media,
mediana y máximo = 40).

**Split:**

| Partición | Total | Negativos | Positivos |
|---|---:|---:|---:|
| Train | 464 | 232 | 232 |
| Validación | 92 | 46 | 46 |
| Test | 92 | 46 | 46 |

### Modelos sobre el diseño final (tema + longitud)

| Modelo | Split | F1 macro | MCC | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| TF-IDF + LogReg | Validación | 0,5862 | 0,1746 | 0,6621 | 0,6988 |
| TF-IDF + LogReg | Test | **0,5939** | **0,1995** | 0,5969 | 0,6049 |
| DistilBERT | Test | **0,6157** | **0,2835** | 0,6531 | 0,6493 |
| Umbral de longitud | Validación | 0,3333 | 0,0000 | 0,5000 | 0,5000 |
| Umbral de longitud | Test | 0,3333 | 0,0000 | 0,5000 | 0,5000 |
| Query-only | Validación | 0,3333 | 0,0000 | 0,5000 | 0,5000 |
| Query-only | Test | 0,3333 | 0,0000 | 0,5000 | 0,5000 |

**Matriz de confusión en test (TF-IDF):** TN = 23, FP = 23, FN = 14, TP = 32.

**Matriz de confusión en test (DistilBERT):** TN = 38, FP = 8, FN = 26, TP = 20.

**DistilBERT, entrenamiento:**

| Dato | Valor |
|---|---|
| `max_length` | 256 |
| Épocas | 3 |
| `train_loss` final | 0,6846 |
| `train_runtime` | 3,85 s |
| Mejor checkpoint | `checkpoint-58` (época 2) |
| Mejor F1 macro de validación | 0,5927 |
| Tokens máximos | 69 |
| Truncados | 0 (0 %) |

Un predictor constante sobre el test balanceado obtiene F1 macro = 1/3 y
MCC = 0. Por eso **0,3333 es el piso de azar**.

### Referencias de ablación

| Condición | Modelo | F1 macro | MCC |
|---|---|---:|---:|
| Sin control | TF-IDF (texto completo) | 0,8571 | 0,7160 |
| Sin control | Umbral de longitud | 0,9003 | 0,8187 |
| Tema controlado | TF-IDF (texto completo) | 0,6934 | 0,4535 |
| Tema controlado | Umbral de longitud | 0,6675 | 0,4154 |
| Tema + longitud (diseño final) | TF-IDF | 0,5939 | 0,1995 |
| Tema + longitud (diseño final) | Umbral de longitud | 0,3333 | 0,0000 |

**Referencia “sin control”:** 648 ejemplos balanceados de respuestas completas
(324 por clase). En test quedaron 91 ejemplos (46 negativos / 45 positivos)
porque el split no es exactamente 50/50. Longitud media: 34,91 palabras
(limpia) vs. 75,69 (alucinada); mediana 22,5 vs. 66. Umbral de longitud elegido:
38,201.

**Referencia “tema controlado, longitud no”:** los mismos 648 ejemplos pareados
pero con texto completo. Umbral de longitud elegido: 58,754.

### Diagnóstico del artefacto de recorte

| Grupo | Entrada | F1 macro | MCC | n |
|---|---|---:|---:|---:|
| Recorte ingenuo | Regla de puntuación | **0,9459** | 0,8939 | 296 |
| Recorte ingenuo | Umbral de longitud | 0,5000 | 0,0000 | 296 |
| Recorte ingenuo | TF-IDF | 0,5999 | 0,2042 | 296 |
| Diseño final | Regla de puntuación | 0,3333 | 0,0000 | 92 |
| Diseño final | Umbral de longitud | 0,3333 | 0,0000 | 92 |
| Diseño final | Query-only | 0,3333 | 0,0000 | 92 |
| Diseño final | TF-IDF | 0,5939 | 0,1995 | 92 |
| Diseño final | DistilBERT | 0,6157 | 0,2835 | 92 |

En el recorte ingenuo, el 90,82 % de las respuestas limpias termina en
puntuación, pero solo el 2,13 % de las alucinadas. Una regla trivial “si termina
en punto, es limpia” alcanza F1 = 0,9459. Al quitar la puntuación, esa regla cae
al piso de azar. El gráfico `exp2b_artifact_diagnostic.html` compara ambos
mundos.

### Sensibilidad a la longitud de recorte (TF-IDF)

| `target_words` | Pares | n test | F1 macro | MCC | ROC-AUC | Control de longitud F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 299 | 84 | 0,5204 | 0,0765 | 0,6227 | 0,3333 |
| 40 | 324 | 92 | 0,5939 | 0,1995 | 0,5969 | 0,3333 |
| 60 | 264 | 74 | 0,6878 | 0,3819 | 0,6654 | 0,3333 |

**Lectura:** al controlar fuente, tema, longitud y puntuación, el F1 cae de
~0,86 a ~0,60. Queda señal por encima del azar, pero muy lejos del F1 casi
perfecto del Experimento 2. La conclusión es que buena parte de ese 0,98 era
separación de fuente, tema y verbosidad.

**Top features del TF-IDF final (hacia alucinación):** `potentially` (0,7544),
`comprehensive` (0,6107), `includes` (0,5496), `resulting` (0,5008),
`including` (0,4315), `liquidity` (0,4298), `role` (0,3937), `reflecting`
(0,3758), `substantial` (0,3589), `approximately` (0,3584). Hacia limpio:
`able` (−0,4177), `ii` (−0,4064), `risk risks` (−0,3463), `applicable`
(−0,3158), `shall` (−0,3150), `document` (−0,3077).

### Limitaciones

- El test tiene 92 ejemplos: intervalos de confianza amplios.
- Solo se conservan pares donde ambas respuestas son largas (≥ 40 tokens).
- El recorte fijo puede eliminar la cola donde aparece la alucinación.
- No es directamente comparable con el Experimento 2 porque usa solo Phantom,
  pares de la misma query y textos recortados.

### Archivos generados

```text
results/exp2b/exp2b_balanced_pairs.csv
results/exp2b/exp2b_pair_diagnostics.csv
results/exp2b/exp2b_baseline_predictions.csv
results/exp2b/exp2b_baseline_predictions_validation.csv
results/exp2b/exp2b_transformer_predictions.csv
results/exp2b/exp2b_transformer_training_log.csv
results/exp2b/exp2b_length_control_predictions.csv
results/exp2b/exp2b_query_only_predictions.csv
results/exp2b/exp2b_metrics.csv
results/exp2b/exp2b_control_ablation.csv
results/exp2b/exp2b_artifact_diagnostic.csv
results/exp2b/exp2b_length_sensitivity.csv
results/exp2b/exp2b_top_features.csv
results/exp2b/exp2b_summary.json
results/exp2b/transformer_checkpoints/

graphs/exp2b/exp2b_split_balance.html
graphs/exp2b/exp2b_length_control.html
graphs/exp2b/exp2b_pair_overlap.html
graphs/exp2b/exp2b_top_features.html
graphs/exp2b/exp2b_confusion_baseline.html
graphs/exp2b/exp2b_confusion_transformer.html
graphs/exp2b/exp2b_transformer_training_curves.html
graphs/exp2b/exp2b_model_comparison.html
graphs/exp2b/exp2b_probability_distribution.html
graphs/exp2b/exp2b_artifact_diagnostic.html
graphs/exp2b/exp2b_control_ablation.html
```

---

## Experimento 3 — Entidades vs. LLM few-shot

**Script:** `experiments/exp3_logical_consistency.py`

### Dataset construido

| Dato | Valor |
|---|---|
| Dataset | `seyled/Phantom_Hallucination_Detection` |
| Filas crudas | 30.886 |
| Filas después de deduplicar `(query, context, answer)` | 30.886 |
| Muestra | 200 (100 alucinaciones + 100 limpias) |
| Grupos (queries) únicos | 96 |
| Desarrollo | 100 (50 + 50) |
| Validación | 50 (25 + 25) |
| Test | 50 (25 + 25) |

**Longitud en palabras por clase:**

| Clase | Media | Mediana | Máximo |
|---|---:|---:|---:|
| Sin alucinación | 30,73 | 24 | 135 |
| Alucinación | 75,85 | 71 | 171 |

### Baseline de entidades con spaCy

Score: `unsupported_entity_rate` (fracción de entidades de la respuesta que no
aparecen en `query + context`). Umbral elegido en desarrollo: **0,035734**.

| Split | n | F1 macro | MCC | ROC-AUC | IC 95 % F1 | IC 95 % MCC |
|---|---:|---:|---:|---:|---|---|
| Desarrollo | 100 | 0,5711 | 0,1983 | 0,5656 | [0,4715, 0,6658] | [0,0038, 0,3939] |
| Validación | 50 | 0,5536 | 0,1236 | 0,5408 | [0,4066, 0,6847] | [−0,1574, 0,3869] |
| Test | 50 | **0,6528** | **0,4345** | 0,6736 | [0,5225, 0,7831] | [0,2203, 0,6218] |

**Detalle en test:**

| Métrica | Valor |
|---|---:|
| F1 alucinación | 0,5556 |
| F1 sin alucinación | 0,7500 |
| Precisión alucinación | 0,9091 |
| Recall alucinación | 0,4000 |
| Precisión sin alucinación | 0,6154 |
| Recall sin alucinación | 0,9600 |

**Matriz de confusión en test:** TN = 24, FP = 1, FN = 15, TP = 10.

El baseline es muy conservador: casi no comete falsos positivos, pero se le
escapan 15 de 25 alucinaciones.

### Control de longitud

Score: `n_words`. Umbral elegido en desarrollo: **49,0**.

| Split | n | F1 macro | MCC | ROC-AUC | IC 95 % F1 | IC 95 % MCC |
|---|---:|---:|---:|---:|---|---|
| Desarrollo | 100 | 0,8591 | 0,7294 | 0,8712 | [0,7864, 0,9274] | [0,5930, 0,8561] |
| Validación | 50 | 0,7537 | 0,5489 | 0,8416 | [0,6186, 0,8586] | [0,3092, 0,7349] |
| Test | 50 | **0,8397** | **0,6822** | 0,8728 | [0,7182, 0,9388] | [0,4480, 0,8796] |

**Detalle en test:**

| Métrica | Valor |
|---|---:|
| F1 alucinación | 0,8462 |
| F1 sin alucinación | 0,8333 |
| Precisión alucinación | 0,8148 |
| Recall alucinación | 0,8800 |
| Precisión sin alucinación | 0,8696 |
| Recall sin alucinación | 0,8000 |

**Matriz de confusión en test:** TN = 20, FP = 5, FN = 3, TP = 22.

### LLM few-shot

Modelo: `Qwen/Qwen2.5-3B-Instruct`. Cuatro ejemplos etiquetados, contexto
recuperado por TF-IDF, decodificación greedy, `max_new_tokens = 12`.

| Split | n | F1 macro | MCC | IC 95 % F1 | IC 95 % MCC |
|---|---:|---:|---:|---|---|
| Validación | 50 | 0,5942 | 0,2060 | [0,4492, 0,7196] | [−0,0727, 0,4695] |
| Test | 50 | **0,6753** | **0,3708** | [0,5398, 0,7971] | [0,1088, 0,6103] |

**Detalle en test:**

| Métrica | Valor |
|---|---:|
| F1 alucinación | 0,7143 |
| F1 sin alucinación | 0,6364 |
| Precisión alucinación | 0,6452 |
| Recall alucinación | 0,8000 |
| Precisión sin alucinación | 0,7368 |
| Recall sin alucinación | 0,5600 |

**Matriz de confusión en test:** TN = 14, FP = 11, FN = 5, TP = 20.

**Metadatos de generación:**

| Dato | Valor |
|---|---|
| Filas procesadas | 100 (validación + test) |
| Respuestas inválidas | 0 |
| Contextos truncados | 0 |
| Tokens de prompt mínimos | 1.567 |
| Tokens de prompt medianos | 1.835 |
| Tokens de prompt máximos | 3.087 |
| Tiempo de generación | 29,77 s |

Los 4 ejemplos few-shot usados fueron los `uid` 25 (alucinación), 176 (limpia),
19 (alucinación) y 171 (limpia), todos con 3 oraciones de contexto.

### McNemar: baseline vs. LLM

| Dato | Valor |
|---|---:|
| Baseline correcto / LLM incorrecto | 12 |
| LLM correcto / baseline incorrecto | 12 |
| Discordantes | 24 |
| p | **1,0000** |

No hay diferencia estadísticamente detectable entre el baseline de entidades y
el LLM few-shot en el test.

### Variantes de entidades y controles

| Variante | Split | Umbral | AUC | F1 macro | MCC |
|---|---|---:|---:|---:|---:|
| Cobertura de entidades de la query | Dev | 1,0 | 0,6836 | 0,6795 | 0,3612 |
| Cobertura de entidades de la query | Test | 1,0 | **0,7488** | **0,7200** | **0,4400** |
| Cantidad de entidades en la respuesta | Dev | 3,0 | 0,6962 | 0,7172 | 0,4491 |
| Cantidad de entidades en la respuesta | Test | 3,0 | 0,7392 | 0,6753 | 0,3708 |
| Entidades no respaldadas (fracción) | Dev | 0,0357 | 0,5656 | 0,5711 | 0,1983 |
| Entidades no respaldadas (fracción) | Test | 0,0357 | 0,6736 | 0,6528 | 0,4345 |
| Entidades no respaldadas (conteo) | Dev | 1,0 | 0,5844 | 0,5711 | 0,1983 |
| Entidades no respaldadas (conteo) | Test | 1,0 | 0,6824 | 0,6528 | 0,4345 |
| Números no respaldados (fracción) | Dev | 0,0647 | 0,5196 | 0,3912 | 0,1021 |
| Números no respaldados (fracción) | Test | 0,0647 | 0,5384 | 0,4415 | 0,1474 |
| Números no respaldados (conteo) | Dev | 1,0 | 0,5200 | 0,3912 | 0,1021 |
| Números no respaldados (conteo) | Test | 1,0 | 0,5408 | 0,4415 | 0,1474 |
| Longitud de la respuesta (control) | Dev | 49,0 | 0,8712 | 0,8591 | 0,7294 |
| Longitud de la respuesta (control) | Test | 49,0 | **0,8728** | **0,8397** | **0,6822** |

### Análisis de errores

`exp3_errors.csv` contiene **28 filas de test** donde al menos un método falló
(es la unión de errores, no una fila por error):

| Método | Errores en test (de 50) |
|---|---:|
| Baseline de entidades | 16 |
| LLM few-shot | 16 |
| Control de longitud | 4 |
| Baseline y LLM incorrectos a la vez | 4 |

**Lectura:** el control de longitud supera ampliamente a los dos métodos y es el
que menos errores comete (4). Las alucinaciones de Phantom son
sistemáticamente más largas, así que el resultado mide separabilidad
superficial/verbosidad, no detección semántica genuina. El gráfico
`exp3_score_vs_length.html` muestra la relación entre score y longitud.

### Archivos generados

```text
results/exp3/exp3_sample.csv
results/exp3/exp3_predictions_test.csv
results/exp3/exp3_predictions_validation.csv
results/exp3/exp3_metrics.csv
results/exp3/exp3_threshold_search.csv
results/exp3/exp3_length_threshold_search.csv
results/exp3/exp3_entity_variants.csv
results/exp3/exp3_llm_raw_predictions.csv
results/exp3/exp3_llm_few_shot_examples.csv
results/exp3/exp3_errors.csv
results/exp3/exp3_summary.json

graphs/exp3/exp3_split_balance.html
graphs/exp3/exp3_answer_length_distribution.html
graphs/exp3/exp3_entity_grounding.html
graphs/exp3/exp3_threshold_curve.html
graphs/exp3/exp3_confusion_baseline.html
graphs/exp3/exp3_confusion_llm.html
graphs/exp3/exp3_model_comparison.html
graphs/exp3/exp3_score_vs_length.html
graphs/exp3/exp3_entity_variants.html
```

---

## Conclusiones

1. **Señal léxica débil pero real (Exp. 1):** TF-IDF separa segmentos alucinados
   de veraces con F1 0,65, aunque la perplejidad de GPT-2 no muestra diferencia
   (p = 0,985).
2. **El F1 casi perfecto del Exp. 2 es engañoso:** el clasificador separa
   Mu-SHROOM de Phantom, no alucinaciones de respuestas veraces. Los términos
   más predictivos son propios de cada fuente.
3. **Al controlar fuente, tema, longitud y puntuación (Exp. 2b), el rendimiento
   cae a ~0,60.** Sigue habiendo algo de señal por encima del azar, pero muy
   lejos de 0,98. El control de longitud cae exactamente al piso (0,3333).
4. **En el Exp. 3, la longitud gana:** un umbral trivial sobre la cantidad de
   palabras (F1 0,84) supera al baseline de entidades (0,65) y al LLM few-shot
   (0,68). McNemar no encuentra diferencia entre baseline y LLM (p = 1,0).
5. **Conclusión metodológica:** buena parte del rendimiento reportado en tareas
   de detección de alucinaciones puede explicarse por artefactos del dataset
   (fuente, tema, longitud, puntuación). Los experimentos 2b y 3 están diseñados
   para hacer visible ese efecto.
