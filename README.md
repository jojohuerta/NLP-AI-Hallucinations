# NLP-AI-Hallucinations

Trabajo práctico de Procesamiento del Lenguaje Natural (ITBA) sobre **detección
de alucinaciones en modelos de lenguaje**.

El repositorio contiene cuatro experimentos independientes:

| Experimento | Script | Enfoque |
|---|---|---|
| 1 | `exp1_lexical_perplexity.py` | Léxico (TF-IDF) y perplejidad (GPT-2) a nivel segmento |
| 2 | `exp2_binary_classification.py` | Clasificación binaria Mu-SHROOM vs. Phantom |
| 2b | `exp2b_phantom_paired_control.py` | Pares de Phantom con fuente, tema y longitud controlados |
| 3 | `exp3_logical_consistency.py` | Entidades con spaCy vs. LLM few-shot |

Cada script es autocontenido (salvo 2b, que reutiliza helpers de 2), usa semilla
fija y genera sus propios gráficos Altair y archivos de resultados. Los
experimentos **no dependen entre sí** y se pueden correr en cualquier orden.

Unos resultados numéricos de referencia se encuentran en **[Results.md](Results.md)**. 
Este README describe únicamente requisitos, ejecución, estructura, datasets y metodología.

Los siguientes resultados de referencia fueron obtenidos utilizando una **GPU NVIDIA RTX 3080 Ti de 12 GB** con CUDA.

---

## Índice

1. [Requisitos](#requisitos)
2. [Cómo correr todos los experimentos](#cómo-correr-todos-los-experimentos)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Datasets](#datasets)
5. [Experimentos](#experimentos)
6. [Gráficos y exportación](#gráficos-y-exportación)
7. [Reproducibilidad](#reproducibilidad)
8. [Solución de problemas](#solución-de-problemas)

---

## Requisitos

### Hardware y sistema

- **Python 3.12** (probado con esa versión).
- Los cuatro experimentos pueden correr en CPU, aunque mucho más lento. El
  Experimento 1 usa `--device cpu`; los experimentos 2, 2b y 3 aceptan `--cpu`.
- Espacio en disco: varios GB para dependencias, datasets y modelos. El LLM del
  Experimento 3 pesa ~6 GB, GPT-2 ~500 MB y DistilBERT ~250 MB.
- Conexión a internet la primera vez: los datasets y modelos se descargan
  automáticamente de HuggingFace.

### Entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Dependencias por experimento

Las dependencias están separadas en tres archivos `requirements-exp*.txt`. **No
hay un `requirements.txt` único**: hay que instalar el archivo del experimento
que se quiera correr.

| Archivo | Sirve para | Incluye además |
|---|---|---|
| `requirements-exp1.txt` | Experimento 1 | `torch`, `transformers` (GPT-2) |
| `requirements-exp2.txt` | Experimentos 2 y **2b** | `accelerate` (fine-tuning) |
| `requirements-exp3.txt` | Experimento 3 | `accelerate`, `spacy` y el wheel de `en_core_web_sm` |

Todos incluyen `datasets`, `pandas`, `numpy`, `altair`, `scikit-learn` y
`scipy`.

Para correr **todos** los experimentos, instalá los tres archivos:

```bash
pip install -r requirements-exp1.txt
pip install -r requirements-exp2.txt
pip install -r requirements-exp3.txt
```

O en un solo comando:

```bash
pip install -r requirements-exp1.txt -r requirements-exp2.txt -r requirements-exp3.txt
```

> **Experimento 2b:** usa `requirements-exp2.txt`. No tiene archivo propio
> porque reutiliza el pipeline, las métricas y los gráficos del Experimento 2.

> **Exportar gráficos a PNG (opcional):** requiere `vl-convert-python`, que está
> comentado en los tres requirements:
>
> ```bash
> pip install vl-convert-python
> ```

### Modelos de spaCy

`requirements-exp3.txt` intenta instalar `en_core_web_sm` desde un wheel de
GitHub. Si esa instalación falla, ejecutá:

```bash
python -m spacy download en_core_web_sm
```

### Token de HuggingFace

Los datasets usados son públicos y no requieren token. El token solo hace falta
si HuggingFace limita las descargas o si se usa un modelo *gated* (por ejemplo,
Gemma):

```bash
export HF_TOKEN="hf_..."
# alternativa: huggingface-cli login
```

### Verificación rápida de PyTorch y GPU

```bash
python test_pytorch.py
```

Imprime la versión de PyTorch y, si hay CUDA disponible, el nombre de la GPU.

---

## Cómo correr todos los experimentos

Con el entorno activado y las dependencias instaladas, desde la raíz del
repositorio:

```bash
python experiments/exp1_lexical_perplexity.py
python experiments/exp2_binary_classification.py
python experiments/exp2b_phantom_paired_control.py
python experiments/exp3_logical_consistency.py
```

La primera corrida descarga los datasets y los modelos. El Experimento 2b
importa funciones de `exp2_binary_classification.py`, así que ambos archivos
deben estar juntos en `experiments/`; no hace falta correr el 2 antes que el 2b.

### Resumen de ejecución

| Experimento | Instalar | Correr | Salidas |
|---|---|---|---|
| 1 | `pip install -r requirements-exp1.txt` | `python experiments/exp1_lexical_perplexity.py` | `graphs/exp1/`, `results/exp1/` |
| 2 | `pip install -r requirements-exp2.txt` | `python experiments/exp2_binary_classification.py` | `graphs/exp2/`, `results/exp2/` |
| 2b | `pip install -r requirements-exp2.txt` | `python experiments/exp2b_phantom_paired_control.py` | `graphs/exp2b/`, `results/exp2b/` |
| 3 | `pip install -r requirements-exp3.txt` | `python experiments/exp3_logical_consistency.py` | `graphs/exp3/`, `results/exp3/` |

Cada script imprime al final la lista de archivos generados. Todos aceptan
`--help` con la lista completa de opciones.

### Corridas rápidas de verificación

Sirven para comprobar que la instalación funciona sin descargar los modelos más
grandes:

```bash
# Experimento 1: solo parte léxica, sin GPT-2
python experiments/exp1_lexical_perplexity.py --skip-perplexity --max-examples 20

# Experimento 2: solo baseline léxico, sin DistilBERT
python experiments/exp2_binary_classification.py --skip-transformer --max-per-class 20 --epochs 1

# Experimento 2b: solo baselines y controles, sin DistilBERT
python experiments/exp2b_phantom_paired_control.py --skip-transformer --max-pairs 60

# Experimento 3: solo entidades y control de longitud, sin LLM
python experiments/exp3_logical_consistency.py --skip-llm --n-per-class 20
```

### Directorios de salida personalizados

Los cuatro scripts aceptan `--graphs-dir` y `--results-dir` para redirigir las
salidas. Los tres que usan HuggingFace aceptan además `--cache-dir` para cambiar
la caché de datasets:

```bash
python experiments/exp1_lexical_perplexity.py \
  --graphs-dir /tmp/graficos --results-dir /tmp/resultados --cache-dir /tmp/hf-cache
```

### Análisis exploratorio original

`data_analysis.py` es el EDA original del TP (no forma parte de los cuatro
experimentos). Genera cuatro gráficos en `graphs/`:

```bash
python data_analysis.py
```

Requiere las mismas dependencias que el Experimento 1.

---

## Estructura del repositorio

```text
.
├── experiments/
│   ├── exp1_lexical_perplexity.py        # Exp. 1: léxico + perplejidad
│   ├── exp2_binary_classification.py     # Exp. 2: clasificación binaria
│   ├── exp2b_phantom_paired_control.py   # Exp. 2b: pares con controles
│   └── exp3_logical_consistency.py       # Exp. 3: entidades + LLM few-shot
│
├── requirements-exp1.txt                 # Dependencias del Exp. 1
├── requirements-exp2.txt                 # Dependencias de Exp. 2 y 2b
├── requirements-exp3.txt                 # Dependencias del Exp. 3
│
├── data_analysis.py                      # EDA original del TP
├── test_pytorch.py                       # Chequeo de PyTorch/CUDA
│
├── graphs/                               # Generado por los scripts (gitignored)
│   ├── chart_*.html                      # EDA original
│   ├── exp1/
│   ├── exp2/
│   ├── exp2b/
│   └── exp3/
│
├── results/                              # Generado por los scripts (gitignored)
│   ├── exp1/
│   ├── exp2/
│   ├── exp2b/
│   └── exp3/
│
├── README.md
└── Results.md                            # Resultados de la corrida de referencia
```

Cada experimento escribe únicamente en su propio subdirectorio de `graphs/` y
`results/`. Los datasets **no se guardan en el repositorio**: se descargan a la
caché de HuggingFace (`~/.cache/huggingface` por defecto).

`graphs/` y `results/` están en `.gitignore`, al igual que los CSV, los
checkpoints grandes y los entornos virtuales. Se pueden regenerar corriendo los
scripts.

---

## Datasets

Los experimentos usan dos datasets públicos de HuggingFace.

### Mu-SHROOM

- **Nombre:** [`Helsinki-NLP/mu-shroom`](https://huggingface.co/datasets/Helsinki-NLP/mu-shroom)
- **Configuración usada:** `en` (inglés)
- **Splits de la configuración `en`:**
  - `train_unlabeled`: 809 documentos sin etiquetas de spans.
  - `validation`: 50 documentos etiquetados.
  - `test`: 154 documentos etiquetados.
- Los experimentos usan `validation + test` = **204 documentos etiquetados**.
  197 de ellos tienen al menos un span de alucinación.

**Columnas principales:**

| Columna | Contenido |
|---|---|
| `id` | Identificador del documento |
| `lang` | Idioma (`EN` en la configuración usada) |
| `model_input` | Pregunta o instrucción dada al modelo |
| `model_output_text` | Respuesta generada por el modelo |
| `model_id` | Modelo que generó la respuesta |
| `wikipedia_url` | Artículo de referencia |
| `soft_labels` | Etiquetas suaves por token/carácter con probabilidad |
| `hard_labels` | Lista de spans `[start, end)` a nivel **carácter** marcados como alucinación |
| `model_output_logits` | Logits del modelo generador |
| `model_output_tokens` | Tokens del texto generado |
| `annotations` | Anotaciones individuales de cada anotador |

**Uso en los experimentos:**

- **Exp. 1:** los `hard_labels` permiten separar cada respuesta en segmentos
  alucinados (Corpus A) y veraces (Corpus B).
- **Exp. 2:** los documentos con al menos un span se usan como **positivos**.

### Phantom Hallucination Detection

- **Nombre:** [`seyled/Phantom_Hallucination_Detection`](https://huggingface.co/datasets/seyled/Phantom_Hallucination_Detection)
- **Split:** `train` (único), **30.886 filas**.
- **Balance:** 15.443 filas con `ground_truth_label == "hallucination"` y 15.443
  con `"not hallucination"`.

**Columnas:**

| Columna | Contenido |
|---|---|
| `Unnamed: 0` | Índice original de la fila |
| `query` | Pregunta |
| `context` | Documento de contexto (fragmento de un filing financiero, ~miles de palabras) |
| `answer` | Respuesta generada |
| `ground_truth_label` | `"hallucination"` o `"not hallucination"` |

**Características relevantes:**

- Está construido con **pares**: para cada `(query, context)` hay exactamente una
  respuesta limpia y una alucinada. Hay 15.443 pares de este tipo. Esto permite
  controlar el tema en el Experimento 2b.
- Las respuestas se repiten mucho entre preguntas, por eso los scripts
  deduplican por texto normalizado antes de muestrear.
- Hay un **confound de longitud fuerte**: en el dataset completo, la mediana de
  palabras es **25** para respuestas limpias y **67** para alucinadas. Los
  experimentos lo reportan y lo controlan explícitamente.

**Uso en los experimentos:**

- **Exp. 2:** las respuestas `"not hallucination"` se usan como **negativos**.
- **Exp. 2b:** se forman pares limpio/alucinado de la misma `(query, context)`.
- **Exp. 3:** se muestrea un subconjunto balanceado de 100 + 100 respuestas.

### Datasets construidos por experimento

| Experimento | Construcción | Tamaño de la corrida de referencia |
|---|---|---|
| 1 | Segmentos A (alucinados) y B (veraces) extraídos de Mu-SHROOM | 609 segmentos A + 599 B; 797 usados en BoW/TF-IDF |
| 2 | 197 positivos de Mu-SHROOM + 197 negativos únicos de Phantom | 394 ejemplos balanceados 50/50 |
| 2b | Pares limpio/alucinado de Phantom, mismo tema, misma longitud y sin puntuación | 324 pares = 648 ejemplos |
| 3 | Muestra balanceada de Phantom, deduplicada por `(query, context, answer)` | 200 ejemplos (100 + 100) |

---

## Experimentos

### Experimento 1 — Análisis léxico y de perplejidad

**Script:** `experiments/exp1_lexical_perplexity.py`

Compara, dentro de las respuestas de Mu-SHROOM, los segmentos alucinados (Corpus
A) contra los veraces (Corpus B) desde dos perspectivas:

1. **Léxico:** se ejecuta el **mismo pipeline** con **Bag-of-Words**
   (`CountVectorizer`) y **TF-IDF** (`TfidfVectorizer`): split agrupado por
   documento (`GroupShuffleSplit` por `doc_id`) para evitar leakage y
   `LogisticRegression` con `class_weight="balanced"`. Para cada enfoque se
   reportan en test: F1 macro, MCC, precisión y recall por clase, matriz de
   confusión y el **Top 15 de palabras** por valor absoluto del coeficiente
   (con su valor y la dirección: Corpus A o Corpus B).
2. **Perplejidad:** GPT-2 calcula la NLL sobre el texto original completo. La NLL
   se separa según si cada token cae dentro o fuera de un span alucinado. Se
   reportan la **NLL media** y la **perplejidad media** de cada corpus, más un
   **t-test apareado** a nivel documento (estadístico t y p-valor).

Decisiones metodológicas:

- Los offsets de `hard_labels` son `[start, end)` a nivel carácter.
- Los spans se normalizan (clamp, orden, merge de solapamientos y adyacencias).
- BoW y TF-IDF usan exactamente el mismo conjunto de segmentos y el mismo split
  train/test.
- Un token se asigna al Corpus A si al menos el 50 % de sus caracteres cae
  dentro de un span alucinado; si no, va al Corpus B.
- Se descartan tokens sin caracteres alfanuméricos.
- La perplejidad usa el contexto original, no un texto aislado.

Opciones útiles:

```bash
python experiments/exp1_lexical_perplexity.py --help

# Otro modelo causal
python experiments/exp1_lexical_perplexity.py --lm gpt2-medium

# Forzar CPU
python experiments/exp1_lexical_perplexity.py --device cpu

# Elegir splits de Mu-SHROOM
python experiments/exp1_lexical_perplexity.py --splits validation test

# Correr solo la parte léxica (sin descargar el LM)
python experiments/exp1_lexical_perplexity.py --skip-perplexity
```

Salidas:

```text
graphs/exp1/
  exp1_lexical_metrics.html
  exp1_top_features_bow.html
  exp1_top_features_tfidf.html
  exp1_perplexity_boxplot.html
  exp1_perplexity_delta_histogram.html

results/exp1/
  exp1_segments.csv
  exp1_top_features_bow.csv
  exp1_top_features_tfidf.csv
  exp1_perplexity_per_document.csv
  exp1_summary.json
```

---

### Experimento 2 — Clasificación binaria

**Script:** `experiments/exp2_binary_classification.py`

Construye un dataset balanceado 50/50 combinando dos fuentes:

- **Positivos:** todos los documentos ingleses de Mu-SHROOM
  (`validation + test`) con al menos un span en `hard_labels`.
- **Negativos:** respuestas únicas de Phantom con
  `ground_truth_label == "not hallucination"`. Phantom se deduplica por texto
  normalizado antes de muestrear porque repite muchísimo la misma respuesta.

Compara:

1. Baseline léxico: **TF-IDF + LogisticRegression**.
2. Fine-tuning semántico: **DistilBERT** (`distilbert-base-uncased` por defecto).

El split es estratificado 70/15/15. El baseline y el transformer usan
exactamente el mismo train y el mismo test; la validación se usa para elegir la
mejor época.

Métricas: F1 macro, F1 por clase, precisión, recall, MCC, ROC-AUC, PR-AUC y
matriz de confusión. **No se calcula ni se reporta la métrica global de
aciertos**, y no se usa `classification_report`.

> **Advertencia metodológica:** en este diseño la etiqueta está perfectamente
> confundida con la fuente (positivos = Mu-SHROOM, negativos = Phantom). Un F1
> alto puede reflejar separación de tema, estilo o longitud, no detección
> genuina de alucinaciones. El script imprime esta advertencia, exporta los
> términos más asociados a cada fuente y guarda la limitación en
> `exp2_summary.json`. El Experimento 2b está diseñado justamente para eliminar
> ese confound.

Opciones útiles:

```bash
python experiments/exp2_binary_classification.py --help

# BERT base en vez de DistilBERT
python experiments/exp2_binary_classification.py --model bert-base-uncased

# GPU con fp16
python experiments/exp2_binary_classification.py --fp16

# Guardar el modelo fine-tuneado
python experiments/exp2_binary_classification.py --save-model

# Forzar CPU
python experiments/exp2_binary_classification.py --cpu
```

Salidas:

```text
graphs/exp2/
  exp2_class_balance.html
  exp2_answer_length_distribution.html
  exp2_baseline_top_features.html
  exp2_confusion_baseline.html
  exp2_confusion_transformer.html
  exp2_transformer_training_curves.html
  exp2_model_comparison.html
  exp2_probability_distribution.html

results/exp2/
  exp2_balanced_dataset.csv
  exp2_baseline_metrics.json
  exp2_baseline_predictions.csv
  exp2_baseline_predictions_validation.csv
  exp2_baseline_top_features.csv
  exp2_transformer_metrics.json
  exp2_transformer_predictions.csv
  exp2_transformer_training_log.csv
  exp2_metrics_comparison.csv
  exp2_summary.json
  transformer_checkpoints/            # si se entrena el transformer
```

---

### Experimento 2b — Phantom pareado: control de fuente, tema y longitud

**Script:** `experiments/exp2b_phantom_paired_control.py`

Es la versión controlada del Experimento 2. Usa **solo Phantom** para eliminar
el confound fuente-etiqueta y construye **pares dentro de la misma
`(query, context)`**: una respuesta limpia y una alucinada del mismo tema.

Controles aplicados:

1. **Fuente:** todas las filas vienen de Phantom.
2. **Tema:** los dos textos del par responden la misma pregunta con el mismo
   contexto.
3. **Longitud:** solo se conservan pares donde ambas respuestas tienen al menos
   `--target-words` tokens (40 por defecto) y las dos se recortan exactamente a
   esa cantidad. La longitud es idéntica dentro de cada par.
4. **Puntuación:** a los textos recortados se les quita la puntuación. Sin este
   paso el recorte deja una pista trivial: la respuesta alucinada suele ser más
   larga y al cortarla termina a mitad de oración, mientras que la limpia
   termina en punto.
5. **Duplicados:** se descartan los pares con Jaccard ≥ `--max-jaccard` (0.98),
   porque si los dos textos son casi iguales la etiqueta es ambigua por
   construcción.

El split es agrupado por `query` con `StratifiedGroupKFold`: los dos miembros de
un par y todas las respuestas de una misma pregunta caen siempre en la misma
partición.

Compara:

1. **TF-IDF + LogisticRegression** sobre el texto recortado.
2. **DistilBERT** fine-tuneado sobre el texto recortado.
3. **Umbral de longitud** (control trivial).
4. **Query-only** (diagnóstico de fuga por tema).
5. **Regla de puntuación** (diagnóstico del artefacto de recorte).

Además reporta dos referencias para dimensionar cada confound:

- **Sin control:** muestra balanceada de respuestas completas, sin parear.
- **Tema controlado, longitud no:** los mismos pares con el texto completo.

También incluye un análisis de sensibilidad a la longitud de recorte
(`--sensitivity-lengths`, por defecto `25,40,60`).

Métricas: F1 macro, F1 por clase, precisión, recall, MCC, ROC-AUC, PR-AUC y
matriz de confusión. **No se calcula ni se reporta la métrica global de
aciertos.**

Opciones útiles:

```bash
python experiments/exp2b_phantom_paired_control.py --help

# Otro recorte de longitud (tokens por respuesta)
python experiments/exp2b_phantom_paired_control.py --target-words 25

# Ser más estricto con los pares casi idénticos
python experiments/exp2b_phantom_paired_control.py --max-jaccard 0.95

# Sin referencias de ablación (más rápido)
python experiments/exp2b_phantom_paired_control.py --skip-reference

# Otra grilla de sensibilidad
python experiments/exp2b_phantom_paired_control.py --sensitivity-lengths "30,50"
```

Salidas:

```text
graphs/exp2b/
  exp2b_split_balance.html
  exp2b_length_control.html
  exp2b_pair_overlap.html
  exp2b_top_features.html
  exp2b_confusion_baseline.html
  exp2b_confusion_transformer.html
  exp2b_transformer_training_curves.html
  exp2b_probability_distribution.html
  exp2b_artifact_diagnostic.html
  exp2b_control_ablation.html

results/exp2b/
  exp2b_balanced_pairs.csv
  exp2b_pair_diagnostics.csv
  exp2b_baseline_predictions.csv
  exp2b_baseline_predictions_validation.csv
  exp2b_transformer_predictions.csv
  exp2b_transformer_training_log.csv
  exp2b_length_control_predictions.csv
  exp2b_query_only_predictions.csv
  exp2b_metrics.csv
  exp2b_control_ablation.csv
  exp2b_artifact_diagnostic.csv
  exp2b_length_sensitivity.csv
  exp2b_top_features.csv
  exp2b_summary.json
  transformer_checkpoints/            # si se entrena el transformer
```

---

### Experimento 3 — Consistencia lógica: entidades vs. LLM few-shot

**Script:** `experiments/exp3_logical_consistency.py`

Sobre una muestra balanceada de Phantom (100 alucinaciones + 100 respuestas
limpias = 200 filas), compara:

1. **Baseline de entidades con spaCy:** extrae entidades de la `query` y del
   `context` con `en_core_web_sm`, calcula qué fracción de las entidades de la
   respuesta no está respaldada por la evidencia y elige un umbral heurístico
   **solo en desarrollo**.
2. **LLM few-shot:** un modelo instruct recibe `query`, un contexto recuperado y
   `answer`, junto con 4 ejemplos etiquetados, y responde `HALLUCINATION` o
   `CLEAN`.
3. **Control de longitud:** un umbral sobre la cantidad de palabras de la
   respuesta. No es un método propuesto; sirve para dimensionar el confound.

Métricas: F1 macro, F1 de alucinación, precisión y recall por clase, MCC,
matriz de confusión, IC bootstrap al 95 % y test de McNemar entre baseline y
LLM.

Decisiones de diseño:

- Split **50/25/25** (desarrollo/validación/test) agrupado por `query` con
  `StratifiedGroupKFold`.
- La muestra se deduplica por la terna `(query, context, answer)` antes de
  muestrear.
- El umbral del baseline se elige solo en desarrollo.
- El LLM no se entrena: los ejemplos few-shot salen de desarrollo.
- El modelo por defecto es `Qwen/Qwen2.5-3B-Instruct` (no gated, ~6 GB en bf16).
- El contexto que recibe el LLM se recupera con TF-IDF sobre las oraciones del
  contexto, para no truncar arbitrariamente un documento de ~9.000 palabras.

> **Advertencia metodológica:** las respuestas alucinadas de Phantom son
> sistemáticamente más largas que las limpias. El control de longitud es una
> referencia trivial que puede igualar o superar a los métodos evaluados, así
> que el resultado debe interpretarse como separabilidad
> superficial/verbosidad, no como detección semántica genuina. El gráfico
> `exp3_score_vs_length.html` visualiza este confound.

Opciones útiles:

```bash
python experiments/exp3_logical_consistency.py --help

# LLM más chico y rápido
python experiments/exp3_logical_consistency.py \
  --model Qwen/Qwen2.5-1.5B-Instruct

# LLM más grande (puede requerir 4-bit y batch 1)
python experiments/exp3_logical_consistency.py \
  --model Qwen/Qwen2.5-7B-Instruct --batch-size 1

# Gemma (gated: hay que aceptar la licencia en HuggingFace)
python experiments/exp3_logical_consistency.py \
  --model google/gemma-1.1-2b-it --hf-token "$HF_TOKEN"

# Si la GPU se queda sin memoria
python experiments/exp3_logical_consistency.py \
  --batch-size 1 --max-input-tokens 3072

# Forzar CPU (muy lento)
python experiments/exp3_logical_consistency.py --cpu
```

Salidas:

```text
graphs/exp3/
  exp3_split_balance.html
  exp3_answer_length_distribution.html
  exp3_entity_grounding.html
  exp3_threshold_curve.html
  exp3_confusion_baseline.html
  exp3_confusion_llm.html
  exp3_model_comparison.html
  exp3_score_vs_length.html
  exp3_entity_variants.html

results/exp3/
  exp3_sample.csv
  exp3_predictions_test.csv
  exp3_predictions_validation.csv
  exp3_metrics.csv
  exp3_threshold_search.csv
  exp3_length_threshold_search.csv
  exp3_entity_variants.csv
  exp3_llm_raw_predictions.csv
  exp3_llm_few_shot_examples.csv
  exp3_errors.csv
  exp3_summary.json
```

---

## Gráficos y exportación

Los gráficos se guardan como HTML interactivo de Altair y se abren directo en el
navegador:

```bash
xdg-open graphs/exp3/exp3_model_comparison.html
```

Para exportar también PNG, instalá `vl-convert-python` y agregá `--save-png`:

```bash
pip install vl-convert-python
python experiments/exp3_logical_consistency.py --save-png
```

---

## Reproducibilidad

- Todos los scripts usan `--seed 42` por defecto.
- Los splits y muestreos son deterministas; el Experimento 3 también fija sus
  remuestreos bootstrap con la semilla.
- Los experimentos 2, 2b y 3 desactivan `cudnn.benchmark` y fijan
  `torch.backends.cudnn.deterministic`.
- El LLM usa decodificación greedy (`do_sample=False`).
- Los datasets y modelos se pueden fijar con `--cache-dir`, pero las versiones
  descargadas dependen de HuggingFace.
- Las métricas pueden variar levemente por diferencias de hardware, versión de
  `transformers` o actualizaciones de los datasets.

---

## Solución de problemas

**`OSError: [E050] Can't find model 'en_core_web_sm'`**

```bash
python -m spacy download en_core_web_sm
```

**El modelo de HuggingFace está gated (por ejemplo, Gemma)**

```bash
export HF_TOKEN="hf_..."
# o: huggingface-cli login
python experiments/exp3_logical_consistency.py \
  --model google/gemma-1.1-2b-it --hf-token "$HF_TOKEN"
```

**CUDA out of memory en el Experimento 3**

```bash
python experiments/exp3_logical_consistency.py \
  --batch-size 1 --max-input-tokens 3072
# o un modelo más chico
python experiments/exp3_logical_consistency.py \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

**Las descargas de HuggingFace son lentas o fallan por rate limit**

Definí `HF_TOKEN` antes de correr el script. También se puede limpiar la caché:

```bash
rm -rf ~/.cache/huggingface/hub
```

**Los gráficos no abren o pesan demasiado**

Los HTML de Altair incluyen los datos embebidos. Si el navegador se pone lento,
probá abrirlos con `--save-png` o reducí `--max-examples` / `--max-pairs`.

**Quiero ver todas las opciones de un experimento**

```bash
python experiments/exp1_lexical_perplexity.py --help
python experiments/exp2_binary_classification.py --help
python experiments/exp2b_phantom_paired_control.py --help
python experiments/exp3_logical_consistency.py --help
```
