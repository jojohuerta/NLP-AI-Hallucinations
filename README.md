# NLP-AI-Hallucinations

Proyecto de Procesamiento del Lenguaje Natural (ITBA) sobre **detección de
alucinaciones en modelos de lenguaje**.

El repositorio contiene tres experimentos independientes. Cada uno es un script
autocontenido, con semilla fija, métricas reproducibles y gráficos Altair.
Los experimentos se pueden correr por separado: no dependen entre sí.

---

## Estructura

```text
experiments/
  exp1_lexical_perplexity.py       # Exp. 1: léxico + perplejidad
  exp2_binary_classification.py    # Exp. 2: clasificación binaria
  exp2b_phantom_paired_control.py  # Exp. 2b: control de fuente, tema y longitud
  exp3_logical_consistency.py      # Exp. 3: entidades + LLM few-shot

requirements-exp1.txt
requirements-exp2.txt
requirements-exp3.txt

graphs/                            # generado por los scripts (gitignored)
  exp1/
  exp2/
  exp2b/
  exp3/

results/                           # generado por los scripts (gitignored)
  exp1/
  exp2/
  exp2b/
  exp3/

data_analysis.py                   # EDA original del TP
```

Cada experimento escribe únicamente en su propio subdirectorio de `graphs/` y
`results/`, así que se pueden ejecutar en cualquier orden.

---

## Requisitos generales

Probado con **Python 3.12** y GPU NVIDIA (RTX 3080 Ti, 12 GB). Los experimentos
1 y 3 pueden correr en CPU, aunque más lento.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Después instalá las dependencias del experimento que quieras correr:

```bash
pip install -r requirements-exp1.txt
pip install -r requirements-exp2.txt
pip install -r requirements-exp3.txt
```

Los datasets se descargan automáticamente de HuggingFace la primera vez:

- [Helsinki-NLP/mu-shroom](https://huggingface.co/datasets/Helsinki-NLP/mu-shroom)
- [seyled/Phantom_Hallucination_Detection](https://huggingface.co/datasets/seyled/Phantom_Hallucination_Detection)

Si HuggingFace limita las descargas, definí un token:

```bash
export HF_TOKEN="hf_..."
# o: huggingface-cli login
```

---

## Experimento 1 — Análisis léxico y de perplejidad

**Script:** `experiments/exp1_lexical_perplexity.py`

Compara, dentro de las respuestas de **Mu-SHROOM** (configuración `en`), los
segmentos marcados como alucinación contra los segmentos veraces:

1. **Léxico:** TF-IDF + LogisticRegression sobre segmentos, con split agrupado
   por documento para evitar leakage y bootstrap agrupado para los coeficientes.
2. **Perplejidad:** GPT-2 sobre el texto original; se calcula la NLL solamente
   sobre los tokens que caen dentro y fuera de los spans alucinados, y se aplica
   un t-test apareado a nivel documento.

### Instalación

```bash
pip install -r requirements-exp1.txt
```

### Ejecución completa

```bash
python experiments/exp1_lexical_perplexity.py
```

La primera corrida descarga Mu-SHROOM y GPT-2 (~500 MB).

### Smoke tests

```bash
# Solo la parte léxica (no descarga GPT-2)
python experiments/exp1_lexical_perplexity.py --skip-perplexity

# Pocos documentos
python experiments/exp1_lexical_perplexity.py \
  --max-examples 20 --skip-perplexity
```

### Opciones útiles

```bash
python experiments/exp1_lexical_perplexity.py --help

# Otro modelo causal
python experiments/exp1_lexical_perplexity.py --lm gpt2-medium

# Forzar CPU
python experiments/exp1_lexical_perplexity.py --device cpu
```

### Salidas

```text
graphs/exp1/*.html
results/exp1/*.csv
results/exp1/*.json
```

### Resultados de referencia

Con la configuración por defecto (204 documentos, 609 spans A y 599 spans B):

```text
TF-IDF + LogReg        F1 macro = 0.6491
t-test de coeficientes p = 4.73e-4
GPT-2 perplexity       A = 49.81  vs.  B = 49.70
t-test apareado        p = 0.985 (no significativo)
```

---

## Experimento 2 — Clasificación binaria

**Script:** `experiments/exp2_binary_classification.py`

Construye un dataset balanceado 50/50:

- **Positivos:** todos los documentos ingleses de **Mu-SHROOM**
  (`validation + test`) con al menos un span en `hard_labels` → 197 ejemplos.
- **Negativos:** 197 respuestas únicas de **Phantom** con
  `ground_truth_label == "not hallucination"`. Phantom se deduplica antes de
  muestrear porque repite muchísimo la misma respuesta.

Compara:

1. Baseline léxico: **TF-IDF + LogisticRegression**.
2. Fine-tuning semántico: **DistilBERT**.

Métricas: F1 macro, F1 por clase, precisión, recall, MCC, ROC-AUC, PR-AUC y
matriz de confusión. **No se calcula ni se reporta la métrica global de
aciertos**, y no se usa `classification_report`.

### Instalación

```bash
pip install -r requirements-exp2.txt
```

### Ejecución completa

```bash
python experiments/exp2_binary_classification.py
```

La primera corrida descarga Mu-SHROOM, Phantom y DistilBERT (~250 MB).

### Smoke tests

```bash
# Solo baseline, sin transformer
python experiments/exp2_binary_classification.py --skip-transformer

# Pocos ejemplos y una época
python experiments/exp2_binary_classification.py \
  --max-per-class 20 --epochs 1 --batch-size 8
```

### Opciones útiles

```bash
python experiments/exp2_binary_classification.py --help

# BERT base en vez de DistilBERT
python experiments/exp2_binary_classification.py --model bert-base-uncased

# GPU con fp16
python experiments/exp2_binary_classification.py --fp16

# Guardar el modelo fine-tuneado
python experiments/exp2_binary_classification.py --save-model
```

### Salidas

```text
graphs/exp2/*.html
results/exp2/*.csv
results/exp2/*.json
results/exp2/transformer_checkpoints/   # ~257 MB, gitignored
```

### Resultados de referencia

```text
TF-IDF + LogReg   F1 macro = 0.9833   MCC = 0.9672
DistilBERT        F1 macro = 1.0000   MCC = 1.0000
Truncados        7/394 (1.8 %) con max_length = 256
```

> **Advertencia metodológica:** en este diseño la etiqueta está perfectamente
> confundida con la fuente (positivos = Mu-SHROOM, negativos = Phantom). Un F1
> casi perfecto puede reflejar separación de tema, estilo o longitud, no
> detección genuina de alucinaciones. El script imprime esta advertencia,
> exporta los términos más asociados a cada fuente (`exp2_baseline_top_features.html`)
> y guarda la limitación en `exp2_summary.json`.

---

## Experimento 2b — Phantom pareado: control de fuente, tema y longitud

**Script:** `experiments/exp2b_phantom_paired_control.py`

Es la versión justa del Experimento 2. Usa **solo Phantom** para eliminar el
confound fuente-etiqueta y construye **pares dentro de la misma
`(query, context)`**: una respuesta limpia y una alucinada del mismo tema.

Para no cambiar un confound por otro, aplica tres controles:

1. **Tema:** los dos textos del par responden la misma pregunta con el mismo
   contexto.
2. **Longitud:** solo se conservan pares donde ambas respuestas tienen al menos
   `--target-words` tokens (40 por defecto) y las dos se recortan exactamente a
   esa cantidad. La longitud es idéntica dentro de cada par, no solo en
   promedio.
3. **Puntuación:** a los textos recortados se les quita la puntuación. Sin este
   paso el recorte deja una pista trivial: la respuesta alucinada suele ser más
   larga y al cortarla termina a mitad de oración, mientras que la limpia
   termina en punto.

También se descartan los pares con Jaccard ≥ `--max-jaccard` (0.98), porque si
los dos textos son casi iguales la etiqueta es ambigua por construcción.

El split es agrupado por `query` (`StratifiedGroupKFold`): los dos miembros de
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

Métricas: F1 macro, F1 por clase, precisión, recall, MCC, ROC-AUC, PR-AUC y
matriz de confusión. **No se calcula ni se reporta la métrica global de
aciertos**, y no se usa `classification_report`.

### Instalación

Usa las mismas dependencias que el Experimento 2:

```bash
pip install -r requirements-exp2.txt
```

### Ejecución completa

```bash
python experiments/exp2b_phantom_paired_control.py
```

La primera corrida descarga Phantom y DistilBERT (~250 MB).

### Smoke tests

```bash
# Solo baselines y controles, sin transformer
python experiments/exp2b_phantom_paired_control.py --skip-transformer

# Pocos pares y dos longitudes de sensibilidad
python experiments/exp2b_phantom_paired_control.py \
  --max-pairs 60 --skip-transformer --sensitivity-lengths "25,40"
```

### Opciones útiles

```bash
python experiments/exp2b_phantom_paired_control.py --help

# Otro recorte de longitud (tokens por respuesta)
python experiments/exp2b_phantom_paired_control.py --target-words 25

# Ser más estricto con los pares casi idénticos
python experiments/exp2b_phantom_paired_control.py --max-jaccard 0.95

# Sin referencias de ablación (más rápido)
python experiments/exp2b_phantom_paired_control.py --skip-reference
```

### Salidas

```text
graphs/exp2b/*.html
results/exp2b/*.csv
results/exp2b/*.json
results/exp2b/transformer_checkpoints/   # gitignored
```

### Resultados de referencia

Con `target_words = 40`, 324 pares (648 ejemplos): train 464, validación 92,
test 92.

```text
Condición                        Modelo              F1 macro   MCC
Sin control                      TF-IDF              0.8571     0.7160
Sin control                      Umbral de longitud  0.9003     0.8187
Tema controlado (texto completo) TF-IDF              0.6934     0.4535
Tema controlado (texto completo) Umbral de longitud  0.6675     0.4154
Tema + longitud (diseño final)   TF-IDF              0.5939     0.1995
Tema + longitud (diseño final)   Umbral de longitud  0.3333     0.0000
Tema + longitud (diseño final)   Query-only          0.3333     0.0000
Tema + longitud (diseño final)   DistilBERT          0.6157     0.2835
```

Un predictor constante sobre un test balanceado obtiene F1 macro = 1/3 y
MCC = 0, así que **0.3333 es el piso de azar** de estas tablas.

Sensibilidad al recorte (solo TF-IDF):

```text
target_words   pares   F1 macro   MCC
25             299     0.5204     0.0765
40             324     0.5939     0.1995
60             264     0.6878     0.3819
```

> **El artefacto que motivó el rediseño.** Un primer intento recortaba cada par
> a `min(palabras_limpias, palabras_alucinadas)`. Como en 1.059 de 1.082 pares
> la alucinación era más larga, el recorte la dejaba cortada a mitad de oración.
> Una regla trivial —"si termina en punto, es limpia"— alcanzaba **F1 = 0.924**
> y DistilBERT llegaba a 0.9415 explotando esa pista, no detectando
> alucinaciones. El diseño actual conserva ese diagnóstico: sobre el recorte
> ingenuo la regla de puntuación da **F1 = 0.9459** (el conjunto de pares del
> diagnóstico difiere levemente del primer intento), mientras que sobre el texto
> final cae a 0.3333. El gráfico `exp2b_artifact_diagnostic.html` compara ambos
> mundos lado a lado.

Lectura: al controlar fuente, tema, longitud y puntuación, el rendimiento cae
de ~0.86 a ~0.60. Queda señal por encima del azar (TF-IDF 0.59, DistilBERT
0.62), pero muy lejos del F1 casi perfecto del Experimento 2. La conclusión es
que buena parte de ese 0.98 era separación de fuente, tema y verbosidad.

Limitaciones: el test tiene 92 ejemplos (intervalos amplios), solo se conservan
pares donde ambas respuestas son largas, y el recorte puede eliminar la cola
donde aparece la alucinación. Todo queda registrado en `exp2b_summary.json`.

---

## Experimento 3 — Consistencia lógica: entidades vs. LLM few-shot

**Script:** `experiments/exp3_logical_consistency.py`

Sobre una muestra balanceada de **Phantom** (100 alucinaciones + 100 respuestas
limpias = 200 filas), compara:

1. **Baseline de entidades con spaCy:** extrae entidades de la `query` y del
   `context` con `en_core_web_sm`, calcula qué fracción de las entidades de la
   respuesta no está respaldada por la evidencia y elige un umbral heurístico
   **solo en desarrollo**.
2. **LLM few-shot:** un modelo instruct recibe `query`, un contexto recuperado
   y `answer`, junto con 4 ejemplos etiquetados, y responde `HALLUCINATION` o
   `CLEAN`.
3. **Control de longitud:** un umbral sobre la cantidad de palabras de la
   respuesta. No es un método propuesto; sirve para dimensionar el confound.

Métricas: F1 macro, F1 de alucinación, precisión y recall por clase, MCC,
matriz de confusión, IC bootstrap al 95 % y test de McNemar entre baseline y LLM.

Decisiones de diseño:

- Split **50/25/25** (desarrollo/validación/test) agrupado por `query` con
  `StratifiedGroupKFold`. Phantom tiene pares de respuestas (limpia/alucinada)
  para la misma pregunta y contexto; agrupar evita leakage entre particiones.
- El modelo por defecto es `Qwen/Qwen2.5-3B-Instruct` (no gated, ~6 GB en
  bf16). Se puede cambiar con `--model`.
- El contexto que recibe el LLM se recupera con TF-IDF sobre las oraciones del
  contexto, para no truncar arbitrariamente un documento de ~9.000 palabras.

### Instalación

```bash
pip install -r requirements-exp3.txt

# Si la rueda del modelo de spaCy no se instaló:
python -m spacy download en_core_web_sm
```

### Ejecución completa

```bash
python experiments/exp3_logical_consistency.py
```

La primera corrida descarga Phantom y el LLM (~6 GB). En una RTX 3080 Ti tardó
aproximadamente 4 minutos en total (la mayor parte, extracción de entidades).

### Smoke tests

```bash
# Solo baseline de entidades + control de longitud (sin descargar LLM)
python experiments/exp3_logical_consistency.py --skip-llm

# Muestra chica, sin LLM
python experiments/exp3_logical_consistency.py \
  --n-per-class 20 --skip-llm
```

### Opciones útiles

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
```

### Salidas

```text
graphs/exp3/*.html

results/exp3/exp3_sample.csv
results/exp3/exp3_predictions_test.csv
results/exp3/exp3_predictions_validation.csv
results/exp3/exp3_metrics.csv
results/exp3/exp3_threshold_search.csv
results/exp3/exp3_length_threshold_search.csv
results/exp3/exp3_entity_variants.csv
results/exp3/exp3_llm_raw_predictions.csv
results/exp3/exp3_errors.csv
results/exp3/exp3_summary.json
```

### Resultados de referencia

Split: 100 dev / 50 validación / 50 test (25 + 25 en test).

```text
Entidades (baseline)   F1 macro = 0.6528   MCC = 0.4345
LLM few-shot           F1 macro = 0.6753   MCC = 0.3708
Control de longitud    F1 macro = 0.8397   MCC = 0.6822
McNemar baseline vs. LLM: p = 1.0000
```

> **Advertencia metodológica:** las respuestas alucinadas de Phantom son
> sistemáticamente más largas que las limpias (mediana 71 vs. 24 palabras en
> la muestra). El control de longitud supera a ambos métodos, así que el
> resultado debe interpretarse como **separabilidad superficial/verbosidad**,
> no como detección semántica genuina. El script reporta esta limitación y el
> gráfico `exp3_score_vs_length.html` la visualiza.

---

## Ejecución individual: resumen

| Experimento | Instalar | Correr | Salidas |
|---|---|---|---|
| 1 | `pip install -r requirements-exp1.txt` | `python experiments/exp1_lexical_perplexity.py` | `graphs/exp1/`, `results/exp1/` |
| 2 | `pip install -r requirements-exp2.txt` | `python experiments/exp2_binary_classification.py` | `graphs/exp2/`, `results/exp2/` |
| 2b | `pip install -r requirements-exp2.txt` | `python experiments/exp2b_phantom_paired_control.py` | `graphs/exp2b/`, `results/exp2b/` |
| 3 | `pip install -r requirements-exp3.txt` | `python experiments/exp3_logical_consistency.py` | `graphs/exp3/`, `results/exp3/` |

Cada script imprime al final la lista de archivos generados y acepta `--help`
con todas sus opciones.

---

## Gráficos y exportación

Los gráficos se guardan como HTML interactivo de Altair y se abren directo en
el navegador:

```bash
xdg-open graphs/exp3/exp3_model_comparison.html
```

Para exportar también PNG:

```bash
pip install vl-convert-python
python experiments/exp3_logical_consistency.py --save-png
```

Los directorios `graphs/` y `results/` están en `.gitignore`, igual que los
`.csv` y los checkpoints grandes. Los scripts se pueden volver a correr en
cualquier momento para regenerarlos.

---

## Reproducibilidad

- Todos los scripts usan `--seed 42` por defecto.
- Los splits, muestreos y remuestreos bootstrap son deterministas.
- Los experimentos 2 y 3 desactivan `cudnn.benchmark` y fijan
  `torch.backends.cudnn.deterministic`.
- El LLM usa decodificación greedy (`do_sample=False`).
- Las métricas de referencia pueden variar levemente por diferencias de
  hardware, versión de `transformers` o actualizaciones de los datasets.

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
