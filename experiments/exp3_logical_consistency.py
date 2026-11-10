#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 3 — Consistencia lógica: entidades vs. LLM few-shot
================================================================

Objetivo
--------
Sobre una muestra balanceada de Phantom_Hallucination_Detection:

1. **Baseline de entidades (spaCy)**: se extraen entidades de la ``query`` y del
   ``context`` con ``en_core_web_sm``. Para cada respuesta se calcula qué
   fracción de sus entidades NO está respaldada por la evidencia (query +
   contexto). Un umbral heurístico, elegido en el conjunto de desarrollo,
   decide si la respuesta es alucinación.
2. **LLM few-shot**: un modelo instruct pequeño recibe ``query``, un contexto
   recuperado y ``answer``, junto con ejemplos etiquetados, y debe responder
   ``HALLUCINATION`` o ``CLEAN``.
3. **Control de longitud**: un umbral sobre la cantidad de palabras de la
   respuesta. No es un método propuesto: sirve para dimensionar el confound
   más importante del dataset.

Métricas principales
--------------------
* F1 macro.
* F1 de la clase "alucinación".
* Precisión y recall por clase.
* MCC (Matthews Correlation Coefficient).
* Matriz de confusión.
* Intervalos de confianza bootstrap al 95 % para F1 macro y MCC.
* Test de McNemar entre baseline y LLM.

Decisiones metodológicas
------------------------
* La muestra es balanceada 50/50 (``--n-per-class``, por defecto 100 por clase).
* La muestra se deduplica por la terna (query, context, answer) antes de
  muestrear.
* El split es agrupado por ``query`` con ``StratifiedGroupKFold``. Esto evita
  que una misma pregunta/contexto aparezca en dev y test con respuestas
  distintas: Phantom está construido con pares (respuesta limpia / respuesta
  alucinada) para la misma pregunta.
* El umbral del baseline se elige SOLO en desarrollo (50 %). Validación (25 %)
  y test (25 %) quedan para evaluar.
* El LLM no se entrena: los ejemplos few-shot salen de desarrollo y el test no
  se usa para elegir nada.
* Todos los procesos usan semilla fija (``--seed``).

Advertencia sobre el dataset
----------------------------
Phantom tiene un confound de longitud muy fuerte: las respuestas alucinadas
son, en promedio, mucho más largas que las limpias. Por eso el script reporta
un **control de longitud** además del baseline y del LLM. Si el control de
longitud rinde parecido a los métodos "reales", el resultado debe interpretarse
como separabilidad superficial y no como detección genuina de alucinaciones.

Salidas
-------
``graphs/exp3/``  : gráficos Altair en HTML (y PNG opcional).
``results/exp3/`` : CSV, JSON, predicciones y errores.

Uso
---
Ver el bloque de instrucciones al final del archivo o ejecutar ``--help``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import altair as alt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

SEED: int = 42
DATASET_NAME: str = "seyled/Phantom_Hallucination_Detection"
DEFAULT_SPACY_MODEL: str = "en_core_web_sm"
DEFAULT_LLM: str = "Qwen/Qwen2.5-3B-Instruct"

COLOR_HALLUCINATION = "#E45756"  # rojo
COLOR_TRUTHFUL = "#54A24B"       # verde
COLOR_NEUTRAL = "#4C78A8"        # azul
COLOR_LLM = "#B279A2"            # violeta
COLOR_LENGTH = "#F2CF5B"         # amarillo

LABEL_NAMES = {0: "Sin alucinación", 1: "Alucinación"}
SPLIT_NAMES = {"dev": "Desarrollo", "validation": "Validación", "test": "Test"}

NUMBER_RE = re.compile(r"\b\d[\d,.]*%?")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
alt.data_transformers.disable_max_rows()


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fija las semillas de random, NumPy y PyTorch (si está instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def project_root() -> Path:
    """Raíz del repositorio, independientemente del cwd."""
    return Path(__file__).resolve().parent.parent


def json_default(value: Any) -> Any:
    """Serializa tipos de NumPy/Pandas a JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def normalize_phrase(text: Any) -> str:
    """Minúsculas, sin puntuación, espacios colapsados."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_key(text: Any) -> str:
    """Normalización agresiva para deduplicar."""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def count_words(text: Any) -> int:
    return len(str(text).split())


def phrase_in_text(phrase: str, text_norm_padded: str) -> bool:
    """Chequeo de frase completa sobre un texto ya normalizado y con espacios."""
    if not phrase:
        return False
    return f" {phrase} " in text_norm_padded


def number_key(value: str) -> str:
    """Normaliza números para comparar 41 %, 41.0 y 41 como equivalentes."""
    cleaned = str(value).replace(",", "").rstrip(".")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return f"{float(cleaned):g}"
    except ValueError:
        return cleaned


def number_keys(text: Any) -> set[str]:
    return {number_key(match) for match in NUMBER_RE.findall(str(text))}


def split_sentences(text: Any, min_words: int = 5) -> list[str]:
    """Segmentación simple por puntuación, suficiente para recuperar contexto."""
    raw = re.split(r"(?<=[.!?])\s+", str(text))
    return [sentence.strip() for sentence in raw if count_words(sentence) >= min_words]


# ---------------------------------------------------------------------------
# 1. Carga y construcción de la muestra balanceada
# ---------------------------------------------------------------------------

def load_phantom_sample(
    n_per_class: int = 100,
    seed: int = SEED,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carga Phantom, deduplica terneras y muestrea ``n_per_class`` por clase."""
    ds = load_dataset(DATASET_NAME, cache_dir=cache_dir)
    split = "train" if "train" in ds else list(ds.keys())[0]
    raw = ds[split].to_pandas().copy()

    required = {"query", "context", "answer", "ground_truth_label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas en Phantom: {sorted(missing)}")

    raw["query"] = raw["query"].fillna("").astype(str)
    raw["context"] = raw["context"].fillna("").astype(str)
    raw["answer"] = raw["answer"].fillna("").astype(str)
    raw["label_text"] = raw["ground_truth_label"].astype(str).str.strip().str.lower()

    valid_labels = {"hallucination", "not hallucination"}
    raw = raw[raw["label_text"].isin(valid_labels)].copy()

    # Deduplicar por la terna completa antes de muestrear.
    raw["dedup_key"] = (
        raw["query"].map(normalize_text_key)
        + " || "
        + raw["context"].map(normalize_text_key)
        + " || "
        + raw["answer"].map(normalize_text_key)
    )
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["dedup_key"], keep="first").copy()

    n_available = raw["label_text"].value_counts().to_dict()
    for label in valid_labels:
        if n_available.get(label, 0) < n_per_class:
            raise ValueError(
                f"No hay suficientes ejemplos de '{label}': "
                f"pedidos {n_per_class}, disponibles {n_available.get(label, 0)}."
            )

    positives = raw[raw["label_text"] == "hallucination"].sample(
        n=n_per_class, random_state=seed
    )
    negatives = raw[raw["label_text"] == "not hallucination"].sample(
        n=n_per_class, random_state=seed
    )
    df = pd.concat([positives, negatives], ignore_index=True)

    df["label"] = (df["label_text"] == "hallucination").astype(int)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["n_words"] = df["answer"].map(count_words)
    df["group"] = df["query"].map(normalize_text_key)

    # Protección: una misma respuesta no puede estar en ambas clases.
    df["answer_key"] = df["answer"].map(normalize_text_key)
    leaked = df.groupby("answer_key")["label"].nunique().gt(1).sum()
    if leaked:
        print(
            f"  [aviso] {leaked} respuestas aparecen en ambas clases; "
            "se eliminan para evitar leakage."
        )
        df = df.sort_values("label").drop_duplicates(subset=["answer_key"], keep="first")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = df["label"].value_counts().to_dict()
    if counts.get(0, 0) != counts.get(1, 0):
        raise RuntimeError(f"La muestra no quedó balanceada: {counts}")

    metadata = {
        "dataset": DATASET_NAME,
        "raw_rows": int(n_before),
        "rows_after_dedup": int(len(raw)),
        "n_per_class": int(counts[1]),
        "n_total": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "label_counts": {LABEL_NAMES[k]: int(v) for k, v in counts.items()},
    }
    return df.reset_index(drop=True), metadata


def split_dataset_grouped(
    df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 50/25/25 agrupado por query, estratificado por etiqueta.

    ``StratifiedGroupKFold`` con 4 folds garantiza que una misma query (y por
    lo tanto el mismo contexto) no aparezca en dos particiones distintas.
    Se usan dos folds como desarrollo (50 %), uno como validación (25 %) y uno
    como test (25 %).
    """
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    folds = list(splitter.split(df, df["label"], df["group"]))
    if len(folds) != 4:
        raise RuntimeError("Se esperaban 4 folds.")

    dev_idx = np.concatenate([folds[0][1], folds[1][1]])
    val_idx = folds[2][1]
    test_idx = folds[3][1]

    dev_df = df.iloc[dev_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    dev_df["split"] = "dev"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    # Verificación explícita: ningún grupo puede repetirse entre particiones.
    groups = {
        "dev": set(dev_df["group"]),
        "validation": set(val_df["group"]),
        "test": set(test_df["group"]),
    }
    for a, b in (("dev", "validation"), ("dev", "test"), ("validation", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise RuntimeError(
                f"Leakage de grupos entre {a} y {b}: {len(overlap)} queries repetidas."
            )

    return (
        dev_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 2. Features de entidades con spaCy
# ---------------------------------------------------------------------------

def load_spacy(model_name: str = DEFAULT_SPACY_MODEL) -> Any:
    """Carga spaCy desactivando componentes que no se usan."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy no está instalado. Instalalo con:\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        ) from exc

    try:
        return spacy.load(
            model_name,
            disable=["parser", "lemmatizer", "attribute_ruler"],
        )
    except OSError as exc:
        raise RuntimeError(
            f"No se pudo cargar el modelo de spaCy '{model_name}'.\n"
            "Instalalo con:\n"
            "  python -m spacy download en_core_web_sm"
        ) from exc


def entity_texts(nlp: Any, texts: Iterable[str], batch_size: int = 16) -> list[list[str]]:
    """Extrae las entidades de una lista de textos de forma eficiente en memoria."""
    output: list[list[str]] = []
    for doc in nlp.pipe(list(texts), batch_size=batch_size):
        output.append([ent.text for ent in doc.ents])
    return output


def fuzzy_in_entity_set(
    phrase: str,
    entity_set: set[str],
    entity_token_sets: list[set[str]],
    max_fuzzy_entities: int = 4000,
) -> bool:
    """Match exacto o aproximado (Jaccard >= 0.8) contra las entidades de evidencia."""
    if not phrase:
        return False
    if phrase in entity_set:
        return True
    if len(entity_set) > max_fuzzy_entities:
        return False

    phrase_tokens = set(phrase.split())
    if not phrase_tokens:
        return False

    for tokens in entity_token_sets:
        intersection = len(phrase_tokens & tokens)
        if not intersection:
            continue
        union = len(phrase_tokens | tokens)
        if union and intersection / union >= 0.8:
            return True
    return False


def compute_entity_features(
    df: pd.DataFrame,
    nlp: Any,
    max_context_chars: int = 60_000,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Calcula features de grounding de entidades para cada respuesta.

    Columnas principales:
    * ``n_unsupported_entities`` / ``unsupported_entity_rate``: entidades de la
      respuesta que no aparecen en la evidencia (query + contexto).
    * ``n_unsupported_numbers`` / ``unsupported_number_rate``: ídem para
      números, fechas y porcentajes.
    * ``query_entity_coverage``: fracción de entidades de la query que la
      respuesta menciona. Es una señal de solapamiento, no de grounding.
    """
    answers = df["answer"].astype(str).tolist()
    queries = df["query"].astype(str).tolist()
    contexts = df["context"].astype(str).tolist()

    answer_entities = entity_texts(nlp, answers, batch_size=batch_size)
    query_entities = entity_texts(nlp, queries, batch_size=batch_size)
    context_entities = entity_texts(
        nlp,
        (context[:max_context_chars] for context in contexts),
        batch_size=max(1, batch_size // 4),
    )

    records: list[dict[str, Any]] = []
    for i, answer in enumerate(answers):
        answer_norm = normalize_phrase(answer)
        answer_norm_padded = f" {answer_norm} "

        evidence_text = normalize_phrase(queries[i] + " " + contexts[i])
        evidence_norm_padded = f" {evidence_text} "

        # Entidades de evidencia: query + contexto truncado para NER.
        evidence_entities = [
            normalize_phrase(entity)
            for entity in (query_entities[i] + context_entities[i])
        ]
        evidence_entities = [entity for entity in evidence_entities if entity]
        evidence_set = set(evidence_entities)
        evidence_token_sets = [set(entity.split()) for entity in evidence_entities]

        answer_ents = [normalize_phrase(entity) for entity in answer_entities[i]]
        answer_ents = [entity for entity in answer_ents if len(entity) > 1]

        supported = 0
        for entity in answer_ents:
            if phrase_in_text(entity, evidence_norm_padded) or fuzzy_in_entity_set(
                entity, evidence_set, evidence_token_sets
            ):
                supported += 1

        n_answer_entities = len(answer_ents)
        n_unsupported_entities = n_answer_entities - supported

        answer_numbers = number_keys(answer)
        evidence_numbers = number_keys(queries[i] + " " + contexts[i])
        unsupported_numbers = answer_numbers - evidence_numbers

        query_ents = [normalize_phrase(entity) for entity in query_entities[i]]
        query_ents = [entity for entity in query_ents if len(entity) > 1]
        if query_ents:
            covered = sum(phrase_in_text(entity, answer_norm_padded) for entity in query_ents)
            query_entity_coverage = covered / len(query_ents)
        else:
            query_entity_coverage = 0.0

        records.append(
            {
                "n_answer_entities": n_answer_entities,
                "n_supported_entities": supported,
                "n_unsupported_entities": n_unsupported_entities,
                "unsupported_entity_rate": (
                    n_unsupported_entities / n_answer_entities
                    if n_answer_entities
                    else 0.0
                ),
                "n_answer_numbers": len(answer_numbers),
                "n_unsupported_numbers": len(unsupported_numbers),
                "unsupported_number_rate": (
                    len(unsupported_numbers) / len(answer_numbers)
                    if answer_numbers
                    else 0.0
                ),
                "n_query_entities": len(query_ents),
                "query_entity_coverage": query_entity_coverage,
                "n_words": count_words(answer),
            }
        )

    return pd.DataFrame(records, index=df.index)


# ---------------------------------------------------------------------------
# 3. Umbrales, métricas e incertidumbre
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    n_grid: int = 200,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Elige el umbral que maximiza F1 macro en el conjunto de desarrollo.

    La regla de decisión es ``score >= threshold``. Se exploran cuantiles del
    score más ``+inf`` (nunca predecir positivo). Los empates se rompen por MCC
    y, si persiste, por un umbral más alto (regla más conservadora).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("Todos los scores son no finitos.")

    if np.unique(finite).size == 1:
        candidates = np.array([finite[0], np.inf])
    else:
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, n_grid))
        candidates = np.unique(np.concatenate([quantiles, finite + 1e-9, [np.inf]]))

    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1_macro": float(
                    f1_score(y_true, predictions, average="macro", zero_division=0)
                ),
                "mcc": float(matthews_corrcoef(y_true, predictions)),
                "n_pred_positive": int(predictions.sum()),
                "n_pred_negative": int((1 - predictions).sum()),
            }
        )

    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        ["f1_macro", "mcc", "threshold"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = grid.iloc[0].to_dict()
    return best, grid


def binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """F1 macro, F1 por clase, precisión, recall, MCC y matriz de confusión."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        "n": int(len(y_true)),
        "n_negative": int((y_true == 0).sum()),
        "n_positive": int((y_true == 1).sum()),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_hallucination": float(f1[1]),
        "f1_no_hallucination": float(f1[0]),
        "precision_hallucination": float(precision[1]),
        "recall_hallucination": float(recall[1]),
        "precision_no_hallucination": float(precision[0]),
        "recall_no_hallucination": float(recall[0]),
        "support_hallucination": int(support[1]),
        "support_no_hallucination": int(support[0]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def safe_roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_boot: int = 1000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """IC percentil bootstrap para F1 macro y MCC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0 or n_boot <= 0:
        return {
            "f1_macro_ci_low": float("nan"),
            "f1_macro_ci_high": float("nan"),
            "mcc_ci_low": float("nan"),
            "mcc_ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    f1_values = np.empty(n_boot, dtype=float)
    mcc_values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_values[i] = f1_score(
            y_true[idx], y_pred[idx], average="macro", zero_division=0
        )
        mcc_values[i] = matthews_corrcoef(y_true[idx], y_pred[idx])

    lower = 100 * (alpha / 2)
    upper = 100 * (1 - alpha / 2)
    return {
        "f1_macro_ci_low": float(np.percentile(f1_values, lower)),
        "f1_macro_ci_high": float(np.percentile(f1_values, upper)),
        "mcc_ci_low": float(np.percentile(mcc_values, lower)),
        "mcc_ci_high": float(np.percentile(mcc_values, upper)),
    }


def mcnemar_test(
    y_true: Sequence[int],
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
) -> dict[str, Any]:
    """Test exacto de McNemar sobre los pares discordantes."""
    y_true = np.asarray(y_true, dtype=int)
    predictions_a = np.asarray(predictions_a, dtype=int)
    predictions_b = np.asarray(predictions_b, dtype=int)

    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(only_a, discordant, 0.5).pvalue)

    return {
        "a_correct_b_wrong": only_a,
        "b_correct_a_wrong": only_b,
        "discordant": discordant,
        "p_value": p_value,
    }


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    cm = metrics["confusion_matrix"]
    return pd.DataFrame(
        [
            ("F1 macro", f"{metrics['f1_macro']:.4f}"),
            ("F1 alucinación", f"{metrics['f1_hallucination']:.4f}"),
            ("F1 sin alucinación", f"{metrics['f1_no_hallucination']:.4f}"),
            ("Precisión alucinación", f"{metrics['precision_hallucination']:.4f}"),
            ("Recall alucinación", f"{metrics['recall_hallucination']:.4f}"),
            ("Precisión sin alucinación", f"{metrics['precision_no_hallucination']:.4f}"),
            ("Recall sin alucinación", f"{metrics['recall_no_hallucination']:.4f}"),
            ("MCC", f"{metrics['mcc']:.4f}"),
            ("Matriz de confusión", f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"),
        ],
        columns=["Métrica", "Valor"],
    )


# ---------------------------------------------------------------------------
# 4. Recuperación de contexto y LLM few-shot
# ---------------------------------------------------------------------------

def retrieve_context(
    context: str,
    query: str,
    top_k: int = 10,
) -> tuple[str, int]:
    """Recupera las ``top_k`` oraciones del contexto más similares a la query.

    Se usa TF-IDF sobre las oraciones del contexto. Las oraciones se devuelven
    en el orden original para que el texto sea coherente. Si el contexto es
    corto, se devuelve completo.
    """
    sentences = split_sentences(context)
    if not sentences:
        return str(context)[:4000], 0
    if top_k <= 0 or len(sentences) <= top_k:
        return " ".join(sentences), len(sentences)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
        query_vector = vectorizer.transform([query])
        scores = (matrix @ query_vector.T).toarray().ravel()
    except ValueError:
        return " ".join(sentences[:top_k]), top_k

    selected = np.argsort(scores)[::-1][:top_k]
    selected = sorted(selected.tolist())
    return " ".join(sentences[i] for i in selected), len(selected)


def build_few_shot_examples(
    dev_df: pd.DataFrame,
    seed: int = SEED,
    n_per_class: int = 2,
    top_k_context: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    """Selecciona ejemplos few-shot balanceados y representativos de dev.

    Se prefieren alucinaciones con al menos una entidad no respaldada y
    respuestas limpias sin entidades no respaldadas. Si no alcanzan, se
    completa con muestreo aleatorio reproducible.
    """
    frame = dev_df.copy()
    rng = np.random.default_rng(seed)

    positives = frame[
        (frame["label"] == 1) & (frame["n_unsupported_entities"] >= 2)
    ]
    negatives = frame[
        (frame["label"] == 0) & (frame["n_unsupported_entities"] == 0)
    ]

    if len(positives) < n_per_class:
        positives = frame[frame["label"] == 1]
    if len(negatives) < n_per_class:
        negatives = frame[frame["label"] == 0]

    def pick(pool: pd.DataFrame) -> pd.DataFrame:
        if len(pool) <= n_per_class:
            return pool
        indices = rng.choice(pool.index.to_numpy(), size=n_per_class, replace=False)
        return pool.loc[indices]

    selected = pd.concat([pick(positives), pick(negatives)])
    selected = selected.sort_index()

    blocks: list[str] = []
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        context, n_sentences = retrieve_context(
            row["context"], row["query"], top_k=top_k_context
        )
        label = "HALLUCINATION" if int(row["label"]) == 1 else "CLEAN"
        blocks.append(
            f"### Example {position}\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{row['query']}\n\n"
            f"ANSWER:\n{row['answer']}\n\n"
            f"Label: {label}"
        )
        records.append(
            {
                "uid": int(row["uid"]) if "uid" in row.index else None,
                "source_index": int(row.name),
                "label": int(row["label"]),
                "n_context_sentences": int(n_sentences),
            }
        )

    return "\n\n".join(blocks), records


def build_llm_messages(
    system_prompt: str,
    few_shot_block: str,
    context: str,
    query: str,
    answer: str,
) -> list[dict[str, str]]:
    user_prompt = (
        f"{few_shot_block}\n\n"
        "### Your turn\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Label:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def resolve_torch_device(use_cpu: bool = False) -> Any:
    import torch

    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(device: Any) -> Any:
    import torch

    if device.type == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_model_dtype_kwarg() -> str:
    """``dtype`` en transformers 5.x, ``torch_dtype`` en 4.x."""
    import transformers

    try:
        major = int(transformers.__version__.split(".")[0])
    except (AttributeError, ValueError):
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


def load_llm(
    model_name: str,
    device: Any,
    hf_token: str | None = None,
) -> tuple[Any, Any]:
    """Carga tokenizer y modelo causal en el dispositivo elegido."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers no está instalado. Instalalo con:\n"
            "  pip install transformers accelerate torch"
        ) from exc

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    token_kwargs = {"token": hf_token} if hf_token else {}
    dtype = resolve_torch_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {
        resolve_model_dtype_kwarg(): dtype,
        **token_kwargs,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "cuda"
    else:
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return tokenizer, model


def fit_prompt_to_budget(
    tokenizer: Any,
    system_prompt: str,
    few_shot_block: str,
    row: pd.Series,
    top_k_context: int,
    max_input_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Construye el prompt reduciendo el contexto hasta entrar en el presupuesto.

    Nunca se trunca la respuesta ni la query: si hiciera falta, se recorta el
    contexto. Esto evita cortar justamente la parte que se quiere evaluar.
    """
    k = int(top_k_context)
    context, n_sentences = retrieve_context(row["context"], row["query"], top_k=k)

    for _ in range(max(0, k)):
        messages = build_llm_messages(
            system_prompt,
            few_shot_block,
            context,
            str(row["query"]),
            str(row["answer"]),
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens <= max_input_tokens:
            return messages, {
                "n_prompt_tokens": int(n_tokens),
                "n_context_sentences": int(n_sentences),
                "context_truncated": False,
            }

        k -= 1
        if k <= 0:
            break
        context, n_sentences = retrieve_context(row["context"], row["query"], top_k=k)

    # Último recurso: recortar caracteres del contexto.
    messages = build_llm_messages(
        system_prompt,
        few_shot_block,
        context,
        str(row["query"]),
        str(row["answer"]),
    )
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])

    while n_tokens > max_input_tokens and len(context) > 500:
        context = context[: int(len(context) * 0.9)]
        messages = build_llm_messages(
            system_prompt,
            few_shot_block,
            context,
            str(row["query"]),
            str(row["answer"]),
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return messages, {
        "n_prompt_tokens": int(n_tokens),
        "n_context_sentences": int(n_sentences),
        "context_truncated": True,
    }


def parse_llm_output(raw: str) -> tuple[int, bool]:
    """Convierte la salida del LLM en 0/1. Devuelve (predicción, válida)."""
    upper = str(raw).upper()
    if "HALLUCIN" in upper or "NOT SUPPORTED" in upper or "UNSUPPORTED" in upper:
        return 1, True
    if "CLEAN" in upper or "SUPPORTED" in upper:
        return 0, True
    return 0, False


def run_llm_predictions(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    system_prompt: str,
    few_shot_block: str,
    device: Any,
    top_k_context: int = 10,
    max_input_tokens: int = 6144,
    batch_size: int = 8,
    max_new_tokens: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Genera predicciones del LLM en batches."""
    import torch

    records: list[dict[str, Any]] = []
    started = time.time()
    total = len(frame)

    for start in range(0, total, batch_size):
        batch = frame.iloc[start : start + batch_size]
        prompts: list[str] = []
        batch_meta: list[dict[str, Any]] = []

        for _, row in batch.iterrows():
            messages, meta = fit_prompt_to_budget(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                few_shot_block=few_shot_block,
                row=row,
                top_k_context=top_k_context,
                max_input_tokens=max_input_tokens,
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            batch_meta.append(meta)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA se quedó sin memoria durante la generación. Probá con:\n"
                "  --batch-size 1 --max-input-tokens 3072\n"
                "o usá un modelo más chico:\n"
                "  --model Qwen/Qwen2.5-1.5B-Instruct"
            ) from exc

        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_length:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for row_index, raw, meta in zip(batch.index, decoded, batch_meta):
            prediction, valid = parse_llm_output(raw)
            records.append(
                {
                    "row_index": int(row_index),
                    "llm_raw": str(raw).strip(),
                    "llm_pred": prediction,
                    "llm_valid": bool(valid),
                    **meta,
                }
            )

        done = min(start + len(batch), total)
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"  LLM: {done}/{total} ({rate:.2f} filas/s)")

    predictions = pd.DataFrame(records).set_index("row_index")
    metadata = {
        "n_rows": int(total),
        "n_invalid": int((~predictions["llm_valid"]).sum()),
        "n_context_truncated": int(predictions["context_truncated"].sum()),
        "prompt_tokens_min": int(predictions["n_prompt_tokens"].min()),
        "prompt_tokens_median": float(predictions["n_prompt_tokens"].median()),
        "prompt_tokens_max": int(predictions["n_prompt_tokens"].max()),
        "elapsed_seconds": float(time.time() - started),
    }
    return predictions, metadata


# ---------------------------------------------------------------------------
# 5. Gráficos
# ---------------------------------------------------------------------------

def save_chart(chart: Any, stem: str, graphs_dir: Path, save_png: bool = False) -> Path:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    html_path = graphs_dir / f"{stem}.html"
    chart.save(html_path)
    if save_png:
        try:
            chart.save(graphs_dir / f"{stem}.png", scale_factor=2.0)
        except Exception as exc:
            print(
                f"  [aviso] No se pudo guardar {stem}.png ({exc}). "
                "Instalá 'vl-convert-python' para exportar PNG."
            )
    return html_path


def build_split_balance_chart(df: pd.DataFrame) -> alt.Chart:
    plot_df = df.copy()
    plot_df["split_name"] = plot_df["split"].map(SPLIT_NAMES)
    counts = (
        plot_df.groupby(["split_name", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return (
        alt.Chart(counts)
        .mark_bar(size=48, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "split_name:N",
                title=None,
                sort=list(SPLIT_NAMES.values()),
            ),
            xOffset=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
            y=alt.Y("count:Q", title="Cantidad de ejemplos"),
            color=alt.Color(
                "label_name:N",
                title="Clase",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
            ),
            tooltip=[
                alt.Tooltip("split_name:N", title="Split"),
                alt.Tooltip("label_name:N", title="Clase"),
                alt.Tooltip("count:Q", title="Ejemplos"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Exp. 3 — Muestra balanceada y split agrupado por query",
                subtitle="Split 50/25/25 (dev/validación/test); ninguna query se repite entre particiones",
            ),
            width=460,
            height=320,
        )
    )


def build_length_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    rng = np.random.default_rng(SEED)
    plot_df = df[["label_name", "n_words", "query"]].copy()
    plot_df["query_preview"] = plot_df["query"].astype(str).str.slice(0, 110)
    plot_df = plot_df.drop(columns=["query"])
    plot_df["jitter"] = rng.uniform(-0.25, 0.25, size=len(plot_df))

    box = (
        alt.Chart(plot_df)
        .mark_boxplot(extent="min-max", size=44, median={"color": "black"})
        .encode(
            x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
            y=alt.Y(
                "n_words:Q",
                title="Palabras por respuesta (escala log)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color(
                "label_name:N",
                title="Clase",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
            ),
        )
    )
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=18, opacity=0.35)
        .encode(
            x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
            xOffset=alt.X("jitter:Q", scale=alt.Scale(domain=[-0.35, 0.35])),
            y=alt.Y("n_words:Q"),
            color=alt.Color(
                "label_name:N",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("query_preview:N", title="Query"),
                alt.Tooltip("label_name:N", title="Clase"),
                alt.Tooltip("n_words:Q", title="Palabras"),
            ],
        )
    )
    return (box + points).properties(
        title=alt.TitleParams(
            text="Exp. 3 — Longitud de las respuestas por clase",
            subtitle="Confound principal: las alucinaciones son mucho más largas que las respuestas limpias",
        ),
        width=460,
        height=360,
    )


def build_entity_risk_chart(df: pd.DataFrame) -> alt.Chart:
    rng = np.random.default_rng(SEED)
    plot_df = df[
        [
            "label_name",
            "unsupported_entity_rate",
            "n_answer_entities",
            "n_unsupported_entities",
        ]
    ].copy()
    plot_df["jitter"] = rng.uniform(-0.25, 0.25, size=len(plot_df))

    box = (
        alt.Chart(plot_df)
        .mark_boxplot(extent="min-max", size=42, median={"color": "black"})
        .encode(
            x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
            y=alt.Y(
                "unsupported_entity_rate:Q",
                title="Fracción de entidades no respaldadas",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "label_name:N",
                title="Clase",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
            ),
        )
    )
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=16, opacity=0.30)
        .encode(
            x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
            xOffset=alt.X("jitter:Q", scale=alt.Scale(domain=[-0.35, 0.35])),
            y=alt.Y("unsupported_entity_rate:Q"),
            color=alt.Color(
                "label_name:N",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("n_answer_entities:Q", title="Entidades en respuesta"),
                alt.Tooltip("n_unsupported_entities:Q", title="No respaldadas"),
                alt.Tooltip("unsupported_entity_rate:Q", format=".2f", title="Fracción"),
            ],
        )
    )
    return (box + points).properties(
        title=alt.TitleParams(
            text="Exp. 3 — Grounding de entidades por clase",
            subtitle="El baseline etiqueta como alucinación cuando la fracción no respaldada supera un umbral",
        ),
        width=460,
        height=360,
    )


def build_threshold_curve_chart(
    grid: pd.DataFrame,
    best_threshold: float,
    score_name: str = "unsupported_entity_rate",
) -> alt.Chart:
    plot_df = grid[grid["threshold"] < np.inf].copy()
    if plot_df.empty:
        plot_df = grid.head(1).copy()

    max_threshold = float(plot_df["threshold"].max())
    if np.isfinite(max_threshold) and max_threshold > 0:
        plot_df = plot_df[plot_df["threshold"] <= np.quantile(plot_df["threshold"], 0.98)]
    plot_df = plot_df.sort_values("threshold").reset_index(drop=True)

    threshold_for_rule = (
        float(best_threshold) if np.isfinite(best_threshold) else max_threshold
    )

    long_df = plot_df.melt(
        id_vars=["threshold"],
        value_vars=["f1_macro", "mcc"],
        var_name="metrica",
        value_name="valor",
    )
    long_df["metrica"] = long_df["metrica"].map({"f1_macro": "F1 macro", "mcc": "MCC"})

    lines = (
        alt.Chart(long_df)
        .mark_line(point=False, strokeWidth=2.5)
        .encode(
            x=alt.X("threshold:Q", title=f"Umbral sobre {score_name}"),
            y=alt.Y("valor:Q", title="Valor en desarrollo", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metrica:N",
                title="Métrica",
                scale=alt.Scale(domain=["F1 macro", "MCC"], range=[COLOR_NEUTRAL, COLOR_LENGTH]),
            ),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".4f", title="Umbral"),
                alt.Tooltip("metrica:N", title="Métrica"),
                alt.Tooltip("valor:Q", format=".4f", title="Valor"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [threshold_for_rule]}))
        .mark_rule(color=COLOR_HALLUCINATION, strokeDash=[5, 4], strokeWidth=2)
        .encode(
            x="x:Q",
            tooltip=[alt.Tooltip("x:Q", format=".4f", title="Umbral elegido")],
        )
    )
    return (lines + rule).properties(
        title=alt.TitleParams(
            text="Exp. 3 — Búsqueda del umbral del baseline",
            subtitle=f"El umbral se elige solo en desarrollo: {threshold_for_rule:.4f}",
        ),
        width=560,
        height=340,
    )


def build_confusion_matrix_chart(metrics: dict[str, Any], model_label: str) -> alt.Chart:
    cm = metrics["confusion_matrix"]
    records = []
    for true_label in (0, 1):
        for pred_label in (0, 1):
            key = {(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[
                (true_label, pred_label)
            ]
            records.append(
                {
                    "real": LABEL_NAMES[true_label],
                    "prediccion": LABEL_NAMES[pred_label],
                    "casos": cm[key],
                    "tipo": key.upper(),
                }
            )
    cm_df = pd.DataFrame(records)
    max_count = max(1, int(cm_df["casos"].max()))
    order = list(LABEL_NAMES.values())

    base = alt.Chart(cm_df).encode(
        x=alt.X("prediccion:N", title="Predicción", sort=order),
        y=alt.Y("real:N", title="Etiqueta real", sort=order),
    )
    heatmap = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("casos:Q", title="Casos", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip("real:N", title="Real"),
            alt.Tooltip("prediccion:N", title="Predicción"),
            alt.Tooltip("casos:Q", title="Casos"),
            alt.Tooltip("tipo:N", title="Celda"),
        ],
    )
    labels = base.mark_text(fontSize=20, fontWeight="bold").encode(
        text=alt.Text("casos:Q"),
        color=alt.condition(
            alt.datum.casos > max_count / 2,
            alt.value("white"),
            alt.value("black"),
        ),
    )
    return (heatmap + labels).properties(
        title=alt.TitleParams(
            text=f"Exp. 3 — Matriz de confusión · {model_label}",
            subtitle=(
                f"F1 macro={metrics['f1_macro']:.4f} · MCC={metrics['mcc']:.4f} · "
                f"n={metrics['n']}"
            ),
        ),
        width=380,
        height=320,
    )


def build_model_comparison_chart(metrics_by_model: dict[str, dict[str, Any]]) -> alt.Chart:
    records = []
    for model_label, metrics in metrics_by_model.items():
        for metric_label, key in (
            ("F1 macro", "f1_macro"),
            ("F1 alucinación", "f1_hallucination"),
            ("MCC", "mcc"),
        ):
            records.append(
                {"modelo": model_label, "metrica": metric_label, "valor": metrics[key]}
            )
    plot_df = pd.DataFrame(records)

    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=34, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("modelo:N", title=None),
            xOffset=alt.X(
                "metrica:N",
                title=None,
                sort=["F1 macro", "F1 alucinación", "MCC"],
            ),
            y=alt.Y("valor:Q", title="Valor de la métrica", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metrica:N",
                title="Métrica",
                scale=alt.Scale(
                    domain=["F1 macro", "F1 alucinación", "MCC"],
                    range=[COLOR_NEUTRAL, COLOR_HALLUCINATION, COLOR_LENGTH],
                ),
            ),
            tooltip=[
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("metrica:N", title="Métrica"),
                alt.Tooltip("valor:Q", format=".4f", title="Valor"),
            ],
        )
    )
    text = bars.mark_text(dy=-8, fontSize=10).encode(
        text=alt.Text("valor:Q", format=".3f"),
        color=alt.value("#222222"),
    )
    return (bars + text).properties(
        title=alt.TitleParams(
            text="Exp. 3 — Entidades vs. LLM few-shot vs. control de longitud",
            subtitle="Test agrupado por query; el control de longitud es una referencia, no un método propuesto",
        ),
        width=560,
        height=380,
    )


def build_score_vs_length_chart(df: pd.DataFrame) -> alt.Chart:
    plot_df = df[
        [
            "label_name",
            "n_words",
            "unsupported_entity_rate",
            "n_unsupported_entities",
            "query",
        ]
    ].copy()
    plot_df["query_preview"] = plot_df["query"].astype(str).str.slice(0, 110)
    plot_df = plot_df.drop(columns=["query"])
    return (
        alt.Chart(plot_df)
        .mark_circle(size=45, opacity=0.65, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "n_words:Q",
                title="Palabras de la respuesta (escala log)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "unsupported_entity_rate:Q",
                title="Fracción de entidades no respaldadas",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "label_name:N",
                title="Clase",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
            ),
            tooltip=[
                alt.Tooltip("query_preview:N", title="Query"),
                alt.Tooltip("label_name:N", title="Clase"),
                alt.Tooltip("n_words:Q", title="Palabras"),
                alt.Tooltip("n_unsupported_entities:Q", title="Entidades no respaldadas"),
                alt.Tooltip("unsupported_entity_rate:Q", format=".2f", title="Fracción"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Exp. 3 — Grounding de entidades vs. longitud",
                subtitle="Si las clases se separan por longitud, el detector puede estar midiendo verbosidad",
            ),
            width=600,
            height=360,
        )
    )


def build_entity_variants_chart(variants: pd.DataFrame) -> alt.Chart:
    plot_df = variants.copy()
    plot_df = plot_df[plot_df["split"].isin(["dev", "test"])]
    plot_df["split"] = plot_df["split"].map({"dev": "Desarrollo", "test": "Test"})
    return (
        alt.Chart(plot_df)
        .mark_bar(size=20, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            y=alt.Y(
                "variant:N",
                title=None,
                sort=alt.SortField(field="variant_order", order="ascending"),
            ),
            x=alt.X("auc:Q", title="ROC-AUC", scale=alt.Scale(domain=[0, 1])),
            yOffset=alt.Y("split:N", title="Split"),
            color=alt.Color(
                "split:N",
                title="Split",
                scale=alt.Scale(domain=["Desarrollo", "Test"], range=[COLOR_NEUTRAL, COLOR_LLM]),
            ),
            tooltip=[
                alt.Tooltip("variant:N", title="Score"),
                alt.Tooltip("split:N", title="Split"),
                alt.Tooltip("auc:Q", format=".3f", title="ROC-AUC"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Exp. 3 — Poder discriminativo de cada score de entidades",
                subtitle="Ningún score de entidades supera claramente al azar; la longitud sí lo hace",
            ),
            width=560,
            height=300,
        )
    )


# ---------------------------------------------------------------------------
# 6. Evaluación y exportación
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    split_name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: Sequence[float] | None = None,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    metrics = binary_metrics(y_true, y_pred)
    metrics.update(
        {
            "model": model_name,
            "split": split_name,
            "roc_auc": (
                safe_roc_auc(y_true, scores) if scores is not None else float("nan")
            ),
        }
    )
    metrics.update(bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed))
    return metrics


def metrics_row(metrics: dict[str, Any]) -> dict[str, Any]:
    cm = metrics["confusion_matrix"]
    return {
        "model": metrics["model"],
        "split": metrics["split"],
        "n": metrics["n"],
        "f1_macro": metrics["f1_macro"],
        "f1_hallucination": metrics["f1_hallucination"],
        "f1_no_hallucination": metrics["f1_no_hallucination"],
        "precision_hallucination": metrics["precision_hallucination"],
        "recall_hallucination": metrics["recall_hallucination"],
        "precision_no_hallucination": metrics["precision_no_hallucination"],
        "recall_no_hallucination": metrics["recall_no_hallucination"],
        "mcc": metrics["mcc"],
        "roc_auc": metrics["roc_auc"],
        "f1_macro_ci_low": metrics["f1_macro_ci_low"],
        "f1_macro_ci_high": metrics["f1_macro_ci_high"],
        "mcc_ci_low": metrics["mcc_ci_low"],
        "mcc_ci_high": metrics["mcc_ci_high"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
    }


def build_predictions_frame(
    frame: pd.DataFrame,
    score: np.ndarray,
    prediction: np.ndarray,
    prefix: str,
    llm_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye un DataFrame de predicciones para test o validación."""
    columns = [
        "uid",
        "split",
        "query",
        "answer",
        "label",
        "label_name",
        "n_words",
        "n_answer_entities",
        "n_unsupported_entities",
        "unsupported_entity_rate",
        "n_unsupported_numbers",
        "unsupported_number_rate",
        "query_entity_coverage",
    ]
    available = [column for column in columns if column in frame.columns]
    output = frame[available].copy().reset_index(drop=True)
    output[f"{prefix}_score"] = np.asarray(score)
    output[f"{prefix}_pred"] = np.asarray(prediction, dtype=int)
    output["answer_preview"] = frame["answer"].astype(str).str.slice(0, 400).to_numpy()

    if llm_predictions is not None:
        llm_columns = [
            "uid",
            "llm_raw",
            "llm_pred",
            "llm_valid",
            "n_prompt_tokens",
            "n_context_sentences",
        ]
        present = [column for column in llm_columns if column in llm_predictions.columns]
        output = output.merge(
            llm_predictions[present], on="uid", how="left", validate="one_to_one"
        )

    return output


def export_results(
    df: pd.DataFrame,
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline: dict[str, Any],
    length_control: dict[str, Any],
    variants: pd.DataFrame,
    llm_result: dict[str, Any] | None,
    metadata: dict[str, Any],
    config: dict[str, Any],
    graphs_dir: Path,
    results_dir: Path,
    save_png: bool,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # --- Predicciones ------------------------------------------------------
    test_predictions = build_predictions_frame(
        test_df,
        score=baseline["test_scores"],
        prediction=baseline["test_predictions"],
        prefix="baseline",
        llm_predictions=llm_result["predictions_test"] if llm_result else None,
    )
    test_predictions["length_score"] = length_control["test_scores"]
    test_predictions["length_pred"] = length_control["test_predictions"]

    val_predictions = build_predictions_frame(
        val_df,
        score=baseline["val_scores"],
        prediction=baseline["val_predictions"],
        prefix="baseline",
        llm_predictions=llm_result["predictions_validation"] if llm_result else None,
    )
    val_predictions["length_score"] = length_control["val_scores"]
    val_predictions["length_pred"] = length_control["val_predictions"]

    test_predictions.to_csv(
        results_dir / "exp3_predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    val_predictions.to_csv(
        results_dir / "exp3_predictions_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample = pd.concat([dev_df, val_df, test_df], ignore_index=True)
    # No exportar el contexto completo (9k palabras por fila): el CSV quedaría
    # enorme. Se conserva ``Unnamed: 0`` para localizar la fila original.
    sample_export = sample.drop(
        columns=["context", "dedup_key", "answer_key", "group"],
        errors="ignore",
    )
    sample_export.to_csv(
        results_dir / "exp3_sample.csv", index=False, encoding="utf-8-sig"
    )

    baseline["threshold_grid"].to_csv(
        results_dir / "exp3_threshold_search.csv", index=False, encoding="utf-8-sig"
    )
    length_control["threshold_grid"].to_csv(
        results_dir / "exp3_length_threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    variants.to_csv(
        results_dir / "exp3_entity_variants.csv", index=False, encoding="utf-8-sig"
    )

    if llm_result is not None:
        llm_result["raw_predictions"].to_csv(
            results_dir / "exp3_llm_raw_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(llm_result["few_shot_examples"]).to_csv(
            results_dir / "exp3_llm_few_shot_examples.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --- Tabla de métricas -------------------------------------------------
    metric_rows = [
        metrics_row(baseline["metrics_dev"]),
        metrics_row(baseline["metrics_val"]),
        metrics_row(baseline["metrics_test"]),
        metrics_row(length_control["metrics_dev"]),
        metrics_row(length_control["metrics_val"]),
        metrics_row(length_control["metrics_test"]),
    ]
    if llm_result is not None:
        metric_rows.extend(
            [
                metrics_row(llm_result["metrics_validation"]),
                metrics_row(llm_result["metrics_test"]),
            ]
        )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        results_dir / "exp3_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # --- Errores -----------------------------------------------------------
    error_mask = test_predictions["baseline_pred"] != test_predictions["label"]
    if llm_result is not None and "llm_pred" in test_predictions.columns:
        error_mask = error_mask | (
            test_predictions["llm_pred"].fillna(-1).astype(int)
            != test_predictions["label"]
        )
    errors = test_predictions[error_mask].copy()
    errors.to_csv(
        results_dir / "exp3_errors.csv", index=False, encoding="utf-8-sig"
    )

    # --- Gráficos ----------------------------------------------------------
    charts: list[str] = []

    save_chart(build_split_balance_chart(sample), "exp3_split_balance", graphs_dir, save_png)
    charts.append("exp3_split_balance.html")

    save_chart(
        build_length_distribution_chart(sample),
        "exp3_answer_length_distribution",
        graphs_dir,
        save_png,
    )
    charts.append("exp3_answer_length_distribution.html")

    save_chart(
        build_entity_risk_chart(sample), "exp3_entity_grounding", graphs_dir, save_png
    )
    charts.append("exp3_entity_grounding.html")

    save_chart(
        build_threshold_curve_chart(
            baseline["threshold_grid"],
            baseline["threshold"],
            score_name="grounding de entidades",
        ),
        "exp3_threshold_curve",
        graphs_dir,
        save_png,
    )
    charts.append("exp3_threshold_curve.html")

    save_chart(
        build_confusion_matrix_chart(baseline["metrics_test"], "Baseline de entidades"),
        "exp3_confusion_baseline",
        graphs_dir,
        save_png,
    )
    charts.append("exp3_confusion_baseline.html")

    if llm_result is not None:
        save_chart(
            build_confusion_matrix_chart(llm_result["metrics_test"], "LLM few-shot"),
            "exp3_confusion_llm",
            graphs_dir,
            save_png,
        )
        charts.append("exp3_confusion_llm.html")

    model_metrics = {
        "Entidades (baseline)": baseline["metrics_test"],
        "Control de longitud": length_control["metrics_test"],
    }
    if llm_result is not None:
        model_metrics["LLM few-shot"] = llm_result["metrics_test"]

    save_chart(
        build_model_comparison_chart(model_metrics),
        "exp3_model_comparison",
        graphs_dir,
        save_png,
    )
    charts.append("exp3_model_comparison.html")

    save_chart(
        build_score_vs_length_chart(sample),
        "exp3_score_vs_length",
        graphs_dir,
        save_png,
    )
    charts.append("exp3_score_vs_length.html")

    save_chart(
        build_entity_variants_chart(variants),
        "exp3_entity_variants",
        graphs_dir,
        save_png,
    )
    charts.append("exp3_entity_variants.html")

    # --- Resumen JSON ------------------------------------------------------
    mcnemar: dict[str, Any] | None = None
    if llm_result is not None:
        mcnemar = mcnemar_test(
            test_df["label"].to_numpy(dtype=int),
            baseline["test_predictions"],
            llm_result["test_predictions"],
        )

    summary: dict[str, Any] = {
        "experiment": "exp3_logical_consistency",
        "seed": seed,
        "config": config,
        "dataset": {
            **metadata,
            "n_dev": int(len(dev_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "dev_label_counts": dev_df["label"].value_counts().sort_index().to_dict(),
            "validation_label_counts": val_df["label"].value_counts().sort_index().to_dict(),
            "test_label_counts": test_df["label"].value_counts().sort_index().to_dict(),
            "length_words_by_class": {
                LABEL_NAMES[int(label)]: {
                    "mean": float(group["n_words"].mean()),
                    "median": float(group["n_words"].median()),
                    "max": int(group["n_words"].max()),
                }
                for label, group in sample.groupby("label")
            },
        },
        "baseline_entities": {
            "score": baseline["score_name"],
            "threshold": baseline["threshold"],
            "metrics_dev": baseline["metrics_dev"],
            "metrics_validation": baseline["metrics_val"],
            "metrics_test": baseline["metrics_test"],
        },
        "length_control": {
            "score": length_control["score_name"],
            "threshold": length_control["threshold"],
            "metrics_dev": length_control["metrics_dev"],
            "metrics_validation": length_control["metrics_val"],
            "metrics_test": length_control["metrics_test"],
        },
        "entity_variants": variants.to_dict(orient="records"),
        "llm": None,
        "mcnemar_baseline_vs_llm": mcnemar,
        "limitations": {
            "length_confound": True,
            "message": (
                "Las respuestas alucinadas de Phantom son sistemáticamente más "
                "largas que las limpias. Un umbral sobre la longitud de la "
                "respuesta es una referencia trivial que puede igualar o superar "
                "a los métodos evaluados; el resultado no debe leerse como "
                "detección semántica genuina."
            ),
            "query_context_pairs": (
                "Phantom contiene pares de respuestas (limpia/alucinada) para la "
                "misma query y contexto. El split agrupado por query evita que un "
                "mismo contexto aparezca en dev y test."
            ),
        },
        "outputs": {
            "graphs_dir": str(graphs_dir),
            "results_dir": str(results_dir),
            "charts": charts,
        },
    }

    if llm_result is not None:
        summary["llm"] = {
            "model_name": llm_result["model_name"],
            "few_shot_examples": llm_result["few_shot_examples"],
            "metadata": llm_result["metadata"],
            "metrics_validation": llm_result["metrics_validation"],
            "metrics_test": llm_result["metrics_test"],
        }

    with open(results_dir / "exp3_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)

    return summary


# ---------------------------------------------------------------------------
# 7. Main / CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Experimento 3: consistencia lógica en Phantom. Baseline de "
            "grounding de entidades con spaCy vs. LLM few-shot. Reporta F1 "
            "macro, MCC, matrices de confusión e IC bootstrap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla global.")
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=100,
        help="Ejemplos por clase en la muestra balanceada.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=60_000,
        help="Caracteres de contexto usados para la extracción de entidades.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default=DEFAULT_SPACY_MODEL,
        help="Modelo de spaCy para NER.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=int,
        default=200,
        help="Cantidad de umbrales candidatos en la búsqueda.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM,
        help=(
            "Modelo instruct de HuggingFace. Alternativas no gated: "
            "Qwen/Qwen2.5-1.5B-Instruct, HuggingFaceTB/SmolLM2-1.7B-Instruct. "
            "Para Gemma hace falta --hf-token."
        ),
    )
    parser.add_argument(
        "--top-k-context",
        type=int,
        default=8,
        help="Oraciones de contexto recuperadas para el prompt del LLM.",
    )
    parser.add_argument(
        "--few-shot-context-sentences",
        type=int,
        default=3,
        help="Oraciones de contexto en cada ejemplo few-shot.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="Presupuesto máximo de tokens del prompt.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size para generación."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Tokens máximos generados por respuesta.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Ejecutar solo el baseline de entidades y el control de longitud.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Forzar el LLM en CPU (muy lento)."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token de HuggingFace para modelos gated (o usá `huggingface-cli login`).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Caché de HuggingFace para el dataset.",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=root / "graphs" / "exp3",
        help="Directorio de gráficos.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "exp3",
        help="Directorio de resultados.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Remuestreos bootstrap para los intervalos de confianza.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Exportar también PNG (requiere vl-convert-python).",
    )
    return parser.parse_args(argv)


def compute_variants(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variants: list[tuple[str, str]],
    n_grid: int,
) -> pd.DataFrame:
    """Evalúa varios scores de entidades con umbral elegido en desarrollo."""
    rows: list[dict[str, Any]] = []
    for label, column in variants:
        if column not in dev_df.columns:
            continue
        best, _ = tune_threshold(dev_df["label"], dev_df[column], n_grid=n_grid)
        threshold = best["threshold"]

        dev_pred = (dev_df[column].to_numpy(dtype=float) >= threshold).astype(int)
        test_pred = (test_df[column].to_numpy(dtype=float) >= threshold).astype(int)

        rows.append(
            {
                "variant": label,
                "column": column,
                "threshold": float(threshold),
                "split": "dev",
                "auc": safe_roc_auc(dev_df["label"], dev_df[column]),
                "f1_macro": binary_metrics(dev_df["label"], dev_pred)["f1_macro"],
                "mcc": binary_metrics(dev_df["label"], dev_pred)["mcc"],
            }
        )
        rows.append(
            {
                "variant": label,
                "column": column,
                "threshold": float(threshold),
                "split": "test",
                "auc": safe_roc_auc(test_df["label"], test_df[column]),
                "f1_macro": binary_metrics(test_df["label"], test_pred)["f1_macro"],
                "mcc": binary_metrics(test_df["label"], test_pred)["mcc"],
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        return output

    test_auc = (
        output[output["split"] == "test"]
        .set_index("variant")["auc"]
        .sort_values(ascending=False)
    )
    order = {variant: position for position, variant in enumerate(test_auc.index)}
    output["variant_order"] = output["variant"].map(order).fillna(len(order)).astype(int)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    print_header("EXPERIMENTO 3 — CONSISTENCIA LÓGICA CON ENTIDADES Y LLM")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Modelo spaCy: {args.spacy_model}")
    print(f"Modelo LLM: {args.model if not args.skip_llm else 'omitido (--skip-llm)'}")

    # ------------------------------------------------------------------ [1/5]
    print_section("[1/5] MUESTRA BALANCEADA Y SPLIT AGRUPADO POR QUERY")
    df, metadata = load_phantom_sample(
        n_per_class=args.n_per_class,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    df["uid"] = np.arange(len(df), dtype=int)
    print(f"  Filas crudas: {metadata['raw_rows']}")
    print(f"  Filas luego de deduplicar terneras: {metadata['rows_after_dedup']}")
    print(f"  Muestra: {metadata['n_total']} filas ({metadata['n_per_class']} por clase)")
    print(f"  Queries únicas: {metadata['n_groups']}")

    print("  Extrayendo entidades con spaCy (puede tardar un minuto)...")
    nlp = load_spacy(args.spacy_model)
    started = time.time()
    features = compute_entity_features(
        df, nlp, max_context_chars=args.max_context_chars
    )
    # ``n_words`` ya existe en df: evitar columnas duplicadas al concatenar.
    features = features[[column for column in features.columns if column not in df.columns]]
    df = pd.concat([df, features], axis=1)
    print(f"  Entidades extraídas en {time.time() - started:.1f} s")

    dev_df, val_df, test_df = split_dataset_grouped(df, seed=args.seed)
    print(
        f"  Split: dev={len(dev_df)} | validación={len(val_df)} | test={len(test_df)}"
    )
    for name, frame in (
        ("dev", dev_df),
        ("validación", val_df),
        ("test", test_df),
    ):
        counts = frame["label"].value_counts().sort_index().to_dict()
        print(f"    {name}: {counts}")
    print(
        "  Longitud media (palabras): "
        + " | ".join(
            f"{LABEL_NAMES[int(label)]}={group['n_words'].mean():.1f}"
            for label, group in df.groupby("label")
        )
    )
    print(
        "  [ADVERTENCIA] Las respuestas alucinadas son mucho más largas que las "
        "limpias; se reporta un control de longitud."
    )

    # ------------------------------------------------------------------ [2/5]
    print_section("[2/5] BASELINE DE GROUNDING DE ENTIDADES")
    score_name = "unsupported_entity_rate"
    best, grid = tune_threshold(
        dev_df["label"], dev_df[score_name], n_grid=args.threshold_grid
    )
    threshold = float(best["threshold"])

    def predictions_for(frame: pd.DataFrame) -> np.ndarray:
        return (frame[score_name].to_numpy(dtype=float) >= threshold).astype(int)

    baseline = {
        "score_name": score_name,
        "threshold": threshold,
        "threshold_grid": grid,
        "dev_scores": dev_df[score_name].to_numpy(dtype=float),
        "val_scores": val_df[score_name].to_numpy(dtype=float),
        "test_scores": test_df[score_name].to_numpy(dtype=float),
        "dev_predictions": predictions_for(dev_df),
        "val_predictions": predictions_for(val_df),
        "test_predictions": predictions_for(test_df),
    }
    baseline["metrics_dev"] = evaluate_model(
        "Entidades (baseline)",
        "dev",
        dev_df["label"],
        baseline["dev_predictions"],
        baseline["dev_scores"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    baseline["metrics_val"] = evaluate_model(
        "Entidades (baseline)",
        "validation",
        val_df["label"],
        baseline["val_predictions"],
        baseline["val_scores"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    baseline["metrics_test"] = evaluate_model(
        "Entidades (baseline)",
        "test",
        test_df["label"],
        baseline["test_predictions"],
        baseline["test_scores"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    print(f"  Score: {score_name}")
    print(f"  Umbral elegido en dev: {threshold:.4f}")
    print(
        f"  Dev: F1 macro={baseline['metrics_dev']['f1_macro']:.4f} | "
        f"MCC={baseline['metrics_dev']['mcc']:.4f}"
    )
    print("\n  Test:")
    print(metrics_table(baseline["metrics_test"]).to_string(index=False))
    print(
        f"  ROC-AUC (score continuo): {baseline['metrics_test']['roc_auc']:.4f}"
    )
    print(
        f"  IC 95 % F1 macro: "
        f"[{baseline['metrics_test']['f1_macro_ci_low']:.4f}, "
        f"{baseline['metrics_test']['f1_macro_ci_high']:.4f}]"
    )
    print(
        f"  IC 95 % MCC: "
        f"[{baseline['metrics_test']['mcc_ci_low']:.4f}, "
        f"{baseline['metrics_test']['mcc_ci_high']:.4f}]"
    )

    # Control de longitud: referencia trivial.
    length_best, length_grid = tune_threshold(
        dev_df["label"], dev_df["n_words"], n_grid=args.threshold_grid
    )
    length_threshold = float(length_best["threshold"])
    length_control = {
        "score_name": "n_words",
        "threshold": length_threshold,
        "threshold_grid": length_grid,
        "dev_scores": dev_df["n_words"].to_numpy(dtype=float),
        "val_scores": val_df["n_words"].to_numpy(dtype=float),
        "test_scores": test_df["n_words"].to_numpy(dtype=float),
        "dev_predictions": (
            dev_df["n_words"].to_numpy(dtype=float) >= length_threshold
        ).astype(int),
        "val_predictions": (
            val_df["n_words"].to_numpy(dtype=float) >= length_threshold
        ).astype(int),
        "test_predictions": (
            test_df["n_words"].to_numpy(dtype=float) >= length_threshold
        ).astype(int),
    }
    length_control["metrics_dev"] = evaluate_model(
        "Control de longitud",
        "dev",
        dev_df["label"],
        length_control["dev_predictions"],
        length_control["dev_scores"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    length_control["metrics_val"] = evaluate_model(
        "Control de longitud",
        "validation",
        val_df["label"],
        length_control["val_predictions"],
        length_control["val_scores"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    length_control["metrics_test"] = evaluate_model(
        "Control de longitud",
        "test",
        test_df["label"],
        length_control["test_predictions"],
        length_control["test_scores"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    print("\n  Control de longitud (referencia, NO es un método propuesto):")
    print(f"    Umbral elegido en dev: {length_threshold:.2f} palabras")
    print(
        f"    Test: F1 macro={length_control['metrics_test']['f1_macro']:.4f} | "
        f"MCC={length_control['metrics_test']['mcc']:.4f} | "
        f"ROC-AUC={length_control['metrics_test']['roc_auc']:.4f}"
    )

    # Variantes de scores de entidades.
    variant_specs = [
        ("Entidades no respaldadas (fracción)", "unsupported_entity_rate"),
        ("Entidades no respaldadas (conteo)", "n_unsupported_entities"),
        ("Números no respaldados (fracción)", "unsupported_number_rate"),
        ("Números no respaldados (conteo)", "n_unsupported_numbers"),
        ("Cobertura de entidades de la query", "query_entity_coverage"),
        ("Cantidad de entidades en la respuesta", "n_answer_entities"),
        ("Longitud de la respuesta (control)", "n_words"),
    ]
    variants = compute_variants(
        dev_df=dev_df,
        test_df=test_df,
        variants=variant_specs,
        n_grid=args.threshold_grid,
    )
    print("\n  Variantes evaluadas (umbral elegido en dev, métricas en test):")
    if not variants.empty:
        print(
            variants[variants["split"] == "test"][
                ["variant", "threshold", "auc", "f1_macro", "mcc"]
            ]
            .sort_values("auc", ascending=False)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------ [3/5]
    llm_result: dict[str, Any] | None = None
    if args.skip_llm:
        print_section("[3/5] LLM FEW-SHOT — OMITIDO (--skip-llm)")
    else:
        print_section("[3/5] LLM FEW-SHOT")
        device = resolve_torch_device(use_cpu=args.cpu)
        print(f"  Dispositivo: {device}")
        load_started = time.time()
        tokenizer, model = load_llm(
            args.model, device=device, hf_token=args.hf_token
        )
        print(
            f"  Modelo cargado en {time.time() - load_started:.1f} s "
            f"(dtype={resolve_torch_dtype(device)})"
        )

        few_shot_block, few_shot_records = build_few_shot_examples(
            dev_df=dev_df,
            seed=args.seed,
            n_per_class=2,
            top_k_context=args.few_shot_context_sentences,
        )
        print(f"  Ejemplos few-shot: {len(few_shot_records)} (2 por clase)")

        system_prompt = (
            "You are a rigorous fact-checker for financial documents. Given a "
            "CONTEXT, a QUESTION and an ANSWER, decide whether the ANSWER is "
            "fully supported by the CONTEXT.\n"
            "- HALLUCINATION: the answer adds specific facts, numbers, names, "
            "dates, percentages or details that are not present in the context, "
            "or contradicts the context.\n"
            "- CLEAN: every factual claim in the answer is supported by the "
            "context.\n"
            "Do not use the length of the answer as evidence. Focus only on "
            "whether its factual claims are supported.\n"
            "Reply with a single word: HALLUCINATION or CLEAN."
        )

        llm_frame = pd.concat([val_df, test_df], ignore_index=True)
        llm_frame = llm_frame.reset_index(drop=True)
        llm_raw, llm_metadata = run_llm_predictions(
            frame=llm_frame,
            tokenizer=tokenizer,
            model=model,
            system_prompt=system_prompt,
            few_shot_block=few_shot_block,
            device=device,
            top_k_context=args.top_k_context,
            max_input_tokens=args.max_input_tokens,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        llm_raw = llm_raw.reset_index().rename(columns={"row_index": "frame_index"})
        llm_enriched = llm_frame.join(
            llm_raw.set_index("frame_index"), how="left"
        )

        llm_validation = llm_enriched[llm_enriched["split"] == "validation"].copy()
        llm_test = llm_enriched[llm_enriched["split"] == "test"].copy()

        llm_prediction_columns = [
            "uid",
            "llm_raw",
            "llm_pred",
            "llm_valid",
            "n_prompt_tokens",
            "n_context_sentences",
        ]
        llm_result = {
            "model_name": args.model,
            "predictions_validation": llm_validation[llm_prediction_columns],
            "predictions_test": llm_test[llm_prediction_columns],
            "raw_predictions": llm_enriched[
                ["uid", "split", "query", "answer", "label"]
                + [
                    column
                    for column in llm_prediction_columns
                    if column not in {"uid", "query", "answer", "label"}
                ]
            ].copy(),
            "metadata": llm_metadata,
            "few_shot_examples": few_shot_records,
            "metrics_validation": evaluate_model(
                "LLM few-shot",
                "validation",
                llm_validation["label"],
                llm_validation["llm_pred"],
                scores=None,
                n_boot=args.bootstrap,
                seed=args.seed,
            ),
            "metrics_test": evaluate_model(
                "LLM few-shot",
                "test",
                llm_test["label"],
                llm_test["llm_pred"],
                scores=None,
                n_boot=args.bootstrap,
                seed=args.seed,
            ),
            "test_predictions": llm_test["llm_pred"].to_numpy(dtype=int),
        }

        print(
            f"  Prompt: {llm_metadata['prompt_tokens_min']}–"
            f"{llm_metadata['prompt_tokens_max']} tokens "
            f"(mediana {llm_metadata['prompt_tokens_median']:.0f})"
        )
        print(
            f"  Salidas inválidas: {llm_metadata['n_invalid']} | "
            f"contextos recortados: {llm_metadata['n_context_truncated']}"
        )
        print("\n  Test:")
        print(metrics_table(llm_result["metrics_test"]).to_string(index=False))
        print(
            f"  IC 95 % F1 macro: "
            f"[{llm_result['metrics_test']['f1_macro_ci_low']:.4f}, "
            f"{llm_result['metrics_test']['f1_macro_ci_high']:.4f}]"
        )
        print(
            f"  IC 95 % MCC: "
            f"[{llm_result['metrics_test']['mcc_ci_low']:.4f}, "
            f"{llm_result['metrics_test']['mcc_ci_high']:.4f}]"
        )

        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ [4/5]
    print_section("[4/5] COMPARACIÓN FINAL (TEST)")
    comparison_rows = [
        {
            "Modelo": "Entidades (baseline)",
            "F1 macro": f"{baseline['metrics_test']['f1_macro']:.4f}",
            "F1 alucinación": f"{baseline['metrics_test']['f1_hallucination']:.4f}",
            "MCC": f"{baseline['metrics_test']['mcc']:.4f}",
        },
        {
            "Modelo": "Control de longitud",
            "F1 macro": f"{length_control['metrics_test']['f1_macro']:.4f}",
            "F1 alucinación": f"{length_control['metrics_test']['f1_hallucination']:.4f}",
            "MCC": f"{length_control['metrics_test']['mcc']:.4f}",
        },
    ]
    if llm_result is not None:
        comparison_rows.insert(
            1,
            {
                "Modelo": "LLM few-shot",
                "F1 macro": f"{llm_result['metrics_test']['f1_macro']:.4f}",
                "F1 alucinación": f"{llm_result['metrics_test']['f1_hallucination']:.4f}",
                "MCC": f"{llm_result['metrics_test']['mcc']:.4f}",
            },
        )
    print(pd.DataFrame(comparison_rows).to_string(index=False))

    if llm_result is not None:
        mcnemar = mcnemar_test(
            test_df["label"].to_numpy(dtype=int),
            baseline["test_predictions"],
            llm_result["test_predictions"],
        )
        print("\n  McNemar baseline vs. LLM (test):")
        print(
            f"    baseline correcto / LLM incorrecto: {mcnemar['a_correct_b_wrong']}"
        )
        print(
            f"    LLM correcto / baseline incorrecto: {mcnemar['b_correct_a_wrong']}"
        )
        print(f"    p-valor exacto: {mcnemar['p_value']:.4f}")

    print(
        "\n  [INTERPRETACIÓN] Si el control de longitud iguala o supera a los "
        "métodos, el resultado mide verbosidad, no detección semántica."
    )

    # ------------------------------------------------------------------ [5/5]
    print_section("[5/5] EXPORTACIÓN DE RESULTADOS")
    config = {
        "n_per_class": args.n_per_class,
        "max_context_chars": args.max_context_chars,
        "spacy_model": args.spacy_model,
        "threshold_grid": args.threshold_grid,
        "llm_model": None if args.skip_llm else args.model,
        "top_k_context": args.top_k_context,
        "few_shot_context_sentences": args.few_shot_context_sentences,
        "max_input_tokens": args.max_input_tokens,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "bootstrap": args.bootstrap,
    }

    summary = export_results(
        df=df,
        dev_df=dev_df,
        val_df=val_df,
        test_df=test_df,
        baseline=baseline,
        length_control=length_control,
        variants=variants,
        llm_result=llm_result,
        metadata=metadata,
        config=config,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        save_png=args.save_png,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    print(f"  Gráficos: {summary['outputs']['graphs_dir']}")
    for chart in summary["outputs"]["charts"]:
        print(f"    - {chart}")
    print(f"  Resultados: {summary['outputs']['results_dir']}")
    for filename in (
        "exp3_sample.csv",
        "exp3_predictions_test.csv",
        "exp3_predictions_validation.csv",
        "exp3_metrics.csv",
        "exp3_threshold_search.csv",
        "exp3_length_threshold_search.csv",
        "exp3_entity_variants.csv",
        "exp3_errors.csv",
        "exp3_summary.json",
    ):
        if (args.results_dir / filename).exists():
            print(f"    - {filename}")

    print_header("EXPERIMENTO 3 FINALIZADO")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# INSTRUCCIONES DE EJECUCIÓN (resumen)
# -----------------------------------------------------------------------------
# 1) Instalar dependencias:
#
#       source .venv/bin/activate
#       pip install -r requirements-exp3.txt
#       # si el modelo de spaCy no se instaló:
#       python -m spacy download en_core_web_sm
#
# 2) Ejecución completa (muestra 100+100, LLM Qwen2.5-3B-Instruct):
#
#       python experiments/exp3_logical_consistency.py
#
# 3) Solo baseline de entidades + control de longitud (sin descargar LLM):
#
#       python experiments/exp3_logical_consistency.py --skip-llm
#
# 4) Smoke test con 20 ejemplos por clase:
#
#       python experiments/exp3_logical_consistency.py \
#           --n-per-class 20 --skip-llm
#
# 5) LLM más chico y rápido:
#
#       python experiments/exp3_logical_consistency.py \
#           --model Qwen/Qwen2.5-1.5B-Instruct
#
# 6) LLM más grande (requiere ~15 GB de VRAM/RAM; puede necesitar 4-bit):
#
#       python experiments/exp3_logical_consistency.py \
#           --model Qwen/Qwen2.5-7B-Instruct --batch-size 2
#
# 7) Gemma (modelo gated; requiere haber aceptado la licencia):
#
#       python experiments/exp3_logical_consistency.py \
#           --model google/gemma-1.1-2b-it --hf-token "$HF_TOKEN"
#
# 8) Exportar PNG:
#
#       pip install vl-convert-python
#       python experiments/exp3_logical_consistency.py --save-png
#
# Salidas principales:
#   graphs/exp3/exp3_split_balance.html
#   graphs/exp3/exp3_answer_length_distribution.html
#   graphs/exp3/exp3_entity_grounding.html
#   graphs/exp3/exp3_threshold_curve.html
#   graphs/exp3/exp3_confusion_baseline.html
#   graphs/exp3/exp3_confusion_llm.html
#   graphs/exp3/exp3_model_comparison.html
#   graphs/exp3/exp3_score_vs_length.html
#   graphs/exp3/exp3_entity_variants.html
#   results/exp3/exp3_sample.csv
#   results/exp3/exp3_predictions_test.csv
#   results/exp3/exp3_metrics.csv
#   results/exp3/exp3_entity_variants.csv
#   results/exp3/exp3_errors.csv
#   results/exp3/exp3_summary.json
# -----------------------------------------------------------------------------
