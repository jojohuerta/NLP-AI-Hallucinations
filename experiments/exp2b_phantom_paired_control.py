#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 2b — Clasificación binaria en Phantom con control de tema y longitud
=================================================================================

Motivación
----------
El Experimento 2 original mezcla dos fuentes: los positivos vienen de Mu-SHROOM y
los negativos de Phantom. Eso hace que la etiqueta esté perfectamente confundida
con la fuente, de modo que un F1 casi perfecto puede reflejar separación de tema,
estilo o longitud, y no detección genuina de alucinaciones.

Este experimento elimina ese problema usando **solo Phantom**, que tiene pares
de respuestas (una limpia y una alucinada) para la misma pregunta y el mismo
contexto. Además controla el confound de longitud, que en Phantom es muy fuerte.

Diseño
------
1. Se deduplican las respuestas por texto.
2. Se forman **pares dentro de la misma (query, context)**: una respuesta limpia
   y una alucinada del mismo tema. Eso elimina el confound fuente-etiqueta y el
   confound de tema.
3. **Control de longitud simétrico**: se conservan solo los pares en los que
   ambas respuestas tienen al menos ``--target-words`` palabras y las dos se
   recortan exactamente a esa cantidad. La longitud queda idéntica en las dos
   clases, no solo en promedio.
4. A los textos recortados **se les quita la puntuación**. Sin este paso el
   recorte deja una pista trivial: la respuesta alucinada suele ser más larga y
   al cortarla termina en medio de una palabra, mientras que la limpia termina
   en punto. Una regla "si termina en punto, es limpia" alcanza F1 ≈ 0,92 sobre
   el recorte ingenuo; el experimento mide ese artefacto explícitamente y lo
   neutraliza, en lugar de explotarlo.
5. Se descartan los pares en los que los dos textos recortados son idénticos o
   casi idénticos (Jaccard ≥ ``--max-jaccard``): si el texto es el mismo, la
   etiqueta es ambigua por construcción y solo agrega ruido.
6. El split es agrupado por ``query`` con ``StratifiedGroupKFold``: los dos
   miembros de un par, y todas las respuestas de una misma pregunta, caen
   siempre en la misma partición.
7. Se comparan:
   * **TF-IDF + LogisticRegression** sobre el texto de la respuesta.
   * **DistilBERT** fine-tuneado sobre el texto de la respuesta.
   * **Umbral de longitud** (control trivial; no debe superar el azar).
   * **Query-only** (diagnóstico de fuga por tema; debe quedar en azar).
   * **Regla de puntuación** (diagnóstico del artefacto de recorte).
8. Se reportan además dos referencias para dimensionar el efecto del control:
   * **Sin control**: muestra balanceada de respuestas completas, sin parear.
   * **Tema controlado, longitud no**: los mismos pares con el texto completo.

Métricas
--------
F1 macro, F1 de la clase alucinación, precisión y recall por clase, MCC,
ROC-AUC, PR-AUC y matriz de confusión. No se calcula ni se reporta la métrica
global de aciertos, y no se usa ``classification_report``.

Salidas
-------
``graphs/exp2b/``  : gráficos Altair en HTML (y PNG opcional).
``results/exp2b/`` : CSV, JSON, predicciones y diagnóstico de pares.

Uso
---
Ver el bloque de instrucciones al final del archivo o ejecutar ``--help``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import altair as alt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

# Se importan helpers del Experimento 2 para no duplicar pipeline, métricas,
# fine-tuning ni gráficos. El script vive en ``experiments/``, así que ese
# directorio ya está en ``sys.path`` al ejecutarlo; el fallback cubre el caso
# de importarlo como módulo desde la raíz del repo.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from exp2_binary_classification import (  # noqa: E402
    COLOR_HALLUCINATION,
    COLOR_NEUTRAL,
    COLOR_TRUTHFUL,
    DEFAULT_TRANSFORMER,
    LABEL_NAMES,
    PHANTOM_NAME,
    SEED,
    build_baseline_pipeline,
    build_confusion_matrix_chart,
    build_model_comparison_chart,
    build_probability_chart,
    build_training_curves_chart,
    classification_metrics,
    count_words,
    json_default,
    metrics_table,
    normalize_text_key,
    print_header,
    print_section,
    project_root,
    save_chart,
    set_seed,
    train_transformer,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
alt.data_transformers.disable_max_rows()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DEFAULT_MIN_WORDS: int = 20
DEFAULT_TARGET_WORDS: int = 40
DEFAULT_MAX_JACCARD: float = 0.98
DEFAULT_N_SPLITS: int = 7
DEFAULT_SENSITIVITY_LENGTHS: tuple[int, ...] = (25, 40, 60)
SPLIT_NAMES = {"train": "Train", "validation": "Validación", "test": "Test"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalize_group(text: Any) -> str:
    """Normalización agresiva para agrupar queries/contextos equivalentes."""
    import re

    return re.sub(r"\W+", " ", str(text).lower()).strip()


def truncate_words(text: Any, n_words: int) -> str:
    """Devuelve las primeras ``n_words`` palabras del texto."""
    return " ".join(str(text).split()[: int(n_words)])


def clean_tokens(text: Any) -> list[str]:
    """Tokeniza en palabras/números alfanuméricos, descartando la puntuación.

    Es parte del control de longitud: si se recorta a mitad de oración, el
    final del texto delata la clase (limpia = termina en punto, alucinada =
    termina cortada). Al eliminar la puntuación, el punto de corte deja de ser
    una señal explotable.
    """
    import re

    return re.findall(r"[A-Za-z0-9%]+", str(text))


def prepare_paired_text(text: Any, n_words: int) -> str:
    """Toma exactamente ``n_words`` tokens limpios (sin puntuación)."""
    return " ".join(clean_tokens(text)[: int(n_words)])


def ends_with_sentence_punctuation(text: Any) -> bool:
    """``True`` si el texto termina en punto, signo de exclamación o pregunta."""
    stripped = str(text).rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def word_jaccard(first: str, second: str) -> float:
    """Jaccard de tokens entre dos textos (0 = disjuntos, 1 = mismo vocabulario)."""
    import re

    tokens_a = set(re.findall(r"[a-z0-9]+", str(first).lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", str(second).lower()))
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(tokens_a & tokens_b) / len(union)


def greedy_match_by_length(
    clean: pd.DataFrame,
    hallucinated: pd.DataFrame,
) -> list[tuple[pd.Series, pd.Series]]:
    """Empareja respuestas limpias y alucinadas por longitud, uno a uno.

    En Phantom cada (query, context) con ambas etiquetas suele tener exactamente
    una respuesta de cada clase. La función igual soporta grupos con varias
    respuestas por clase: empareja la combinación de longitudes más parecida.
    """
    clean_sorted = clean.sort_values("n_words").reset_index(drop=True)
    hallucinated_sorted = hallucinated.sort_values("n_words").reset_index(drop=True)
    used: set[int] = set()
    pairs: list[tuple[pd.Series, pd.Series]] = []

    for _, clean_row in clean_sorted.iterrows():
        best_distance: float | None = None
        best_index: int | None = None
        for index, hall_row in hallucinated_sorted.iterrows():
            if index in used:
                continue
            distance = abs(float(clean_row["n_words"]) - float(hall_row["n_words"]))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = int(index)
        if best_index is not None:
            used.add(best_index)
            pairs.append((clean_row, hallucinated_sorted.loc[best_index]))

    return pairs


# ---------------------------------------------------------------------------
# 1. Carga y construcción de pares con longitud controlada
# ---------------------------------------------------------------------------

def load_phantom_pairs(
    cache_dir: str | None = None,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_jaccard: float = DEFAULT_MAX_JACCARD,
    max_pairs: int = 0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construye el dataset de pares (limpia, alucinada) de Phantom.

    Cada par comparte ``(query, context)`` y sus dos textos se recortan a
    exactamente ``target_words`` tokens sin puntuación. Se descartan los pares
    con solapamiento casi total, porque con el mismo texto la etiqueta es
    ambigua por construcción.

    Devuelve:
    * ``df``: dos filas por par, con el texto recortado a la misma longitud.
    * ``pool``: respuestas únicas completas, para la referencia sin control.
    * ``metadata``: conteos y estadísticas del proceso.
    """
    ds = load_dataset(PHANTOM_NAME, cache_dir=cache_dir)
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
    raw = raw[raw["answer"].str.strip().str.len() > 0].copy()

    raw["label"] = (raw["label_text"] == "hallucination").astype(int)
    raw["text"] = raw["answer"]
    raw["text_key"] = raw["text"].map(normalize_text_key)
    raw["n_words"] = raw["text"].map(count_words)
    raw["n_tokens"] = raw["text"].map(lambda value: len(clean_tokens(value)))
    raw["group"] = raw["query"].map(normalize_group)
    raw["context_key"] = raw["context"].map(normalize_group)

    n_raw = int(len(raw))
    pool = raw.drop_duplicates(subset=["text_key"], keep="first").copy()
    pool["source"] = "phantom"
    pool["source_id"] = "phantom:" + pool.index.astype(str)
    pool["label_name"] = pool["label"].map(LABEL_NAMES)

    records: list[dict[str, Any]] = []
    candidate_pairs = 0
    skipped_short = 0
    skipped_duplicate = 0
    skipped_groups = 0

    for (group, context_key), frame in pool.groupby(["group", "context_key"], sort=False):
        clean = frame[frame["label"] == 0]
        hallucinated = frame[frame["label"] == 1]
        if clean.empty or hallucinated.empty:
            skipped_groups += 1
            continue

        for clean_row, hall_row in greedy_match_by_length(clean, hallucinated):
            candidate_pairs += 1
            original_clean = int(clean_row["n_words"])
            original_hall = int(hall_row["n_words"])

            # El criterio de elegibilidad usa los tokens limpios: así el
            # recorte final tiene exactamente ``target_words`` en ambas clases.
            if min(clean_row["n_tokens"], hall_row["n_tokens"]) < int(target_words):
                skipped_short += 1
                continue

            clean_text = prepare_paired_text(clean_row["text"], target_words)
            hall_text = prepare_paired_text(hall_row["text"], target_words)
            identical = clean_text == hall_text
            jaccard = word_jaccard(clean_text, hall_text)

            if identical or jaccard >= float(max_jaccard):
                skipped_duplicate += 1
                continue

            pair_id = len(records) // 2
            for row, text, original_length in (
                (clean_row, clean_text, original_clean),
                (hall_row, hall_text, original_hall),
            ):
                label = int(row["label"])
                records.append(
                    {
                        "pair_id": pair_id,
                        "query": row["query"],
                        "context_preview": str(row["context"])[:300],
                        "context_key": context_key,
                        "group": group,
                        "text": text,
                        "text_full": row["text"],
                        "n_words": int(target_words),
                        "n_words_original": int(original_length),
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "identical_prefix": bool(identical),
                        "pair_jaccard": float(jaccard),
                        "source": "phantom",
                        "source_id": f"phantom:{row.name}",
                        "text_key": normalize_text_key(text),
                        "text_key_full": row["text_key"],
                    }
                )

    if not records:
        raise RuntimeError(
            "No se pudo construir ningún par. Probá bajando --target-words."
        )

    df = pd.DataFrame(records)

    # Protección extra: un mismo texto recortado no puede aparecer en ambas
    # clases (sería una fila idéntica con dos etiquetas distintas).
    leaked = df.groupby("text_key")["label"].nunique().gt(1).sum()
    if leaked:
        bad_pairs = df[df.groupby("text_key")["label"].transform("nunique") > 1]["pair_id"]
        print(
            f"  [aviso] {leaked} textos recortados aparecen en ambas clases; "
            f"se eliminan {bad_pairs.nunique()} pares."
        )
        df = df[~df["pair_id"].isin(set(bad_pairs))].copy()

    # Submuestreo opcional de pares completos (nunca de filas sueltas).
    pair_ids = df["pair_id"].unique()
    if max_pairs and max_pairs > 0 and len(pair_ids) > max_pairs:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(pair_ids, size=int(max_pairs), replace=False)
        df = df[df["pair_id"].isin(set(chosen))].copy()

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["uid"] = np.arange(len(df), dtype=int)

    counts = df["label"].value_counts().to_dict()
    if counts.get(0, 0) != counts.get(1, 0):
        raise RuntimeError(f"El dataset de pares no quedó balanceado: {counts}")

    n_pairs = int(df["pair_id"].nunique())
    if len(df) != 2 * n_pairs:
        raise RuntimeError("Cada par debe aportar exactamente una fila por clase.")

    pair_lengths = df.groupby("pair_id")["n_words"].nunique()
    if (pair_lengths > 1).any():
        raise RuntimeError("Los dos textos de un par no tienen la misma longitud.")

    metadata = {
        "dataset": PHANTOM_NAME,
        "split": split,
        "raw_rows": n_raw,
        "unique_answers": int(len(pool)),
        "unique_queries": int(df["group"].nunique()),
        "unique_contexts": int(df["context_key"].nunique()),
        "candidate_pairs": int(candidate_pairs),
        "used_pairs": n_pairs,
        "skipped_short_pairs": int(skipped_short),
        "skipped_duplicate_pairs": int(skipped_duplicate),
        "skipped_groups_without_both_labels": int(skipped_groups),
        "target_words": int(target_words),
        "max_jaccard": float(max_jaccard),
        "punctuation_stripped": True,
        "n_total": int(len(df)),
        "n_per_class": int(counts[1]),
        "identical_prefix_pairs": int(df.groupby("pair_id")["identical_prefix"].first().sum()),
        "identical_prefix_rate": float(
            df.groupby("pair_id")["identical_prefix"].first().mean()
        ),
        "pair_jaccard": {
            "mean": float(df.groupby("pair_id")["pair_jaccard"].first().mean()),
            "median": float(df.groupby("pair_id")["pair_jaccard"].first().median()),
            "p90": float(
                df.groupby("pair_id")["pair_jaccard"].first().quantile(0.90)
            ),
        },
        "length_words_by_class": {
            LABEL_NAMES[label]: {
                "mean": float(group["n_words"].mean()),
                "median": float(group["n_words"].median()),
                "max": int(group["n_words"].max()),
                "mean_original": float(group["n_words_original"].mean()),
                "median_original": float(group["n_words_original"].median()),
            }
            for label, group in df.groupby("label")
        },
    }
    return df, pool, metadata


# ---------------------------------------------------------------------------
# 2. Split agrupado por query
# ---------------------------------------------------------------------------

def split_pairs(
    df: pd.DataFrame,
    seed: int = SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 5/7 - 1/7 - 1/7 agrupado por query y estratificado por etiqueta.

    Con ``n_splits=7`` se usan 5 folds como train (~71 %), uno como validación
    (~14 %) y uno como test (~14 %). Como cada par comparte ``group``, ambos
    miembros caen siempre en la misma partición y el balance es exacto 50/50.
    """
    if n_splits < 3:
        raise ValueError("n_splits debe ser al menos 3.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(df, df["label"], df["group"]))

    train_idx = np.concatenate([fold[1] for fold in folds[: n_splits - 2]])
    val_idx = folds[n_splits - 2][1]
    test_idx = folds[n_splits - 1][1]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    # Verificaciones de integridad.
    groups = {
        "train": set(train_df["group"]),
        "validation": set(val_df["group"]),
        "test": set(test_df["group"]),
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = groups[first] & groups[second]
        if overlap:
            raise RuntimeError(
                f"Leakage de queries entre {first} y {second}: {len(overlap)}."
            )

    pair_splits = pd.concat(
        [
            train_df[["pair_id", "split"]],
            val_df[["pair_id", "split"]],
            test_df[["pair_id", "split"]],
        ]
    ).groupby("pair_id")["split"].nunique()
    if (pair_splits > 1).any():
        raise RuntimeError("Hay pares repartidos entre dos particiones.")

    contexts = {
        "train": set(train_df["context_key"]),
        "validation": set(val_df["context_key"]),
        "test": set(test_df["context_key"]),
    }
    context_overlap = (
        (contexts["train"] & contexts["test"])
        | (contexts["train"] & contexts["validation"])
        | (contexts["validation"] & contexts["test"])
    )
    if context_overlap:
        print(
            f"  [aviso] {len(context_overlap)} contextos aparecen en más de una "
            "partición (las queries sí están separadas)."
        )

    for name, frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        counts = frame["label"].value_counts().sort_index().to_dict()
        if counts.get(0, 0) != counts.get(1, 0):
            raise RuntimeError(f"Split desbalanceado en {name}: {counts}")

    return (
        train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 3. Modelos y métricas
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    n_grid: int = 200,
) -> tuple[float, float]:
    """Elige el umbral que maximiza F1 macro (regla ``score >= threshold``)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("inf"), 0.0

    if np.unique(finite).size == 1:
        candidates = np.array([float(finite[0]), np.inf])
    else:
        candidates = np.unique(
            np.concatenate(
                [np.quantile(finite, np.linspace(0.0, 1.0, n_grid)), [np.inf]]
            )
        )

    best_f1 = -1.0
    best_threshold = float(candidates[0])
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_length_control(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    length_column: str = "n_words",
    model_label: str = "Umbral de longitud",
) -> dict[str, Any]:
    """Control trivial: predice alucinación si la respuesta supera N palabras.

    El umbral se elige solo en validación. En el dataset principal la longitud
    es idéntica dentro de cada par, así que este control debe quedar en azar.
    """
    threshold, best_val_f1 = tune_threshold(
        val_df["label"], val_df[length_column]
    )

    def evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        y_true = frame["label"].to_numpy(dtype=int)
        scores = frame[length_column].to_numpy(dtype=float)
        y_pred = (scores >= threshold).astype(int)
        metrics = classification_metrics(y_true, y_pred, scores)
        predictions = pd.DataFrame(
            {
                "pair_id": frame["pair_id"].to_numpy(),
                "source_id": frame["source_id"].to_numpy(),
                "split": frame["split"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "prob_hallucination": scores,
                "text_preview": frame["text"].str.slice(0, 300).to_numpy(),
            }
        )
        return metrics, predictions

    val_metrics, val_predictions = evaluate(val_df)
    test_metrics, test_predictions = evaluate(test_df)
    return {
        "model": model_label,
        "threshold": float(threshold),
        "best_val_f1": float(best_val_f1),
        "metrics_validation": val_metrics,
        "metrics_test": test_metrics,
        "predictions_validation": val_predictions,
        "predictions_test": test_predictions,
    }


def train_lexical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    model_label: str = "TF-IDF + LogisticRegression",
    include_top_features: bool = True,
) -> dict[str, Any]:
    """Entrena TF-IDF + LogisticRegression y evalúa validación y test."""
    pipeline = build_baseline_pipeline(seed=SEED)
    pipeline.fit(train_df[text_column], train_df["label"])

    def evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = pipeline.predict(frame[text_column]).astype(int)
        y_prob = pipeline.predict_proba(frame[text_column])[:, 1]
        metrics = classification_metrics(y_true, y_pred, y_prob)
        predictions = pd.DataFrame(
            {
                "pair_id": frame["pair_id"].to_numpy(),
                "source_id": frame["source_id"].to_numpy(),
                "split": frame["split"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "prob_hallucination": y_prob,
                "text_preview": frame[text_column].astype(str).str.slice(0, 300).to_numpy(),
            }
        )
        return metrics, predictions

    val_metrics, val_predictions = evaluate(val_df)
    test_metrics, test_predictions = evaluate(test_df)

    top_features = pd.DataFrame()
    if include_top_features:
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]
        coefficients = classifier.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        top_order = np.argsort(np.abs(coefficients))[::-1][:25]
        top_features = pd.DataFrame(
            {
                "feature": feature_names[top_order],
                "coefficient": coefficients[top_order],
                "abs_coefficient": np.abs(coefficients[top_order]),
            }
        )
        top_features["direction"] = np.where(
            top_features["coefficient"] > 0,
            f"{LABEL_NAMES[1]} (positivo)",
            f"{LABEL_NAMES[0]} (negativo)",
        )

    return {
        "model": model_label,
        "pipeline": pipeline,
        "metrics_validation": val_metrics,
        "metrics_test": test_metrics,
        "predictions_validation": val_predictions,
        "predictions_test": test_predictions,
        "top_features": top_features,
    }


def split_grouped_frame(
    frame: pd.DataFrame,
    seed: int = SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split agrupado genérico para las referencias (sin estructura de pares)."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(frame, frame["label"], frame["group"]))
    train_df = pd.concat(
        [frame.iloc[fold[1]] for fold in folds[: n_splits - 2]]
    ).copy()
    val_df = frame.iloc[folds[n_splits - 2][1]].copy()
    test_df = frame.iloc[folds[n_splits - 1][1]].copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_naive_reference(
    pool: pd.DataFrame,
    n_per_class: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Referencia sin control: muestra balanceada de respuestas completas.

    No empareja por query/contexto ni por longitud. Sirve para cuantificar
    cuánto del rendimiento del Experimento 2 se explica por estos confounds.
    """
    if len(pool[pool["label"] == 0]) < n_per_class or len(
        pool[pool["label"] == 1]
    ) < n_per_class:
        raise RuntimeError("No hay suficientes respuestas únicas para la referencia.")

    negatives = pool[pool["label"] == 0].sample(n=n_per_class, random_state=seed)
    positives = pool[pool["label"] == 1].sample(n=n_per_class, random_state=seed)
    sample = (
        pd.concat([negatives, positives], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    sample["pair_id"] = np.arange(len(sample), dtype=int)
    sample["split"] = "train"
    sample["text"] = sample["text"].astype(str)
    sample["n_words"] = sample["text"].map(count_words)

    train_df, val_df, test_df = split_grouped_frame(sample, seed=seed)
    lexical = train_lexical(
        train_df,
        val_df,
        test_df,
        text_column="text",
        model_label="TF-IDF + LogisticRegression",
        include_top_features=False,
    )
    length_control = evaluate_length_control(train_df, val_df, test_df)

    return {
        "condition": "Sin control",
        "n_per_class": int(n_per_class),
        "n_total": int(len(sample)),
        "lexical": lexical,
        "length_control": length_control,
        "length_words": {
            LABEL_NAMES[label]: {
                "mean": float(group["n_words"].mean()),
                "median": float(group["n_words"].median()),
            }
            for label, group in sample.groupby("label")
        },
    }


def run_paired_full_reference(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Referencia con tema controlado pero sin control de longitud.

    Usa los mismos pares y el mismo split que el experimento principal, pero
    deja las respuestas completas. La diferencia contra el experimento
    principal aísla el efecto del control de longitud.
    """
    lexical = train_lexical(
        train_df,
        val_df,
        test_df,
        text_column="text_full",
        model_label="TF-IDF + LogisticRegression",
        include_top_features=False,
    )
    length_control = evaluate_length_control(
        train_df,
        val_df,
        test_df,
        length_column="n_words_original",
    )
    return {
        "condition": "Tema controlado",
        "n_per_class": int(train_df["label"].value_counts().min()),
        "n_total": int(len(train_df) + len(val_df) + len(test_df)),
        "lexical": lexical,
        "length_control": length_control,
    }


# ---------------------------------------------------------------------------
# 3b. Diagnóstico del artefacto de recorte
# ---------------------------------------------------------------------------

def punctuation_rule_metrics(
    frame: pd.DataFrame,
    text_column: str = "text",
) -> dict[str, Any]:
    """Regla trivial: "si NO termina en puntuación, es alucinación".

    Sobre el recorte ingenuo (sin quitar puntuación) esta regla explota el
    hecho de que la respuesta alucinada suele ser más larga y queda cortada a
    mitad de oración. Sirve como diagnóstico del artefacto, no como modelo.
    """
    ends = frame[text_column].map(ends_with_sentence_punctuation).to_numpy()
    predictions = (~ends).astype(int)
    return classification_metrics(
        frame["label"].to_numpy(dtype=int),
        predictions,
        predictions.astype(float),
    )


def build_naive_truncated_frame(
    pool: pd.DataFrame,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = 120,
) -> pd.DataFrame:
    """Reconstruye el recorte ingenuo del diseño v1 (sin quitar puntuación).

    Se usa solo para cuantificar el artefacto: cada par se corta a
    ``min(limpias, alucinadas)`` palabras, de modo que la respuesta alucinada
    suele terminar a mitad de oración y la limpia en un punto.
    """
    records: list[dict[str, Any]] = []
    for (group, context_key), frame in pool.groupby(["group", "context_key"], sort=False):
        clean = frame[frame["label"] == 0]
        hallucinated = frame[frame["label"] == 1]
        if clean.empty or hallucinated.empty:
            continue
        for clean_row, hall_row in greedy_match_by_length(clean, hallucinated):
            common_length = min(
                int(clean_row["n_words"]), int(hall_row["n_words"]), int(max_words)
            )
            if common_length < int(min_words):
                continue
            clean_text = truncate_words(clean_row["text"], common_length)
            hall_text = truncate_words(hall_row["text"], common_length)
            if clean_text == hall_text:
                continue
            pair_id = len(records) // 2
            for row, text in ((clean_row, clean_text), (hall_row, hall_text)):
                label = int(row["label"])
                records.append(
                    {
                        "pair_id": pair_id,
                        "group": group,
                        "context_key": context_key,
                        "text": text,
                        "n_words": int(common_length),
                        "n_words_original": int(row["n_words"]),
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "source": "phantom",
                        "source_id": f"phantom_naive:{row.name}",
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No se pudo reconstruir el recorte ingenuo.")

    # Un par válido tiene exactamente dos filas, una por clase, con textos
    # distintos. Se descartan los pares que no cumplen eso para que el split
    # balanceado siga siendo válido.
    sizes = frame.groupby("pair_id")["label"].agg(["size", "nunique"])
    bad_pairs = set(sizes[(sizes["size"] != 2) | (sizes["nunique"] != 2)].index)
    duplicated = frame[frame.duplicated(subset=["text", "label"], keep=False)]
    bad_pairs |= set(duplicated["pair_id"])
    frame = frame[~frame["pair_id"].isin(bad_pairs)].copy()
    frame["pair_id"] = frame.groupby(["group", "context_key"]).ngroup()
    return frame.reset_index(drop=True)


def run_artifact_diagnostic(
    pool: pd.DataFrame,
    final_df: pd.DataFrame,
    seed: int = SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, Any]:
    """Cuantifica el artefacto de puntuación y comprueba que quedó neutralizado.

    * ``naive``: recorte v1 sin quitar puntuación. La regla de puntuación y la
      longitud separan las clases casi tan bien como un transformer.
    * ``final``: dataset del experimento, con puntuación eliminada. La misma
      regla debe caer al azar.
    """
    naive = build_naive_truncated_frame(pool)
    naive_train, naive_val, naive_test = split_pairs(naive, seed=seed, n_splits=n_splits)

    naive_length = evaluate_length_control(naive_train, naive_val, naive_test)
    naive_lexical = train_lexical(
        naive_train,
        naive_val,
        naive_test,
        text_column="text",
        model_label="TF-IDF (recorte ingenuo)",
        include_top_features=False,
    )

    return {
        "naive": {
            "condition": "Recorte ingenuo (con puntuación)",
            "pairs": int(naive["pair_id"].nunique()),
            "ends_with_punctuation_rate": {
                LABEL_NAMES[label]: float(
                    group["text"].map(ends_with_sentence_punctuation).mean()
                )
                for label, group in naive.groupby("label")
            },
            "punctuation_rule_all": punctuation_rule_metrics(naive),
            "punctuation_rule_test": punctuation_rule_metrics(naive_test),
            "length_control_test": naive_length["metrics_test"],
            "length_threshold": naive_length["threshold"],
            "lexical_test": naive_lexical["metrics_test"],
        },
        "final": {
            "condition": "Texto final (sin puntuación)",
            "pairs": int(final_df["pair_id"].nunique()),
            "ends_with_punctuation_rate": {
                LABEL_NAMES[label]: float(
                    group["text"].map(ends_with_sentence_punctuation).mean()
                )
                for label, group in final_df.groupby("label")
            },
            "punctuation_rule_all": punctuation_rule_metrics(final_df),
        },
    }


# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------

def build_split_balance_chart(df: pd.DataFrame) -> alt.Chart:
    """Cantidad de ejemplos por split y clase."""
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
                text="Exp. 2b — Dataset pareado y balanceado 50/50",
                subtitle=(
                    "Cada par aporta una respuesta limpia y una alucinada; "
                    "el split agrupado por query mantiene ambos en la misma partición"
                ),
            ),
            width=460,
            height=320,
        )
    )


def build_length_control_chart(df: pd.DataFrame) -> alt.Chart:
    """Longitud antes y después del recorte pareado."""
    before = df[["label_name", "n_words_original"]].copy()
    before = before.rename(columns={"n_words_original": "n_words"})
    before["stage"] = "Antes: respuesta completa"

    after = df[["label_name", "n_words"]].copy()
    after["stage"] = "Después: recorte fijo"

    plot_df = pd.concat([before, after], ignore_index=True)
    rng = np.random.default_rng(SEED)
    plot_df["jitter"] = rng.uniform(-0.25, 0.25, size=len(plot_df))

    box = (
        alt.Chart(plot_df)
        .mark_boxplot(extent="min-max", size=42, median={"color": "black"})
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
        .mark_circle(size=14, opacity=0.25)
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
                alt.Tooltip("label_name:N", title="Clase"),
                alt.Tooltip("n_words:Q", title="Palabras"),
            ],
        )
    )
    return (
        (box + points)
        .properties(
            title=alt.TitleParams(
                text="Exp. 2b — La longitud deja de ser un atajo",
                subtitle=(
                    "Izquierda: las alucinaciones son mucho más largas. "
                    "Derecha: todos los textos tienen exactamente la misma cantidad de tokens y sin puntuación"
                ),
            ),
            width=250,
            height=340,
        )
        .facet(column=alt.Column("stage:N", title=None, sort=[
            "Antes: respuesta completa",
            "Después: recorte fijo",
        ]))
    )


def build_jaccard_chart(
    df: pd.DataFrame,
    max_jaccard: float = DEFAULT_MAX_JACCARD,
    n_excluded: int = 0,
) -> alt.Chart:
    """Solapamiento de vocabulario dentro de cada par retenido."""
    pairs = (
        df.groupby("pair_id")
        .agg(pair_jaccard=("pair_jaccard", "first"))
        .reset_index()
    )
    median_jaccard = float(pairs["pair_jaccard"].median())

    histogram = (
        alt.Chart(pairs)
        .mark_bar(binSpacing=1, opacity=0.85, color=COLOR_NEUTRAL)
        .encode(
            x=alt.X(
                "pair_jaccard:Q",
                bin=alt.Bin(step=0.05),
                title="Jaccard de vocabulario dentro del par",
            ),
            y=alt.Y("count():Q", title="Cantidad de pares"),
            tooltip=[
                alt.Tooltip("pair_jaccard:Q", bin=alt.Bin(step=0.05), title="Jaccard"),
                alt.Tooltip("count():Q", title="Pares"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [max_jaccard]}))
        .mark_rule(color=COLOR_HALLUCINATION, strokeDash=[4, 4], size=2)
        .encode(x="x:Q")
    )
    label = (
        alt.Chart(pd.DataFrame({"x": [max_jaccard], "y": [0], "texto": ["umbral de exclusión"]}))
        .mark_text(align="right", dx=-6, dy=10, color=COLOR_HALLUCINATION, fontSize=10)
        .encode(x="x:Q", y="y:Q", text="texto:N")
    )
    return (histogram + rule + label).properties(
        title=alt.TitleParams(
            text="Exp. 2b — Cuánto se parecen las dos respuestas de un par",
            subtitle=(
                f"Jaccard mediano = {median_jaccard:.2f}. Se descartaron "
                f"{n_excluded} pares con Jaccard ≥ {max_jaccard:.2f}: si el texto "
                "es el mismo, la etiqueta es ambigua"
            ),
        ),
        width=560,
        height=300,
    )


def build_top_features_chart(top_features: pd.DataFrame) -> alt.Chart:
    """Barras de los términos de mayor peso del baseline léxico."""
    order = top_features["feature"].tolist()
    positive_label = f"{LABEL_NAMES[1]} (positivo)"
    negative_label = f"{LABEL_NAMES[0]} (negativo)"
    base = alt.Chart(top_features).encode(
        y=alt.Y("feature:N", sort=order, title=None)
    )
    bars = base.mark_bar(size=14).encode(
        x=alt.X("coefficient:Q", title="Coeficiente LogisticRegression (log-odds)"),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=[positive_label, negative_label],
                range=[COLOR_HALLUCINATION, COLOR_TRUTHFUL],
            ),
            title=None,
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Término"),
            alt.Tooltip("coefficient:Q", format=".4f", title="Coef."),
            alt.Tooltip("direction:N", title="Asociado a"),
        ],
    )
    zero_rule = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    return (bars + zero_rule).properties(
        title=alt.TitleParams(
            text="Exp. 2b — Términos asociados a alucinación vs. respuesta limpia",
            subtitle=(
                "Con tema y longitud controlados, el vocabulario ya no puede "
                "separar por fuente; quedan señales de estilo o contenido"
            ),
        ),
        width=620,
        height=420,
    )


def build_control_ablation_chart(ablation: pd.DataFrame) -> alt.Chart:
    """F1 y MCC con y sin control de tema/longitud."""
    condition_order = ["Sin control", "Tema controlado", "Tema + longitud"]
    model_order = ["TF-IDF (texto)", "Umbral de longitud"]
    metric_order = ["F1 macro", "MCC"]

    bars = (
        alt.Chart(ablation)
        .mark_bar(size=26, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(
                "condition:N",
                title=None,
                sort=condition_order,
                axis=alt.Axis(labelAngle=-15),
            ),
            y=alt.Y(
                "value:Q",
                title="Valor en test",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "metric:N",
                title="Métrica",
                scale=alt.Scale(
                    domain=metric_order,
                    range=[COLOR_NEUTRAL, COLOR_HALLUCINATION],
                ),
            ),
            xOffset=alt.X("metric:N", title=None, sort=metric_order),
            tooltip=[
                alt.Tooltip("condition:N", title="Condición"),
                alt.Tooltip("metric:N", title="Métrica"),
                alt.Tooltip("value:Q", format=".4f", title="Valor"),
            ],
        )
    )
    text = bars.mark_text(dy=-7, fontSize=9).encode(
        text=alt.Text("value:Q", format=".3f"),
        color=alt.value("#222222"),
    )
    return (
        (bars + text)
        .properties(
            title=alt.TitleParams(
                text="Exp. 2b — Qué pasa al controlar tema y longitud",
                subtitle=(
                    "Sin control, el umbral de longitud casi alcanza al modelo. "
                    "Al parear por tema y longitud, el atajo desaparece"
                ),
            ),
            width=280,
            height=340,
        )
        .facet(column=alt.Column("model:N", title=None, sort=model_order))
    )


def build_artifact_chart(rows: pd.DataFrame) -> alt.Chart:
    """F1 macro de reglas triviales y modelos, recorte ingenuo vs. texto final.

    La línea punteada marca el azar de un predictor constante sobre un test
    balanceado (F1 macro = 1/3, MCC = 0). Si una regla trivial queda muy por
    encima de esa línea, el diseño tiene una fuga explotable.
    """
    order = rows["entry"].tolist()
    bars = (
        alt.Chart(rows)
        .mark_bar(size=24, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("entry:N", sort=order, title=None),
            x=alt.X(
                "f1_macro:Q",
                title="F1 macro en test",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "grupo:N",
                title=None,
                scale=alt.Scale(
                    domain=["Recorte ingenuo", "Diseño final"],
                    range=[COLOR_NEUTRAL, COLOR_HALLUCINATION],
                ),
            ),
            tooltip=[
                alt.Tooltip("entry:N", title="Regla / modelo"),
                alt.Tooltip("grupo:N", title="Diseño"),
                alt.Tooltip("f1_macro:Q", format=".4f", title="F1 macro"),
                alt.Tooltip("mcc:Q", format=".4f", title="MCC"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=4, fontSize=9).encode(
        text=alt.Text("f1_macro:Q", format=".3f")
    )
    chance = (
        alt.Chart(pd.DataFrame({"x": [1 / 3]}))
        .mark_rule(color="black", strokeDash=[5, 4], size=2)
        .encode(x="x:Q")
    )
    return (bars + text + chance).properties(
        title=alt.TitleParams(
            text="Exp. 2b — Atajos triviales vs. señal real",
            subtitle=(
                "Con el recorte ingenuo, 'termina en punto' alcanza F1 ≈ 0,92. "
                "Al quitar la puntuación, todas las reglas triviales caen al azar "
                "(línea punteada = 1/3) y solo queda la señal de los modelos"
            ),
        ),
        width=620,
        height=300,
    )


def build_pair_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por par, con longitudes originales y diagnóstico de solapamiento."""
    clean = (
        df[df["label"] == 0]
        .groupby("pair_id")
        .agg(
            query=("query", "first"),
            split=("split", "first"),
            clean_words_original=("n_words_original", "first"),
            truncated_words=("n_words", "first"),
            identical_prefix=("identical_prefix", "first"),
            pair_jaccard=("pair_jaccard", "first"),
        )
    )
    hallucinated = (
        df[df["label"] == 1]
        .groupby("pair_id")
        .agg(hallucination_words_original=("n_words_original", "first"))
    )
    diagnostics = clean.join(hallucinated).reset_index()
    diagnostics["length_ratio"] = (
        diagnostics["hallucination_words_original"]
        / diagnostics["clean_words_original"].clip(lower=1)
    )
    return diagnostics


# ---------------------------------------------------------------------------
# 5. Exportación de resultados
# ---------------------------------------------------------------------------

def metrics_row(
    model: str,
    split: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    cm = metrics["confusion_matrix"]
    return {
        "model": model,
        "split": split,
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
        "pr_auc": metrics["pr_auc"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
    }


def summarize_reference(reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    return {
        "condition": reference["condition"],
        "n_per_class": reference["n_per_class"],
        "n_total": reference["n_total"],
        "lexical_test": reference["lexical"]["metrics_test"],
        "length_control_test": reference["length_control"]["metrics_test"],
        "length_threshold": reference["length_control"]["threshold"],
        "length_words": reference.get("length_words"),
    }


def build_artifact_rows(
    artifact: dict[str, Any],
    final_rule_test: dict[str, Any],
    model_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Tabla tidy con F1 macro de reglas triviales y modelos, por diseño."""
    rows: list[dict[str, Any]] = []

    def add(group: str, entry: str, metrics: dict[str, Any]) -> None:
        rows.append(
            {
                "grupo": group,
                "entry": entry,
                "f1_macro": float(metrics["f1_macro"]),
                "mcc": float(metrics["mcc"]),
                "n": int(metrics["n"]),
            }
        )

    add(
        "Recorte ingenuo",
        "Ingenuo · regla de puntuación",
        artifact["naive"]["punctuation_rule_test"],
    )
    add(
        "Recorte ingenuo",
        "Ingenuo · umbral de longitud",
        artifact["naive"]["length_control_test"],
    )
    add("Recorte ingenuo", "Ingenuo · TF-IDF", artifact["naive"]["lexical_test"])

    add("Diseño final", "Final · regla de puntuación", final_rule_test)
    add(
        "Diseño final",
        "Final · umbral de longitud",
        model_metrics["Umbral de longitud"],
    )
    add(
        "Diseño final",
        "Final · query-only",
        model_metrics["Query-only (diagnóstico)"],
    )
    add(
        "Diseño final",
        "Final · TF-IDF",
        model_metrics["TF-IDF + LogisticRegression"],
    )
    if "DistilBERT fine-tuned" in model_metrics:
        add(
            "Diseño final",
            "Final · DistilBERT",
            model_metrics["DistilBERT fine-tuned"],
        )
    return pd.DataFrame(rows)


def export_results(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lexical: dict[str, Any],
    query_only: dict[str, Any],
    length_control: dict[str, Any],
    transformer: dict[str, Any] | None,
    naive_reference: dict[str, Any] | None,
    paired_full_reference: dict[str, Any] | None,
    artifact: dict[str, Any],
    ablation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
    graphs_dir: Path,
    results_dir: Path,
    save_png: bool,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.concat([train_df, val_df, test_df], ignore_index=True)
    export_columns = [
        "uid",
        "pair_id",
        "split",
        "query",
        "text",
        "n_words",
        "n_words_original",
        "label",
        "label_name",
        "identical_prefix",
        "pair_jaccard",
        "context_preview",
    ]
    available = [column for column in export_columns if column in sample.columns]
    sample[available].to_csv(
        results_dir / "exp2b_balanced_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics = build_pair_diagnostics(sample)
    diagnostics.to_csv(
        results_dir / "exp2b_pair_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lexical["predictions_test"].to_csv(
        results_dir / "exp2b_baseline_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lexical["predictions_validation"].to_csv(
        results_dir / "exp2b_baseline_predictions_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    query_only["predictions_test"].to_csv(
        results_dir / "exp2b_query_only_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    length_control["predictions_test"].to_csv(
        results_dir / "exp2b_length_control_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not lexical["top_features"].empty:
        lexical["top_features"].to_csv(
            results_dir / "exp2b_top_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metric_rows: list[dict[str, Any]] = []
    for model_label, result in (
        ("Umbral de longitud", length_control),
        ("Query-only (diagnóstico)", query_only),
        ("TF-IDF + LogisticRegression", lexical),
    ):
        metric_rows.append(
            metrics_row(model_label, "validation", result["metrics_validation"])
        )
        metric_rows.append(metrics_row(model_label, "test", result["metrics_test"]))

    if transformer is not None:
        transformer["predictions_test"].to_csv(
            results_dir / "exp2b_transformer_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        transformer["training_log"].to_csv(
            results_dir / "exp2b_transformer_training_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metric_rows.append(
            metrics_row("DistilBERT fine-tuned", "test", transformer["metrics_test"])
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        results_dir / "exp2b_metrics.csv", index=False, encoding="utf-8-sig"
    )
    if not ablation.empty:
        ablation.to_csv(
            results_dir / "exp2b_control_ablation.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not sensitivity.empty:
        sensitivity.to_csv(
            results_dir / "exp2b_length_sensitivity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --- Gráficos ----------------------------------------------------------
    charts: list[str] = []
    save_chart(build_split_balance_chart(sample), "exp2b_split_balance", graphs_dir, save_png)
    charts.append("exp2b_split_balance.html")

    save_chart(
        build_length_control_chart(sample),
        "exp2b_length_control",
        graphs_dir,
        save_png,
    )
    charts.append("exp2b_length_control.html")

    save_chart(
        build_jaccard_chart(
            sample,
            max_jaccard=float(metadata["max_jaccard"]),
            n_excluded=int(metadata["skipped_duplicate_pairs"]),
        ),
        "exp2b_pair_overlap",
        graphs_dir,
        save_png,
    )
    charts.append("exp2b_pair_overlap.html")

    if not lexical["top_features"].empty:
        save_chart(
            build_top_features_chart(lexical["top_features"]),
            "exp2b_top_features",
            graphs_dir,
            save_png,
        )
        charts.append("exp2b_top_features.html")

    save_chart(
        build_confusion_matrix_chart(
            lexical["metrics_test"],
            "TF-IDF + LogisticRegression",
            title_prefix="Exp. 2b",
        ),
        "exp2b_confusion_baseline",
        graphs_dir,
        save_png,
    )
    charts.append("exp2b_confusion_baseline.html")

    model_metrics: dict[str, dict[str, Any]] = {
        "Umbral de longitud": length_control["metrics_test"],
        "Query-only (diagnóstico)": query_only["metrics_test"],
        "TF-IDF + LogisticRegression": lexical["metrics_test"],
    }
    prediction_frames = {"TF-IDF + LogisticRegression": lexical["predictions_test"]}

    if transformer is not None:
        save_chart(
            build_confusion_matrix_chart(
                transformer["metrics_test"],
                "DistilBERT fine-tuned",
                title_prefix="Exp. 2b",
            ),
            "exp2b_confusion_transformer",
            graphs_dir,
            save_png,
        )
        charts.append("exp2b_confusion_transformer.html")
        model_metrics["DistilBERT fine-tuned"] = transformer["metrics_test"]
        prediction_frames["DistilBERT fine-tuned"] = transformer["predictions_test"]

        curves = build_training_curves_chart(
            transformer["training_log"], title_prefix="Exp. 2b"
        )
        if curves is not None:
            save_chart(
                curves,
                "exp2b_transformer_training_curves",
                graphs_dir,
                save_png,
            )
            charts.append("exp2b_transformer_training_curves.html")

    save_chart(
        build_model_comparison_chart(model_metrics, title_prefix="Exp. 2b"),
        "exp2b_model_comparison",
        graphs_dir,
        save_png,
    )
    charts.append("exp2b_model_comparison.html")

    save_chart(
        build_probability_chart(prediction_frames, title_prefix="Exp. 2b"),
        "exp2b_probability_distribution",
        graphs_dir,
        save_png,
    )
    charts.append("exp2b_probability_distribution.html")

    artifact_rows = build_artifact_rows(
        artifact,
        punctuation_rule_metrics(test_df),
        model_metrics,
    )
    artifact_rows.to_csv(
        results_dir / "exp2b_artifact_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_chart(
        build_artifact_chart(artifact_rows),
        "exp2b_artifact_diagnostic",
        graphs_dir,
        save_png,
    )
    charts.append("exp2b_artifact_diagnostic.html")

    if not ablation.empty:
        save_chart(
            build_control_ablation_chart(ablation),
            "exp2b_control_ablation",
            graphs_dir,
            save_png,
        )
        charts.append("exp2b_control_ablation.html")

    # --- Resumen JSON ------------------------------------------------------
    summary: dict[str, Any] = {
        "experiment": "exp2b_phantom_paired_control",
        "seed": config["seed"],
        "config": config,
        "dataset": {
            **metadata,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_pairs_total": int(sample["pair_id"].nunique()),
            "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
            "validation_label_counts": val_df["label"].value_counts().sort_index().to_dict(),
            "test_label_counts": test_df["label"].value_counts().sort_index().to_dict(),
        },
        "controls": {
            "length_only": {
                "threshold": length_control["threshold"],
                "metrics_test": length_control["metrics_test"],
            },
            "query_only": {
                "model": query_only["model"],
                "metrics_validation": query_only["metrics_validation"],
                "metrics_test": query_only["metrics_test"],
            },
            "ablation": ablation.to_dict(orient="records") if not ablation.empty else [],
            "artifact_diagnostic": artifact,
        },
        "length_sensitivity": (
            sensitivity.to_dict(orient="records") if not sensitivity.empty else []
        ),
        "models": {
            "lexical": {
                "model": lexical["model"],
                "metrics_validation": lexical["metrics_validation"],
                "metrics_test": lexical["metrics_test"],
                "top_features": (
                    lexical["top_features"].to_dict(orient="records")
                    if not lexical["top_features"].empty
                    else []
                ),
            },
            "transformer": None,
        },
        "references": {
            "naive_uncontrolled": summarize_reference(naive_reference),
            "paired_full_text": summarize_reference(paired_full_reference),
        },
        "limitations": {
            "source_confound_removed": True,
            "topic_confound_controlled": True,
            "length_confound_controlled": True,
            "message": (
                "Todas las filas provienen de Phantom y cada par comparte query y "
                "contexto. Los dos textos del par tienen exactamente la misma "
                "cantidad de tokens y se les quitó la puntuación, así que ni la "
                "fuente, ni el tema, ni la longitud, ni el corte de oración pueden "
                "separar las clases. El rendimiento restante mide señal de "
                "contenido/estilo."
            ),
            "truncation_tradeoff": (
                "El recorte fijo puede eliminar la cola donde aparece la "
                "alucinación. Es intencional: el objetivo es medir si queda señal "
                "cuando se compara la misma cantidad de texto."
            ),
            "pair_selection": (
                "Solo se conservan pares en los que ambas respuestas tienen al "
                f"menos {metadata['target_words']} tokens, y se descartan "
                f"{metadata['skipped_duplicate_pairs']} pares con Jaccard ≥ "
                f"{metadata['max_jaccard']:.2f}. Eso reduce la muestra a "
                f"{metadata['used_pairs']} pares y selecciona respuestas limpias "
                "más largas que la mediana de Phantom."
            ),
            "statistical_power": (
                "Con pocos pares, el test tiene pocos cientos de ejemplos y el F1 "
                "tiene intervalos de confianza amplios. Las diferencias de pocos "
                "puntos porcentuales no deben interpretarse como concluyentes."
            ),
            "not_directly_comparable_to_exp2": (
                "El Experimento 2 mezcla Mu-SHROOM y Phantom. Este experimento usa "
                "solo Phantom, pares de la misma query y textos recortados; sus "
                "métricas no son directamente comparables con las del Exp. 2."
            ),
        },
        "outputs": {
            "graphs_dir": str(graphs_dir),
            "results_dir": str(results_dir),
            "charts": charts,
        },
    }

    if transformer is not None:
        summary["models"]["transformer"] = {
            "model_name": transformer["model_name"],
            "max_length": transformer["max_length"],
            "metrics_test": transformer["metrics_test"],
            "token_lengths": transformer["token_lengths"],
            "best_metric": transformer["best_metric"],
            "best_model_checkpoint": transformer["best_model_checkpoint"],
        }

    with open(results_dir / "exp2b_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)

    return summary


# ---------------------------------------------------------------------------
# 6. Main / CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Experimento 2b: clasificación binaria en Phantom con control de "
            "fuente, tema y longitud. Pares de la misma (query, context) "
            "recortados a igual cantidad de tokens y sin puntuación."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla global.")
    parser.add_argument(
        "--target-words",
        type=int,
        default=DEFAULT_TARGET_WORDS,
        help=(
            "Tokens a los que se recorta cada respuesta. Solo se usan pares "
            "donde ambas respuestas tienen al menos esa cantidad."
        ),
    )
    parser.add_argument(
        "--max-jaccard",
        type=float,
        default=DEFAULT_MAX_JACCARD,
        help="Jaccard máximo permitido dentro de un par (pares casi idénticos se descartan).",
    )
    parser.add_argument(
        "--sensitivity-lengths",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SENSITIVITY_LENGTHS),
        help=(
            "Lista de longitudes para el análisis de sensibilidad con TF-IDF "
            "(vacío = omitir)."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Máximo de pares a usar (0 = todos).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Folds de StratifiedGroupKFold (5 train, 1 val, 1 test).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TRANSFORMER,
        help="Modelo de transformers para el fine-tuning.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Máximo de tokens para DistilBERT.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Épocas de fine-tuning.")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size para fine-tuning."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Usar fp16 en el fine-tuning (GPU)."
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Forzar CPU para el fine-tuning."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token de HuggingFace para modelos gated.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Caché de HuggingFace para el dataset.",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Ejecutar solo los baselines léxicos y los controles.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Omitir las dos referencias de ablación (más rápido).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Guardar el modelo fine-tuneado.",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=root / "graphs" / "exp2b",
        help="Directorio de gráficos.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "exp2b",
        help="Directorio de resultados.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Exportar también PNG (requiere vl-convert-python).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    print_header("EXPERIMENTO 2B — PHANTOM CON CONTROL DE TEMA Y LONGITUD")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {PHANTOM_NAME}")
    print(f"Modelo transformer: {args.model if not args.skip_transformer else 'omitido'}")
    print(
        f"Pares: target_words={args.target_words} | max_jaccard={args.max_jaccard} | "
        f"max_pairs={args.max_pairs or 'todos'}"
    )

    # ------------------------------------------------------------------ [1/6]
    print_section("[1/6] CONSTRUCCIÓN DE PARES (MISMA QUERY Y CONTEXTO)")
    started = time.time()
    df, pool, metadata = load_phantom_pairs(
        cache_dir=args.cache_dir,
        target_words=args.target_words,
        max_jaccard=args.max_jaccard,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(f"  Filas crudas: {metadata['raw_rows']}")
    print(f"  Respuestas únicas (deduplicadas): {metadata['unique_answers']}")
    print(f"  Pares candidatos: {metadata['candidate_pairs']}")
    print(f"  Pares usados: {metadata['used_pairs']}")
    print(
        f"  Pares descartados por no llegar a {metadata['target_words']} tokens: "
        f"{metadata['skipped_short_pairs']}"
    )
    print(
        f"  Pares descartados por ser casi idénticos (Jaccard ≥ "
        f"{metadata['max_jaccard']:.2f}): {metadata['skipped_duplicate_pairs']}"
    )
    print(f"  Ejemplos: {metadata['n_total']} ({metadata['n_per_class']} por clase)")
    print(
        "  Longitud media (antes -> después): "
        + " | ".join(
            f"{label_name}: "
            f"{values['mean_original']:.1f} -> {values['mean']:.1f}"
            for label_name, values in metadata["length_words_by_class"].items()
        )
    )
    print(
        "  [CONTROL] Cada par tiene exactamente "
        f"{metadata['target_words']} tokens en ambas clases y sin puntuación."
    )

    # ------------------------------------------------------------------ [2/6]
    print_section("[2/6] SPLIT AGRUPADO POR QUERY")
    train_df, val_df, test_df = split_pairs(
        df, seed=args.seed, n_splits=args.n_splits
    )
    print(
        f"  Train={len(train_df)} | Validación={len(val_df)} | Test={len(test_df)}"
    )
    print(
        "  Balance por split: "
        + " | ".join(
            f"{name}={frame['label'].value_counts().sort_index().to_dict()}"
            for name, frame in (
                ("train", train_df),
                ("validación", val_df),
                ("test", test_df),
            )
        )
    )
    print(
        f"  Pares: train={train_df['pair_id'].nunique()} | "
        f"validación={val_df['pair_id'].nunique()} | "
        f"test={test_df['pair_id'].nunique()}"
    )
    print("  Sin queries compartidas entre particiones (verificado).")

    # ------------------------------------------------------------------ [3/6]
    print_section("[3/6] MODELOS PRINCIPALES (TEXTO RECORTADO)")
    lexical = train_lexical(
        train_df,
        val_df,
        test_df,
        text_column="text",
        model_label="TF-IDF + LogisticRegression",
    )
    query_only = train_lexical(
        train_df,
        val_df,
        test_df,
        text_column="query",
        model_label="Query-only (diagnóstico)",
        include_top_features=False,
    )
    length_control = evaluate_length_control(train_df, val_df, test_df)

    print("\n  Umbral de longitud (control trivial):")
    print(f"    Umbral elegido en validación: {length_control['threshold']:.2f} palabras")
    print(metrics_table(length_control["metrics_test"]).to_string(index=False))
    print(
        "    Nota: con la longitud constante, este control equivale a un "
        "predictor constante (F1 macro = 1/3, MCC = 0)."
    )

    print("\n  Query-only (diagnóstico de fuga por tema):")
    print(metrics_table(query_only["metrics_test"]).to_string(index=False))

    print("\n  TF-IDF + LogisticRegression (texto recortado):")
    print(metrics_table(lexical["metrics_test"]).to_string(index=False))

    # ------------------------------------------------------------------ [4/6]
    print_section("[4/6] DIAGNÓSTICO DE ATAJOS Y REFERENCIAS")
    artifact = run_artifact_diagnostic(
        pool, df, seed=args.seed, n_splits=args.n_splits
    )
    naive = artifact["naive"]
    final = artifact["final"]
    naive_punct = naive["ends_with_punctuation_rate"]
    print("  Regla trivial: 'si NO termina en punto, es alucinación'.")
    print(
        f"    Recorte ingenuo ({naive['pairs']} pares): "
        f"F1={naive['punctuation_rule_test']['f1_macro']:.4f} | "
        f"MCC={naive['punctuation_rule_test']['mcc']:.4f}"
    )
    print(
        "      Termina en puntuación -> limpia: "
        f"{naive_punct[LABEL_NAMES[0]]:.1%} | "
        f"alucinada: {naive_punct[LABEL_NAMES[1]]:.1%}"
    )
    print(
        f"      Umbral de longitud: F1={naive['length_control_test']['f1_macro']:.4f} | "
        f"TF-IDF: F1={naive['lexical_test']['f1_macro']:.4f}"
    )
    print(
        f"    Texto final sin puntuación ({final['pairs']} pares): "
        f"F1={final['punctuation_rule_all']['f1_macro']:.4f} | "
        f"MCC={final['punctuation_rule_all']['mcc']:.4f}"
    )
    print(
        "      Termina en puntuación: "
        f"{final['ends_with_punctuation_rate'][LABEL_NAMES[0]]:.1%} vs "
        f"{final['ends_with_punctuation_rate'][LABEL_NAMES[1]]:.1%} "
        "(la puntuación se eliminó del texto)"
    )
    print(
        "\n  [OK] El artefacto de puntuación queda neutralizado: la regla que "
        "antes daba F1 ≈ 0,92 ahora cae al azar."
    )

    # --- Sensibilidad a la longitud de recorte ---------------------------
    sensitivity = pd.DataFrame()
    sensitivity_lengths = [
        int(value) for value in str(args.sensitivity_lengths).split(",") if value.strip()
    ]
    if not sensitivity_lengths:
        print("\n  Sensibilidad a --target-words: omitida.")
    else:
        print("\n  Sensibilidad a la longitud de recorte (solo TF-IDF):")
        rows = []
        for length in sensitivity_lengths:
            try:
                frame_l, _, metadata_l = load_phantom_pairs(
                    cache_dir=args.cache_dir,
                    target_words=length,
                    max_jaccard=args.max_jaccard,
                    max_pairs=args.max_pairs,
                    seed=args.seed,
                )
                train_l, val_l, test_l = split_pairs(
                    frame_l, seed=args.seed, n_splits=args.n_splits
                )
                lexical_l = train_lexical(
                    train_l,
                    val_l,
                    test_l,
                    text_column="text",
                    model_label="TF-IDF + LogisticRegression",
                    include_top_features=False,
                )
                length_l = evaluate_length_control(train_l, val_l, test_l)
                rows.append(
                    {
                        "target_words": int(length),
                        "pairs": int(metadata_l["used_pairs"]),
                        "n_test": int(len(test_l)),
                        "f1_macro": float(lexical_l["metrics_test"]["f1_macro"]),
                        "mcc": float(lexical_l["metrics_test"]["mcc"]),
                        "roc_auc": float(lexical_l["metrics_test"]["roc_auc"]),
                        "length_control_f1": float(
                            length_l["metrics_test"]["f1_macro"]
                        ),
                    }
                )
            except RuntimeError as error:
                print(f"    {length:3d} tokens: omitido ({error})")
        sensitivity = pd.DataFrame(rows)
        if not sensitivity.empty:
            print(sensitivity.round(4).to_string(index=False))
            print(
                "  Lectura: si el F1 se mantiene en un rango acotado al cambiar "
                "la longitud, el resultado no depende de un recorte elegido a ojo."
            )

    # --- Referencias de ablación -----------------------------------------
    naive_reference: dict[str, Any] | None = None
    paired_full_reference: dict[str, Any] | None = None
    ablation = pd.DataFrame()

    if args.skip_reference:
        print_section("[4b/6] REFERENCIAS DE ABLACIÓN — OMITIDAS (--skip-reference)")
    else:
        print_section("[4b/6] REFERENCIAS: CUÁNTO EXPLICA CADA CONFOUND")
        n_reference = max(1, len(df) // 2)
        naive_reference = run_naive_reference(
            pool, n_per_class=n_reference, seed=args.seed
        )
        paired_full_reference = run_paired_full_reference(train_df, val_df, test_df)

        main_reference = {
            "condition": "Tema + longitud",
            "lexical": lexical,
            "length_control": length_control,
        }
        rows: list[dict[str, Any]] = []
        for reference in (naive_reference, paired_full_reference, main_reference):
            for model_label, metrics in (
                ("TF-IDF (texto)", reference["lexical"]["metrics_test"]),
                ("Umbral de longitud", reference["length_control"]["metrics_test"]),
            ):
                for metric_label, key in (
                    ("F1 macro", "f1_macro"),
                    ("MCC", "mcc"),
                ):
                    rows.append(
                        {
                            "condition": reference["condition"],
                            "model": model_label,
                            "metric": metric_label,
                            "value": float(metrics[key]),
                        }
                    )
        ablation = pd.DataFrame(rows)

        table = (
            ablation.pivot_table(
                index=["condition", "model"],
                columns="metric",
                values="value",
            )
            .reset_index()
            .rename(columns={"condition": "Condición", "model": "Modelo"})
        )
        print(table.round(4).to_string(index=False))
        print(
            "\n  Lectura: 'Sin control' mezcla tema y longitud; 'Tema controlado' "
            "usa pares de la misma query/contexto pero texto completo; "
            "'Tema + longitud' usa además el recorte fijo sin puntuación."
        )
        print(
            "  Si el umbral de longitud y el query-only caen al azar (F1 macro "
            "≈ 0,33 y MCC = 0), el diseño está controlando los atajos."
        )

    # ------------------------------------------------------------------ [5/6]
    transformer: dict[str, Any] | None = None
    if args.skip_transformer:
        print_section("[5/6] DISTILBERT — OMITIDO (--skip-transformer)")
    else:
        print_section("[5/6] FINE-TUNING DE DISTILBERT")
        transformer = train_transformer(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name=args.model,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            fp16=args.fp16,
            use_cpu=args.cpu,
            output_dir=args.results_dir / "transformer_checkpoints",
            hf_token=args.hf_token,
            save_model=args.save_model,
        )
        print(metrics_table(transformer["metrics_test"]).to_string(index=False))
        token_info = transformer["token_lengths"]
        print(
            f"  Tokens: max={token_info['max']} | p95={token_info['p95']:.0f} | "
            f"truncados={token_info['n_truncated']} "
            f"({token_info['pct_truncated']:.2%})"
        )

    # ------------------------------------------------------------------ [6/6]
    print_section("[6/6] COMPARACIÓN FINAL Y EXPORTACIÓN")
    comparison_rows = [
        {
            "Modelo": "Umbral de longitud",
            "F1 macro": f"{length_control['metrics_test']['f1_macro']:.4f}",
            "MCC": f"{length_control['metrics_test']['mcc']:.4f}",
        },
        {
            "Modelo": "Query-only (diagnóstico)",
            "F1 macro": f"{query_only['metrics_test']['f1_macro']:.4f}",
            "MCC": f"{query_only['metrics_test']['mcc']:.4f}",
        },
        {
            "Modelo": "TF-IDF + LogisticRegression",
            "F1 macro": f"{lexical['metrics_test']['f1_macro']:.4f}",
            "MCC": f"{lexical['metrics_test']['mcc']:.4f}",
        },
    ]
    if transformer is not None:
        comparison_rows.append(
            {
                "Modelo": "DistilBERT fine-tuned",
                "F1 macro": f"{transformer['metrics_test']['f1_macro']:.4f}",
                "MCC": f"{transformer['metrics_test']['mcc']:.4f}",
            }
        )
    print(pd.DataFrame(comparison_rows).to_string(index=False))

    config = {
        "seed": args.seed,
        "target_words": args.target_words,
        "max_jaccard": args.max_jaccard,
        "sensitivity_lengths": sensitivity_lengths,
        "max_pairs": args.max_pairs,
        "n_splits": args.n_splits,
        "model": None if args.skip_transformer else args.model,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fp16": args.fp16,
        "use_cpu": args.cpu,
        "skip_reference": args.skip_reference,
    }

    summary = export_results(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        lexical=lexical,
        query_only=query_only,
        length_control=length_control,
        transformer=transformer,
        naive_reference=naive_reference,
        paired_full_reference=paired_full_reference,
        artifact=artifact,
        ablation=ablation,
        sensitivity=sensitivity,
        metadata=metadata,
        config=config,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        save_png=args.save_png,
    )

    print(f"\n  Gráficos: {summary['outputs']['graphs_dir']}")
    for chart in summary["outputs"]["charts"]:
        print(f"    - {chart}")
    print(f"  Resultados: {summary['outputs']['results_dir']}")
    for filename in (
        "exp2b_balanced_pairs.csv",
        "exp2b_pair_diagnostics.csv",
        "exp2b_baseline_predictions.csv",
        "exp2b_baseline_predictions_validation.csv",
        "exp2b_query_only_predictions.csv",
        "exp2b_length_control_predictions.csv",
        "exp2b_metrics.csv",
        "exp2b_control_ablation.csv",
        "exp2b_artifact_diagnostic.csv",
        "exp2b_length_sensitivity.csv",
        "exp2b_top_features.csv",
        "exp2b_summary.json",
    ):
        if (args.results_dir / filename).exists():
            print(f"    - {filename}")

    elapsed = time.time() - started
    print(f"\n  Tiempo total: {elapsed / 60:.1f} min")
    print_header("EXPERIMENTO 2B FINALIZADO")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# INSTRUCCIONES DE EJECUCIÓN (resumen)
# -----------------------------------------------------------------------------
# 1) Dependencias (las mismas del Experimento 2):
#
#       source .venv/bin/activate
#       pip install -r requirements-exp2.txt
#
# 2) Ejecución completa (pares + referencias + TF-IDF + DistilBERT):
#
#       python experiments/exp2b_phantom_paired_control.py
#
# 3) Solo baselines y controles, sin transformer:
#
#       python experiments/exp2b_phantom_paired_control.py --skip-transformer
#
# 4) Smoke test rápido:
#
#       python experiments/exp2b_phantom_paired_control.py \
#           --max-pairs 60 --skip-transformer
#
# 5) Sin las referencias de ablación (más rápido):
#
#       python experiments/exp2b_phantom_paired_control.py \
#           --skip-reference --skip-transformer
#
# 6) Cambiar el recorte de longitud (tokens por respuesta):
#
#       python experiments/exp2b_phantom_paired_control.py \
#           --target-words 25 --sensitivity-lengths "20,30,40"
#
# 7) Exportar PNG:
#
#       pip install vl-convert-python
#       python experiments/exp2b_phantom_paired_control.py --save-png
#
# Salidas principales:
#   graphs/exp2b/exp2b_split_balance.html
#   graphs/exp2b/exp2b_length_control.html
#   graphs/exp2b/exp2b_pair_overlap.html
#   graphs/exp2b/exp2b_top_features.html
#   graphs/exp2b/exp2b_confusion_baseline.html
#   graphs/exp2b/exp2b_confusion_transformer.html
#   graphs/exp2b/exp2b_model_comparison.html
#   graphs/exp2b/exp2b_probability_distribution.html
#   graphs/exp2b/exp2b_transformer_training_curves.html
#   graphs/exp2b/exp2b_control_ablation.html
#   graphs/exp2b/exp2b_artifact_diagnostic.html
#   results/exp2b/exp2b_balanced_pairs.csv
#   results/exp2b/exp2b_pair_diagnostics.csv
#   results/exp2b/exp2b_metrics.csv
#   results/exp2b/exp2b_control_ablation.csv
#   results/exp2b/exp2b_artifact_diagnostic.csv
#   results/exp2b/exp2b_length_sensitivity.csv
#   results/exp2b/exp2b_summary.json
# -----------------------------------------------------------------------------
