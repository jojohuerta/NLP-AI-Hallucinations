#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 2 — Clasificación binaria de alucinaciones (dataset balanceado)
===========================================================================

Objetivo
--------
Construir un dataset balanceado 50/50 combinando:

* Mu-SHROOM (configuración "en", splits validation + test): todos los
  documentos que tienen al menos un span en ``hard_labels`` se usan como
  ejemplos POSITIVOS (``has_hallucination = 1``). El texto es
  ``model_output_text``.
* Phantom_Hallucination_Detection: se muestrean ejemplos NEGATIVOS
  (``ground_truth_label == "not hallucination"``) hasta igualar la cantidad de
  positivos. El texto es ``answer``.

Luego se comparan dos aproximaciones:

1. Baseline léxico: TF-IDF + LogisticRegression.
2. Fine-tuning semántico: DistilBERT (o BERT) con la Trainer API de
   HuggingFace.

Métricas principales
--------------------
* F1 macro.
* F1 de la clase "alucinación".
* Precisión y recall por clase.
* MCC (Matthews Correlation Coefficient).
* ROC-AUC y PR-AUC como métricas secundarias.
* Matriz de confusión.

IMPORTANTE: las métricas principales son F1 macro y MCC. El script no usa
``classification_report`` y no reporta métricas globales de aciertos.

Decisiones metodológicas
------------------------
* Los negativos de Phantom se deduplican por texto normalizado antes del
  muestreo. El dataset original tiene muchísimos ``answer`` repetidos; sin
  deduplicar, el mismo texto podría caer en train y test.
* El balance es exacto: ``n_negativos = n_positivos``. Si por algún motivo
  Phantom no tuviera suficientes negativos únicos, se submuestrean los
  positivos y se avisa por consola.
* El split es estratificado 70/15/15 (train/validation/test) por etiqueta.
* El baseline y el transformer usan exactamente el mismo train y el mismo test.
* La validación se usa para elegir la mejor época del transformer; el test no
  se toca hasta la evaluación final.
* Todos los procesos usan semilla fija (``--seed``).

Advertencia sobre el diseño del dataset
---------------------------------------
En este experimento la etiqueta está perfectamente confundida con la fuente:
todos los positivos vienen de Mu-SHROOM y todos los negativos de Phantom. Por
eso el script reporta, además de las métricas globales, el recall sobre los
positivos de Mu-SHROOM y la especificidad sobre los negativos de Phantom. Es
una limitación del diseño solicitado, no un bug: un modelo podría explotar
señales de estilo o longitud propias de cada fuente. La comparación con
DistilBERT y el análisis de longitudes ayudan a dimensionar ese riesgo.

Salidas
-------
``graphs/exp2/``  : gráficos Altair en HTML (y PNG opcional).
``results/exp2/`` : CSV, JSON, predicciones y, opcionalmente, el modelo.

Uso
---
Ver el bloque de instrucciones al final del archivo o ejecutar ``--help``.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import math
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import altair as alt
import numpy as np
import pandas as pd
from datasets import disable_progress_bar, load_dataset
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

SEED: int = 42
MUSHROOM_NAME: str = "Helsinki-NLP/mu-shroom"
MUSHROOM_CONFIG: str = "en"
MUSHROOM_SPLITS: tuple[str, ...] = ("validation", "test")
PHANTOM_NAME: str = "seyled/Phantom_Hallucination_Detection"
DEFAULT_TRANSFORMER: str = "distilbert-base-uncased"

COLOR_HALLUCINATION = "#E45756"  # rojo
COLOR_TRUTHFUL = "#54A24B"       # verde
COLOR_NEUTRAL = "#4C78A8"        # azul
COLOR_MUSHROOM = "#F58518"       # naranja
COLOR_PHANTOM = "#72B7B2"        # turquesa

LABEL_NAMES = {0: "Sin alucinación", 1: "Alucinación"}
SOURCE_NAMES = {"mu_shroom": "Mu-SHROOM", "phantom": "Phantom"}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
alt.data_transformers.disable_max_rows()
disable_progress_bar()


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fija random, NumPy, PyTorch y PYTHONHASHSEED."""
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
    """Raíz del repositorio, independiente del cwd."""
    return Path(__file__).resolve().parent.parent


def json_default(value: Any) -> Any:
    """Convierte tipos de NumPy/Pandas a JSON."""
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


def normalize_text_key(text: str) -> str:
    """Normaliza para deduplicar textos."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def count_words(text: str) -> int:
    return len(str(text).split())


# ---------------------------------------------------------------------------
# 1. Carga y construcción del dataset balanceado
# ---------------------------------------------------------------------------

def load_mushroom_positives(cache_dir: str | None = None) -> pd.DataFrame:
    """Carga todos los positivos ingleses de Mu-SHROOM (validation + test)."""
    ds = load_dataset(MUSHROOM_NAME, MUSHROOM_CONFIG, cache_dir=cache_dir)

    frames: list[pd.DataFrame] = []
    for split in MUSHROOM_SPLITS:
        if split not in ds:
            raise ValueError(
                f"El split '{split}' no existe en Mu-SHROOM. "
                f"Disponibles: {list(ds.keys())}"
            )
        frame = ds[split].to_pandas()[["id", "model_output_text", "hard_labels"]].copy()
        frame["original_split"] = split
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["model_output_text"] = df["model_output_text"].fillna("").astype(str)
    df = df[df["model_output_text"].str.strip().str.len() > 0].copy()

    # Positivo = tiene al menos un span anotado.
    df = df[
        df["hard_labels"].apply(
            lambda labels: isinstance(labels, (list, tuple, np.ndarray)) and len(labels) > 0
        )
    ].copy()

    df["text"] = df["model_output_text"]
    df["source"] = "mu_shroom"
    df["source_id"] = "mu:" + df["original_split"].astype(str) + ":" + df["id"].astype(str)
    df["label"] = 1
    df["label_name"] = LABEL_NAMES[1]

    df["text_key"] = df["text"].map(normalize_text_key)
    df = df.drop_duplicates(subset=["text_key"], keep="first").reset_index(drop=True)

    return df[["source_id", "source", "original_split", "text", "text_key", "label", "label_name"]]


def load_phantom_negatives(
    cache_dir: str | None = None,
    min_words: int = 3,
) -> pd.DataFrame:
    """Carga y deduplica los negativos únicos de Phantom.

    No muestrea todavía: devuelve el pool completo de candidatos negativos
    únicos que cumplen ``min_words``. El muestreo se hace después, cuando ya se
    conoce la cantidad de positivos de Mu-SHROOM.
    """
    ds = load_dataset(PHANTOM_NAME, cache_dir=cache_dir)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas().copy()

    if "ground_truth_label" not in df.columns or "answer" not in df.columns:
        raise ValueError(
            "Phantom no tiene las columnas esperadas "
            "('ground_truth_label', 'answer')."
        )

    labels_normalized = (
        df["ground_truth_label"].astype(str).str.strip().str.lower()
    )
    df = df[labels_normalized == "not hallucination"].copy()
    df["answer"] = df["answer"].fillna("").astype(str)
    df = df[df["answer"].str.strip().str.len() > 0].copy()

    df["text"] = df["answer"]
    df["source"] = "phantom"
    if "Unnamed: 0" in df.columns:
        df["source_id"] = "phantom:" + df["Unnamed: 0"].astype(str)
    else:
        df["source_id"] = "phantom:" + df.index.astype(str)

    df["label"] = 0
    df["label_name"] = LABEL_NAMES[0]
    df["text_key"] = df["text"].map(normalize_text_key)

    # Deduplicar antes de muestrear: el dataset original repite muchísimo
    # la misma respuesta para distintas queries/contextos.
    df = df.drop_duplicates(subset=["text_key"], keep="first").copy()
    df["n_words"] = df["text"].map(count_words)
    df = df[df["n_words"] >= min_words].copy()

    return df[
        ["source_id", "source", "text", "text_key", "label", "label_name", "n_words"]
    ].reset_index(drop=True)


def build_balanced_dataset(
    positives: pd.DataFrame,
    negative_pool: pd.DataFrame,
    seed: int = SEED,
    max_per_class: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combina positivos y negativos logrando un balance exacto 50/50."""
    positives = positives.copy()
    n_positives_original = len(positives)

    if max_per_class and max_per_class > 0 and len(positives) > max_per_class:
        positives = positives.sample(n=max_per_class, random_state=seed)

    n_target = len(positives)
    if n_target == 0:
        raise RuntimeError("No hay positivos disponibles.")

    # Si el pool negativo fuera más chico que los positivos, se submuestrean
    # los positivos para no romper el balance 50/50.
    if len(negative_pool) < n_target:
        print(
            f"  [aviso] Solo hay {len(negative_pool)} negativos únicos; "
            f"se submuestrean los positivos de {n_target} a {len(negative_pool)}."
        )
        positives = positives.sample(n=len(negative_pool), random_state=seed)
        n_target = len(negative_pool)

    negatives = negative_pool.sample(n=n_target, random_state=seed).copy()

    df = pd.concat([positives, negatives], ignore_index=True)
    df["n_words"] = df["text"].map(count_words)
    df["n_chars"] = df["text"].str.len()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Última protección: un mismo texto no puede estar en las dos clases.
    duplicated_across_classes = (
        df.groupby("text_key")["label"].nunique().gt(1).sum()
    )
    if duplicated_across_classes:
        print(
            f"  [aviso] {duplicated_across_classes} textos aparecen en ambas clases; "
            "se eliminan para evitar leakage."
        )
        df = df.sort_values("label").drop_duplicates(subset=["text_key"], keep="first")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = df["label"].value_counts().to_dict()
    if counts.get(0, 0) != counts.get(1, 0):
        raise RuntimeError(f"El dataset no quedó balanceado: {counts}")

    metadata = {
        "n_positives_available": int(n_positives_original),
        "n_positives_used": int(len(positives)),
        "n_negatives_pool_unique": int(len(negative_pool)),
        "n_per_class": int(counts[1]),
        "n_total": int(len(df)),
        "min_words_negatives": int(negative_pool["n_words"].min()),
    }
    return df, metadata


def split_dataset(
    df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado 70/15/15 por etiqueta."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 2. Métricas
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Calcula F1 macro, F1 por clase, precisión, recall, MCC y matriz de confusión.

    No devuelve métricas globales de aciertos ni usa
    ``classification_report``.
    """
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

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_negative": int((y_true == 0).sum()),
        "n_positive": int((y_true == 1).sum()),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_hallucination": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_no_hallucination": float(
            f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "precision_no_hallucination": float(precision[0]),
        "precision_hallucination": float(precision[1]),
        "recall_no_hallucination": float(recall[0]),
        "recall_hallucination": float(recall[1]),
        "support_no_hallucination": int(support[0]),
        "support_hallucination": int(support[1]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    # Especificidad y NPV son útiles porque el diseño tiene una fuente por clase.
    metrics["specificity"] = (
        float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    )
    metrics["negative_predictive_value"] = (
        float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    )

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def source_breakdown(
    df: pd.DataFrame,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Recall sobre positivos Mu-SHROOM y especificidad sobre negativos Phantom.

    En este dataset todos los positivos son de Mu-SHROOM y todos los negativos
    de Phantom, así que estas dos métricas resumen el comportamiento por fuente.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    mu_mask = df["source"].to_numpy() == "mu_shroom"
    phantom_mask = df["source"].to_numpy() == "phantom"

    return {
        "mu_shroom_n": int(mu_mask.sum()),
        "phantom_n": int(phantom_mask.sum()),
        "mu_shroom_recall": (
            float(y_pred[mu_mask].mean()) if mu_mask.any() else float("nan")
        ),
        "phantom_specificity": (
            float(1.0 - y_pred[phantom_mask].mean())
            if phantom_mask.any()
            else float("nan")
        ),
    }


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Tabla compacta para imprimir por consola."""
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
            ("ROC-AUC", f"{metrics['roc_auc']:.4f}"),
            ("PR-AUC", f"{metrics['pr_auc']:.4f}"),
            ("Matriz de confusión", f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"),
        ],
        columns=["Métrica", "Valor"],
    )


# ---------------------------------------------------------------------------
# 3. Baseline léxico: TF-IDF + LogisticRegression
# ---------------------------------------------------------------------------

def build_baseline_pipeline(seed: int = SEED) -> Pipeline:
    """Pipeline TF-IDF + regresión logística."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=50_000,
                    sublinear_tf=True,
                    stop_words="english",
                    token_pattern=r"(?u)\b\w\w+\b",
                    dtype=np.float64,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    class_weight=None,  # el dataset ya está balanceado 50/50
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_lexical_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Entrena el baseline y evalúa en validación y test."""
    pipeline = build_baseline_pipeline(seed)
    pipeline.fit(train_df["text"], train_df["label"])

    def evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = pipeline.predict(frame["text"]).astype(int)
        y_prob = pipeline.predict_proba(frame["text"])[:, 1]
        metrics = classification_metrics(y_true, y_pred, y_prob)
        pred_df = pd.DataFrame(
            {
                "source_id": frame["source_id"].to_numpy(),
                "source": frame["source"].to_numpy(),
                "split": frame["split"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "prob_hallucination": y_prob,
                "text_preview": frame["text"].str.slice(0, 300).to_numpy(),
            }
        )
        return metrics, pred_df

    val_metrics, val_predictions = evaluate(val_df)
    test_metrics, test_predictions = evaluate(test_df)

    # Diagnóstico de interpretabilidad: términos de mayor peso absoluto.
    # Si aparecen temas de una fuente (por ejemplo, vocabulario financiero de
    # Phantom vs. entidades de Mu-SHROOM), es señal de confound por fuente.
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    coefficients = classifier.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    top_order = np.argsort(np.abs(coefficients))[::-1][:20]
    top_features = pd.DataFrame(
        {
            "feature": feature_names[top_order],
            "coefficient": coefficients[top_order],
            "abs_coefficient": np.abs(coefficients[top_order]),
        }
    )
    top_features["direction"] = np.where(
        top_features["coefficient"] > 0,
        "Mu-SHROOM (positivo)",
        "Phantom (negativo)",
    )

    return {
        "pipeline": pipeline,
        "metrics_validation": val_metrics,
        "metrics_test": test_metrics,
        "predictions_validation": val_predictions,
        "predictions_test": test_predictions,
        "source_breakdown_test": source_breakdown(test_df, test_predictions["y_pred"]),
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# 4. Fine-tuning semántico con transformers
# ---------------------------------------------------------------------------

def resolve_device() -> Any:
    """Devuelve el dispositivo disponible para logging."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_compute_metrics():
    """Devuelve la función compute_metrics para la Trainer API."""
    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = eval_prediction.label_ids
        preds = np.argmax(logits, axis=-1)

        return {
            "f1_macro": float(
                f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)
            ),
            "f1_hallucination": float(
                f1_score(labels, preds, pos_label=1, zero_division=0)
            ),
            "mcc": float(matthews_corrcoef(labels, preds)),
        }

    return compute_metrics


def make_training_arguments(
    output_dir: Path,
    n_train: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    fp16: bool,
    use_cpu: bool,
) -> Any:
    """Crea TrainingArguments compatible con transformers 4.x y 5.x."""
    import torch
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(round(0.10 * total_steps)))

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "logging_steps": max(1, steps_per_epoch // 2),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "fp16": bool(fp16 and torch.cuda.is_available() and not use_cpu),
        "use_cpu": bool(use_cpu),
        "dataloader_pin_memory": not use_cpu,
        "disable_tqdm": True,
    }

    # transformers 5 eliminó warmup_ratio; 4.x lo acepta.
    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = 0.10
    else:
        kwargs["warmup_steps"] = warmup_steps

    # En v5 el argumento se llama eval_strategy; en 4.x era evaluation_strategy.
    if eval_strategy_key != "eval_strategy":
        kwargs.pop("eval_strategy", None)
        kwargs["evaluation_strategy"] = "epoch"

    # save_only_model reduce el espacio en disco de los checkpoints.
    if "save_only_model" in params:
        kwargs["save_only_model"] = True

    return TrainingArguments(**kwargs)


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_TRANSFORMER,
    max_length: int = 256,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = SEED,
    fp16: bool = False,
    use_cpu: bool = False,
    output_dir: Path | None = None,
    hf_token: str | None = None,
    save_model: bool = False,
) -> dict[str, Any]:
    """Fine-tunea un modelo de clasificación y evalúa en test."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    device = resolve_device()
    print(f"  Dispositivo para fine-tuning: {device}")
    print(f"  Modelo base: {model_name}")

    token_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **token_kwargs)

    max_length = int(max_length)
    token_lengths = [
        len(tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in pd.concat([train_df["text"], val_df["text"], test_df["text"]])
    ]
    n_truncated = int(sum(length > max_length for length in token_lengths))

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    def make_dataset(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(
            frame[["text", "label"]].reset_index(drop=True),
            preserve_index=False,
        )
        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text"],
            desc=None,
        )
        dataset = dataset.rename_column("label", "labels")
        return dataset

    train_dataset = make_dataset(train_df)
    val_dataset = make_dataset(val_df)
    test_dataset = make_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "sin_alucinacion", 1: "alucinacion"},
        label2id={"sin_alucinacion": 0, "alucinacion": 1},
        **token_kwargs,
    )

    output_dir = output_dir or (project_root() / "results" / "exp2" / "distilbert")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = make_training_arguments(
        output_dir=output_dir,
        n_train=len(train_df),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        fp16=fp16,
        use_cpu=use_cpu,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "compute_metrics": make_compute_metrics(),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # La Trainer API imprime diccionarios de logging y barras de progreso.
    # Para que la consola quede prolija, se captura stdout durante train/predict
    # y luego se imprime un resumen corto.
    suppressed_output = io.StringIO()
    with contextlib.redirect_stdout(suppressed_output):
        train_output = trainer.train()

    best_metric = trainer.state.best_metric
    print(
        "  Entrenamiento finalizado. Mejor F1 macro en validación: "
        + (f"{best_metric:.4f}" if best_metric is not None else "N/D")
    )
    print(f"  Mejor checkpoint: {trainer.state.best_model_checkpoint}")

    with contextlib.redirect_stdout(suppressed_output):
        test_output = trainer.predict(test_dataset)
    logits = test_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = softmax(logits, axis=1)[:, 1]
    predictions = np.argmax(logits, axis=1).astype(int)
    y_true = test_df["label"].to_numpy(dtype=int)

    test_metrics = classification_metrics(y_true, predictions, probabilities)
    prediction_df = pd.DataFrame(
        {
            "source_id": test_df["source_id"].to_numpy(),
            "source": test_df["source"].to_numpy(),
            "split": test_df["split"].to_numpy(),
            "y_true": y_true,
            "y_pred": predictions,
            "prob_hallucination": probabilities,
            "text_preview": test_df["text"].str.slice(0, 300).to_numpy(),
        }
    )

    training_log = pd.DataFrame(trainer.state.log_history)

    if save_model:
        final_dir = output_dir / "final_model"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"  Modelo guardado en: {final_dir}")

    return {
        "trainer": trainer,
        "metrics_test": test_metrics,
        "predictions_test": prediction_df,
        "training_log": training_log,
        "train_metrics": train_output.metrics,
        "source_breakdown_test": source_breakdown(test_df, predictions),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "token_lengths": {
            "max": int(max(token_lengths)),
            "p95": float(np.percentile(token_lengths, 95)),
            "n_truncated": n_truncated,
            "pct_truncated": float(n_truncated / len(token_lengths)),
        },
        "model_name": model_name,
        "max_length": max_length,
    }


# ---------------------------------------------------------------------------
# 5. Gráficos
# ---------------------------------------------------------------------------

def save_chart(chart: Any, stem: str, graphs_dir: Path, save_png: bool = False) -> Path:
    """Guarda un gráfico Altair en HTML (y opcionalmente PNG)."""
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


def build_class_balance_chart(df: pd.DataFrame) -> alt.Chart:
    """Cantidad de ejemplos por clase y fuente."""
    counts = (
        df.groupby(["label_name", "source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return (
        alt.Chart(counts)
        .mark_bar(size=70, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
            y=alt.Y("count:Q", title="Cantidad de ejemplos"),
            color=alt.Color(
                "source:N",
                title="Fuente",
                scale=alt.Scale(
                    domain=list(SOURCE_NAMES.keys()),
                    range=[COLOR_MUSHROOM, COLOR_PHANTOM],
                ),
            ),
            tooltip=[
                alt.Tooltip("label_name:N", title="Clase"),
                alt.Tooltip("source:N", title="Fuente"),
                alt.Tooltip("count:Q", title="Ejemplos"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Exp. 2 — Dataset balanceado 50/50",
                subtitle="Todos los positivos vienen de Mu-SHROOM y todos los negativos de Phantom",
            ),
            width=420,
            height=320,
        )
    )


def build_length_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """Distribución de longitud de respuesta por clase/fuente."""
    rng = np.random.default_rng(SEED)
    plot_df = df.copy()
    plot_df["jitter"] = rng.uniform(-0.25, 0.25, size=len(plot_df))

    box = alt.Chart(plot_df).mark_boxplot(
        extent="min-max", size=44, median={"color": "black"}
    ).encode(
        x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
        y=alt.Y("n_words:Q", title="Palabras por respuesta (escala log)", scale=alt.Scale(type="log")),
        color=alt.Color(
            "source:N",
            title="Fuente",
            scale=alt.Scale(
                domain=list(SOURCE_NAMES.keys()),
                range=[COLOR_MUSHROOM, COLOR_PHANTOM],
            ),
        ),
    )
    points = alt.Chart(plot_df).mark_circle(size=16, opacity=0.30).encode(
        x=alt.X("label_name:N", title=None, sort=list(LABEL_NAMES.values())),
        xOffset=alt.X("jitter:Q", scale=alt.Scale(domain=[-0.35, 0.35])),
        y=alt.Y("n_words:Q"),
        color=alt.Color(
            "source:N",
            title="Fuente",
            scale=alt.Scale(
                domain=list(SOURCE_NAMES.keys()),
                range=[COLOR_MUSHROOM, COLOR_PHANTOM],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("source_id:N", title="ID"),
            alt.Tooltip("source:N", title="Fuente"),
            alt.Tooltip("n_words:Q", title="Palabras"),
        ],
    )
    return (box + points).properties(
        title=alt.TitleParams(
            text="Exp. 2 — Longitud de las respuestas por clase",
            subtitle="La longitud es un posible confound entre fuentes; conviene reportarla",
        ),
        width=460,
        height=360,
    )


def build_confusion_matrix_chart(
    metrics: dict[str, Any],
    model_label: str,
    title_prefix: str = "Exp. 2",
) -> alt.Chart:
    """Heatmap de la matriz de confusión."""
    cm = metrics["confusion_matrix"]
    records = []
    for true_label in (0, 1):
        for pred_label in (0, 1):
            key = {
                (0, 0): "tn",
                (0, 1): "fp",
                (1, 0): "fn",
                (1, 1): "tp",
            }[(true_label, pred_label)]
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
        color=alt.Color(
            "casos:Q",
            title="Casos",
            scale=alt.Scale(scheme="blues"),
        ),
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
            text=f"{title_prefix} — Matriz de confusión · {model_label}",
            subtitle=(
                f"F1 macro={metrics['f1_macro']:.4f} · MCC={metrics['mcc']:.4f} · "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            ),
        ),
        width=380,
        height=320,
    )


def build_model_comparison_chart(
    metrics_by_model: dict[str, dict[str, Any]],
    title_prefix: str = "Exp. 2",
) -> alt.Chart:
    """Barras agrupadas comparando F1 macro, MCC y ROC-AUC."""
    records = []
    for model_label, metrics in metrics_by_model.items():
        for metric_label, key in (
            ("F1 macro", "f1_macro"),
            ("MCC", "mcc"),
            ("ROC-AUC", "roc_auc"),
        ):
            records.append(
                {"modelo": model_label, "metrica": metric_label, "valor": metrics[key]}
            )
    plot_df = pd.DataFrame(records)

    bars = alt.Chart(plot_df).mark_bar(size=38, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("modelo:N", title=None),
        xOffset=alt.X("metrica:N", title=None, sort=["F1 macro", "MCC", "ROC-AUC"]),
        y=alt.Y("valor:Q", title="Valor de la métrica", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "metrica:N",
            title="Métrica",
            scale=alt.Scale(
                domain=["F1 macro", "MCC", "ROC-AUC"],
                range=[COLOR_NEUTRAL, "#F2CF5B", "#B279A2"],
            ),
        ),
        tooltip=[
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip("metrica:N", title="Métrica"),
            alt.Tooltip("valor:Q", format=".4f", title="Valor"),
        ],
    )
    text = bars.mark_text(dy=-8, fontSize=10).encode(
        text=alt.Text("valor:Q", format=".3f"),
        color=alt.value("#222222"),
    )
    return (bars + text).properties(
        title=alt.TitleParams(
            text=f"{title_prefix} — Baseline léxico vs. fine-tuning semántico",
            subtitle="Test set balanceado; métricas principales: F1 macro y MCC",
        ),
        width=520,
        height=360,
    )


def build_top_features_chart(top_features: pd.DataFrame) -> alt.Chart:
    """Barras de los términos con mayor peso absoluto del baseline."""
    order = top_features["feature"].tolist()
    color_scale = alt.Scale(
        domain=["Mu-SHROOM (positivo)", "Phantom (negativo)"],
        range=[COLOR_MUSHROOM, COLOR_PHANTOM],
    )
    base = alt.Chart(top_features).encode(
        y=alt.Y("feature:N", sort=order, title=None),
    )
    bars = base.mark_bar(size=15).encode(
        x=alt.X("coefficient:Q", title="Coeficiente LogisticRegression (log-odds)"),
        color=alt.Color("direction:N", scale=color_scale, title=None),
        tooltip=[
            alt.Tooltip("feature:N", title="Término"),
            alt.Tooltip("coefficient:Q", format=".4f", title="Coef."),
            alt.Tooltip("direction:N", title="Asociado a"),
        ],
    )
    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color="gray", strokeDash=[4, 4]
    ).encode(x="x:Q")

    return (bars + zero_rule).properties(
        title=alt.TitleParams(
            text="Exp. 2 — Términos más asociados a cada fuente",
            subtitle=(
                "Diagnóstico de confound: el baseline puede separar temas/estilos "
                "de Mu-SHROOM vs. Phantom en lugar de alucinaciones"
            ),
        ),
        width=620,
        height=420,
    )


def build_probability_chart(
    prediction_frames: dict[str, pd.DataFrame],
    title_prefix: str = "Exp. 2",
) -> alt.Chart:
    """Histograma de probabilidad predicha de alucinación por modelo."""
    frames = []
    for model_label, frame in prediction_frames.items():
        temp = frame.copy()
        temp["modelo"] = model_label
        temp["clase_real"] = temp["y_true"].map(LABEL_NAMES)
        frames.append(temp[["modelo", "clase_real", "prob_hallucination"]])
    plot_df = pd.concat(frames, ignore_index=True)

    return (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.55, binSpacing=1)
        .encode(
            x=alt.X(
                "prob_hallucination:Q",
                bin=alt.Bin(maxbins=30),
                title="P(alucinación) predicha",
            ),
            y=alt.Y("count():Q", title="Cantidad de ejemplos", stack=None),
            color=alt.Color(
                "clase_real:N",
                title="Clase real",
                scale=alt.Scale(
                    domain=list(LABEL_NAMES.values()),
                    range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION],
                ),
            ),
            tooltip=[
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("clase_real:N", title="Clase real"),
                alt.Tooltip("count():Q", title="Ejemplos"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text=f"{title_prefix} — Distribución de probabilidades predichas",
                subtitle="Un buen modelo separa las distribuciones verde y roja",
            ),
            width=620,
            height=260,
        )
        .facet(row=alt.Row("modelo:N", title=None))
        .resolve_scale(y="independent")
    )


def build_training_curves_chart(
    training_log: pd.DataFrame,
    title_prefix: str = "Exp. 2",
) -> alt.Chart | None:
    """Curvas de F1 macro y MCC en validación por época."""
    if training_log.empty or "eval_f1_macro" not in training_log.columns:
        return None

    eval_log = training_log.dropna(subset=["eval_f1_macro"]).copy()
    if eval_log.empty:
        return None

    if "epoch" not in eval_log.columns:
        eval_log["epoch"] = np.arange(1, len(eval_log) + 1)

    plot_df = eval_log.melt(
        id_vars=["epoch"],
        value_vars=["eval_f1_macro", "eval_mcc"],
        var_name="metrica",
        value_name="valor",
    )
    metric_names = {"eval_f1_macro": "F1 macro", "eval_mcc": "MCC"}
    plot_df["metrica"] = plot_df["metrica"].map(metric_names)

    return (
        alt.Chart(plot_df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Época", axis=alt.Axis(format=".0f")),
            y=alt.Y("valor:Q", title="Valor en validación", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metrica:N",
                title="Métrica",
                scale=alt.Scale(domain=["F1 macro", "MCC"], range=[COLOR_NEUTRAL, "#F2CF5B"]),
            ),
            tooltip=[
                alt.Tooltip("epoch:Q", format=".2f", title="Época"),
                alt.Tooltip("metrica:N", title="Métrica"),
                alt.Tooltip("valor:Q", format=".4f", title="Valor"),
            ],
        )
        .properties(
            title=f"{title_prefix} — Evolución del transformer en validación",
            width=480,
            height=300,
        )
    )


# ---------------------------------------------------------------------------
# 6. Exportación de resultados
# ---------------------------------------------------------------------------

def metrics_to_dataframe(metrics: dict[str, Any], model_label: str) -> pd.DataFrame:
    """Convierte métricas a una fila de DataFrame."""
    cm = metrics["confusion_matrix"]
    return pd.DataFrame(
        [
            {
                "model": model_label,
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
                "specificity": metrics["specificity"],
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tp": cm["tp"],
            }
        ]
    )


def export_results(
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline: dict[str, Any],
    transformer: dict[str, Any] | None,
    dataset_metadata: dict[str, Any],
    graphs_dir: Path,
    results_dir: Path,
    save_png: bool,
) -> dict[str, Any]:
    """Guarda CSV, JSON y gráficos; devuelve el resumen consolidado."""
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # --- CSVs --------------------------------------------------------------
    dataset_with_split = pd.concat([train_df, val_df, test_df], ignore_index=True)
    dataset_with_split.to_csv(
        results_dir / "exp2_balanced_dataset.csv", index=False, encoding="utf-8-sig"
    )
    baseline["predictions_test"].to_csv(
        results_dir / "exp2_baseline_predictions.csv", index=False, encoding="utf-8-sig"
    )
    baseline["predictions_validation"].to_csv(
        results_dir / "exp2_baseline_predictions_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline["top_features"].to_csv(
        results_dir / "exp2_baseline_top_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_frames = [
        metrics_to_dataframe(baseline["metrics_test"], "TF-IDF + LogisticRegression")
    ]

    if transformer is not None:
        transformer["predictions_test"].to_csv(
            results_dir / "exp2_transformer_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        transformer["training_log"].to_csv(
            results_dir / "exp2_transformer_training_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics_frames.append(
            metrics_to_dataframe(transformer["metrics_test"], "DistilBERT fine-tuned")
        )

    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    metrics_df.to_csv(results_dir / "exp2_metrics_comparison.csv", index=False, encoding="utf-8-sig")

    # --- Gráficos ----------------------------------------------------------
    charts = []
    save_chart(build_class_balance_chart(dataset), "exp2_class_balance", graphs_dir, save_png)
    charts.append("exp2_class_balance.html")

    save_chart(build_length_distribution_chart(dataset), "exp2_answer_length_distribution", graphs_dir, save_png)
    charts.append("exp2_answer_length_distribution.html")

    save_chart(
        build_top_features_chart(baseline["top_features"]),
        "exp2_baseline_top_features",
        graphs_dir,
        save_png,
    )
    charts.append("exp2_baseline_top_features.html")

    baseline_cm = build_confusion_matrix_chart(
        baseline["metrics_test"], "TF-IDF + LogisticRegression"
    )
    save_chart(baseline_cm, "exp2_confusion_baseline", graphs_dir, save_png)
    charts.append("exp2_confusion_baseline.html")

    model_metrics = {"TF-IDF + LogisticRegression": baseline["metrics_test"]}
    prediction_frames = {"TF-IDF + LogisticRegression": baseline["predictions_test"]}

    if transformer is not None:
        transformer_cm = build_confusion_matrix_chart(
            transformer["metrics_test"], "DistilBERT fine-tuned"
        )
        save_chart(transformer_cm, "exp2_confusion_transformer", graphs_dir, save_png)
        charts.append("exp2_confusion_transformer.html")

        model_metrics["DistilBERT fine-tuned"] = transformer["metrics_test"]
        prediction_frames["DistilBERT fine-tuned"] = transformer["predictions_test"]

        curves = build_training_curves_chart(transformer["training_log"])
        if curves is not None:
            save_chart(curves, "exp2_transformer_training_curves", graphs_dir, save_png)
            charts.append("exp2_transformer_training_curves.html")

    comparison = build_model_comparison_chart(model_metrics)
    save_chart(comparison, "exp2_model_comparison", graphs_dir, save_png)
    charts.append("exp2_model_comparison.html")

    probability_chart = build_probability_chart(prediction_frames)
    save_chart(probability_chart, "exp2_probability_distribution", graphs_dir, save_png)
    charts.append("exp2_probability_distribution.html")

    # --- JSON --------------------------------------------------------------
    summary: dict[str, Any] = {
        "experiment": "exp2_binary_classification",
        "seed": SEED,
        "dataset": {
            "mushroom_name": MUSHROOM_NAME,
            "mushroom_config": MUSHROOM_CONFIG,
            "mushroom_splits": list(MUSHROOM_SPLITS),
            "phantom_name": PHANTOM_NAME,
            **dataset_metadata,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
            "validation_label_counts": val_df["label"].value_counts().sort_index().to_dict(),
            "test_label_counts": test_df["label"].value_counts().sort_index().to_dict(),
            "length_words_by_class": {
                LABEL_NAMES[int(label)]: {
                    "mean": float(group["n_words"].mean()),
                    "median": float(group["n_words"].median()),
                    "max": int(group["n_words"].max()),
                }
                for label, group in dataset.groupby("label")
            },
        },
        "baseline": {
            "model": "TF-IDF + LogisticRegression",
            "metrics_validation": baseline["metrics_validation"],
            "metrics_test": baseline["metrics_test"],
            "source_breakdown_test": baseline["source_breakdown_test"],
            "top_features": baseline["top_features"].to_dict(orient="records"),
        },
        "transformer": None,
        "comparison": {
            "f1_macro_baseline": baseline["metrics_test"]["f1_macro"],
            "mcc_baseline": baseline["metrics_test"]["mcc"],
        },
        "limitations": {
            "label_source_confound": True,
            "message": (
                "Todos los positivos provienen de Mu-SHROOM y todos los negativos "
                "de Phantom. Un F1/MCC muy alto puede reflejar separación de "
                "fuente, tema o estilo, no detección genuina de alucinaciones."
            ),
            "diagnostics": [
                "exp2_baseline_top_features.html muestra los términos que separan las fuentes.",
                "exp2_answer_length_distribution.html muestra el confound de longitud.",
                "Los positivos de Mu-SHROOM y los negativos de Phantom no comparten distribución.",
            ],
        },
        "outputs": {
            "graphs_dir": str(graphs_dir),
            "results_dir": str(results_dir),
            "charts": charts,
        },
    }

    if transformer is not None:
        summary["transformer"] = {
            "model_name": transformer["model_name"],
            "max_length": transformer["max_length"],
            "metrics_test": transformer["metrics_test"],
            "source_breakdown_test": transformer["source_breakdown_test"],
            "train_metrics": transformer["train_metrics"],
            "best_model_checkpoint": transformer["best_model_checkpoint"],
            "best_metric": transformer["best_metric"],
            "token_lengths": transformer["token_lengths"],
        }
        summary["comparison"].update(
            {
                "f1_macro_transformer": transformer["metrics_test"]["f1_macro"],
                "mcc_transformer": transformer["metrics_test"]["mcc"],
                "delta_f1_macro": (
                    transformer["metrics_test"]["f1_macro"]
                    - baseline["metrics_test"]["f1_macro"]
                ),
                "delta_mcc": (
                    transformer["metrics_test"]["mcc"]
                    - baseline["metrics_test"]["mcc"]
                ),
            }
        )

    with open(results_dir / "exp2_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)

    with open(results_dir / "exp2_baseline_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "validation": baseline["metrics_validation"],
                "test": baseline["metrics_test"],
                "source_breakdown_test": baseline["source_breakdown_test"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )

    if transformer is not None:
        with open(results_dir / "exp2_transformer_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": transformer["model_name"],
                    "test": transformer["metrics_test"],
                    "source_breakdown_test": transformer["source_breakdown_test"],
                    "token_lengths": transformer["token_lengths"],
                },
                handle,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )

    return summary


# ---------------------------------------------------------------------------
# 7. Presentación por consola
# ---------------------------------------------------------------------------

def print_dataset_summary(
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    print_section("[1/3] CONSTRUCCIÓN DEL DATASET BALANCEADO")
    print(f"  Positivos disponibles (Mu-SHROOM): {metadata['n_positives_available']}")
    print(f"  Pool de negativos únicos (Phantom): {metadata['n_negatives_pool_unique']}")
    print(f"  Ejemplos por clase: {metadata['n_per_class']}")
    print(f"  Total: {metadata['n_total']}")
    print(
        f"  Split: train={len(train_df)} | validation={len(val_df)} | test={len(test_df)}"
    )
    print(
        "  Balance por split: "
        + " | ".join(
            f"{name}={frame['label'].value_counts().sort_index().to_dict()}"
            for name, frame in (("train", train_df), ("val", val_df), ("test", test_df))
        )
    )
    print(
        "  Longitud media (palabras): "
        + " | ".join(
            f"{LABEL_NAMES[int(label)]}={group['n_words'].mean():.1f}"
            for label, group in dataset.groupby("label")
        )
    )
    print(
        "  [ADVERTENCIA] Etiqueta y fuente están perfectamente confundidas: "
        "positivos=Mu-SHROOM, negativos=Phantom."
    )
    print(
        "  Un F1 muy alto puede estar midiendo separación de fuente/tema/estilo, "
        "no detección genuina de alucinaciones."
    )


def print_model_metrics(
    model_label: str,
    metrics: dict[str, Any],
    breakdown: dict[str, Any],
) -> None:
    print(f"\n  {model_label}")
    print(metrics_table(metrics).to_string(index=False))
    print(
        f"  Recall sobre positivos Mu-SHROOM: {breakdown['mu_shroom_recall']:.4f} "
        f"(n={breakdown['mu_shroom_n']})"
    )
    print(
        f"  Especificidad sobre negativos Phantom: {breakdown['phantom_specificity']:.4f} "
        f"(n={breakdown['phantom_n']})"
    )


# ---------------------------------------------------------------------------
# 8. Main / CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Experimento 2: clasificación binaria de alucinaciones con dataset "
            "balanceado Mu-SHROOM + Phantom. Baseline TF-IDF/LogReg y fine-tuning "
            "de DistilBERT. Métricas: F1 macro, MCC y matriz de confusión."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla global.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Limita la cantidad de ejemplos por clase (0 = usar todos). Útil para smoke tests.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Mínimo de palabras para aceptar un negativo de Phantom.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TRANSFORMER,
        help="Modelo de HuggingFace para fine-tuning.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Máximo de tokens para el transformer.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Épocas de fine-tuning.")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size de entrenamiento."
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate del fine-tuning."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Usar fp16 en GPU (más rápido; requiere soporte de la GPU).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forzar entrenamiento en CPU.",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Ejecutar solo el baseline léxico (no descarga DistilBERT).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Guardar el modelo fine-tuneado y el tokenizer al final.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token de HuggingFace para modelos gated (o usá `huggingface-cli login`).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=root / "graphs" / "exp2",
        help="Directorio de gráficos.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "exp2",
        help="Directorio de resultados.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Caché de HuggingFace para los datasets.",
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

    print_header("EXPERIMENTO 2 — CLASIFICACIÓN BINARIA DE ALUCINACIONES")
    print(f"Seed: {args.seed}")
    print(f"Mu-SHROOM: {MUSHROOM_NAME} ({MUSHROOM_CONFIG})")
    print(f"Phantom:   {PHANTOM_NAME}")

    # ------------------------------------------------------------------ [1/3]
    positives = load_mushroom_positives(cache_dir=args.cache_dir)
    negative_pool = load_phantom_negatives(
        cache_dir=args.cache_dir,
        min_words=args.min_words,
    )
    dataset, dataset_metadata = build_balanced_dataset(
        positives=positives,
        negative_pool=negative_pool,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    train_df, val_df, test_df = split_dataset(dataset, seed=args.seed)

    print_dataset_summary(dataset, train_df, val_df, test_df, dataset_metadata)

    # ------------------------------------------------------------------ [2/3]
    print_section("[2/3] BASELINE LÉXICO — TF-IDF + LOGISTIC REGRESSION")
    baseline = train_lexical_baseline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        seed=args.seed,
    )
    print_model_metrics(
        "TF-IDF + LogisticRegression (test)",
        baseline["metrics_test"],
        baseline["source_breakdown_test"],
    )
    print("\n  Top 10 términos del baseline (diagnóstico de confound):")
    print(
        baseline["top_features"][["feature", "coefficient", "direction"]]
        .head(10)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------ [3/3]
    transformer: dict[str, Any] | None = None
    if args.skip_transformer:
        print_section("[3/3] FINE-TUNING — OMITIDO (--skip-transformer)")
    else:
        print_section("[3/3] FINE-TUNING SEMÁNTICO CON TRANSFORMERS")
        transformer = train_transformer(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name=args.model,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            fp16=args.fp16,
            use_cpu=args.cpu,
            output_dir=args.results_dir / "transformer_checkpoints",
            hf_token=args.hf_token,
            save_model=args.save_model,
        )
        print_model_metrics(
            f"{args.model} fine-tuned (test)",
            transformer["metrics_test"],
            transformer["source_breakdown_test"],
        )
        print(
            f"  Tokens: max={transformer['token_lengths']['max']} | "
            f"p95={transformer['token_lengths']['p95']:.0f} | "
            f"truncados={transformer['token_lengths']['n_truncated']} "
            f"({transformer['token_lengths']['pct_truncated']:.1%})"
        )

    # ---------------------------------------------------------------- Salidas
    summary = export_results(
        dataset=dataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        baseline=baseline,
        transformer=transformer,
        dataset_metadata=dataset_metadata,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        save_png=args.save_png,
    )

    print_section("COMPARACIÓN FINAL (TEST)")
    comparison_rows = [
        {
            "Modelo": "TF-IDF + LogReg",
            "F1 macro": f"{baseline['metrics_test']['f1_macro']:.4f}",
            "MCC": f"{baseline['metrics_test']['mcc']:.4f}",
            "ROC-AUC": f"{baseline['metrics_test']['roc_auc']:.4f}",
        }
    ]
    if transformer is not None:
        comparison_rows.append(
            {
                "Modelo": "DistilBERT fine-tuned",
                "F1 macro": f"{transformer['metrics_test']['f1_macro']:.4f}",
                "MCC": f"{transformer['metrics_test']['mcc']:.4f}",
                "ROC-AUC": f"{transformer['metrics_test']['roc_auc']:.4f}",
            }
        )
    print(pd.DataFrame(comparison_rows).to_string(index=False))

    print_section("ARCHIVOS GENERADOS")
    print(f"  Gráficos: {summary['outputs']['graphs_dir']}")
    for chart in summary["outputs"]["charts"]:
        print(f"    - {chart}")
    print(f"  Resultados: {summary['outputs']['results_dir']}")
    for filename in (
        "exp2_balanced_dataset.csv",
        "exp2_metrics_comparison.csv",
        "exp2_baseline_predictions.csv",
        "exp2_transformer_predictions.csv",
        "exp2_baseline_metrics.json",
        "exp2_transformer_metrics.json",
        "exp2_summary.json",
    ):
        if (args.results_dir / filename).exists():
            print(f"    - {filename}")

    print_header("EXPERIMENTO 2 FINALIZADO")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# INSTRUCCIONES DE EJECUCIÓN (resumen)
# -----------------------------------------------------------------------------
# 1) Instalar dependencias:
#
#       source .venv/bin/activate
#       pip install -r requirements-exp2.txt
#
# 2) Ejecución completa:
#
#       python experiments/exp2_binary_classification.py
#
# 3) Smoke test sin transformer:
#
#       python experiments/exp2_binary_classification.py --skip-transformer
#
# 4) Smoke test con pocos ejemplos:
#
#       python experiments/exp2_binary_classification.py --max-per-class 20
#
# 5) Fine-tuning con BERT base:
#
#       python experiments/exp2_binary_classification.py --model bert-base-uncased
#
# 6) Exportar PNG:
#
#       pip install vl-convert-python
#       python experiments/exp2_binary_classification.py --save-png
#
# Salidas principales:
#   graphs/exp2/exp2_class_balance.html
#   graphs/exp2/exp2_answer_length_distribution.html
#   graphs/exp2/exp2_confusion_baseline.html
#   graphs/exp2/exp2_confusion_transformer.html
#   graphs/exp2/exp2_model_comparison.html
#   graphs/exp2/exp2_probability_distribution.html
#   results/exp2/exp2_balanced_dataset.csv
#   results/exp2/exp2_metrics_comparison.csv
#   results/exp2/exp2_baseline_predictions.csv
#   results/exp2/exp2_transformer_predictions.csv
#   results/exp2/exp2_summary.json
# -----------------------------------------------------------------------------
