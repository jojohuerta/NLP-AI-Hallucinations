#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 1 — Análisis léxico y de perplejidad (nivel segmento)
==================================================================

Objetivo
--------
Comparar, dentro de las respuestas del dataset Mu-SHROOM (configuración "en"),
los segmentos marcados como alucinación (Corpus A) contra el resto del texto
(Corpus B), desde dos perspectivas complementarias:

1. Léxica: se comparan **Bag-of-Words (CountVectorizer)** y **TF-IDF
   (TfidfVectorizer)** con el mismo pipeline: split agrupado por documento,
   LogisticRegression con ``class_weight="balanced"`` y las mismas métricas de
   test. Para cada enfoque se reportan F1 macro, MCC, precisión y recall por
   clase, matriz de confusión y el Top 15 de palabras por valor absoluto del
   coeficiente.
2. De perplejidad: se pasa el texto ORIGINAL completo por un LM causal (GPT-2
   por defecto) y se calcula la NLL (negative log-likelihood) únicamente sobre
   los tokens que caen dentro de los spans alucinados y sobre los que caen
   fuera. Se reportan la NLL media y la perplejidad media de cada corpus, más
   un t-test apareado a nivel documento sobre la NLL media.

Decisiones metodológicas relevantes
-----------------------------------
* Los offsets de ``hard_labels`` son pares [start, end) a nivel carácter.
* Los spans se normalizan (clamp, orden, merge de solapamientos/adyacencias).
* BoW y TF-IDF usan exactamente el mismo conjunto de segmentos y el mismo split
  train/test: GroupShuffleSplit por ``doc_id`` para evitar leakage entre
  segmentos del mismo documento.
* La perplejidad se calcula con el contexto original: cada token se predice
  usando todos los tokens anteriores del documento, y después se etiqueta como
  A/B según el solapamiento de su span de caracteres con los ``hard_labels``.
* Un token se asigna a A si al menos el 50 % de sus caracteres cae dentro de
  un span alucinado; en caso contrario se asigna a B. Se descartan tokens sin
  caracteres alfanuméricos.
* Todos los splits usan una semilla fija (``--seed``).

Salidas
-------
``graphs/exp1/``   : gráficos Altair en HTML (y PNG opcional).
``results/exp1/``  : CSV y JSON con los números listos para las diapositivas.

Uso
---
Ver el bloque de instrucciones impreso con ``--help`` o el README del TP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import altair as alt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

SEED: int = 42
DATASET_NAME: str = "Helsinki-NLP/mu-shroom"
DATASET_CONFIG: str = "en"
DEFAULT_SPLITS: tuple[str, ...] = ("validation", "test")

COLOR_HALLUCINATION = "#E45756"  # rojo (consistente con el EDA)
COLOR_TRUTHFUL = "#54A24B"       # verde (consistente con el EDA)
COLOR_NEUTRAL = "#4C78A8"        # azul
COLOR_BOW = "#4C78A8"            # azul para Bag-of-Words
COLOR_TFIDF = "#F58518"          # naranja para TF-IDF

TOKEN_PATTERN = r"(?u)\b\w\w+\b"
CONTENT_PATTERN = re.compile(r"[A-Za-z0-9]")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
    except ImportError:
        pass


def project_root() -> Path:
    """Devuelve la raíz del repositorio, independientemente del cwd."""
    return Path(__file__).resolve().parent.parent


def json_default(value: Any) -> Any:
    """Serializa tipos de NumPy/Pandas a JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title.center(78))
    print("=" * 78)


def print_section(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# ---------------------------------------------------------------------------
# 1. Carga del dataset y extracción de segmentos
# ---------------------------------------------------------------------------

def load_mushroom(
    splits: Sequence[str] = DEFAULT_SPLITS,
    cache_dir: str | None = None,
    max_examples: int = 0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Carga Mu-SHROOM y concatena los splits indicados.

    Parameters
    ----------
    splits:
        Splits etiquetados a usar. Por defecto validation + test.
    cache_dir:
        Directorio de caché de HuggingFace (opcional).
    max_examples:
        Si es > 0, submuestrea esa cantidad de documentos (útil para smoke
        tests). El muestreo es aleatorio y reproducible.
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=cache_dir)

    frames: list[pd.DataFrame] = []
    for split in splits:
        if split not in ds:
            raise ValueError(
                f"El split '{split}' no existe. Disponibles: {list(ds.keys())}"
            )
        frame = ds[split].to_pandas()
        frame["split"] = split
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["doc_id"] = df["split"].astype(str) + ":" + df["id"].astype(str)
    df["model_output_text"] = df["model_output_text"].fillna("").astype(str)
    df = df[df["model_output_text"].str.len() > 0].reset_index(drop=True)

    if max_examples and max_examples < len(df):
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    return df


def count_hallucination_spans(raw_labels: Any) -> int:
    """Cuenta spans de forma robusta frente a None/NaN."""
    if raw_labels is None:
        return 0
    try:
        return int(len(raw_labels))
    except TypeError:
        return 0


def normalize_intervals(raw_labels: Any, text_length: int) -> list[tuple[int, int]]:
    """Convierte ``hard_labels`` en intervalos [start, end) válidos y ordenados.

    Se aceptan listas de pares, arrays de NumPy y diccionarios con claves
    ``start``/``end``. Los intervalos se recortan al largo del texto y se
    fusionan si se solapan o son adyacentes.
    """
    if raw_labels is None:
        return []

    if isinstance(raw_labels, np.ndarray):
        raw_labels = raw_labels.tolist()
    if not isinstance(raw_labels, (list, tuple)):
        return []

    intervals: list[tuple[int, int]] = []
    for item in raw_labels:
        if isinstance(item, dict):
            start = item.get("start", item.get("char_start"))
            end = item.get("end", item.get("char_end"))
        else:
            try:
                start, end = item[0], item[1]
            except (TypeError, IndexError, KeyError):
                continue

        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue

        start = max(0, min(start, text_length))
        end = max(0, min(end, text_length))
        if end > start:
            intervals.append((start, end))

    if not intervals:
        return []

    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:  # solapamiento o adyacencia
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def complement_intervals(
    intervals: Sequence[tuple[int, int]], text_length: int
) -> list[tuple[int, int]]:
    """Devuelve el complemento de ``intervals`` dentro de [0, text_length)."""
    if not intervals:
        return [(0, text_length)] if text_length > 0 else []

    complement: list[tuple[int, int]] = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            complement.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < text_length:
        complement.append((cursor, text_length))
    return complement


def trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Recorta espacios en los bordes y devuelve None si el span queda vacío."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def build_segment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame de segmentos A (alucinado) y B (veraz).

    Cada fila es una unidad textual con su ``doc_id`` de origen, el corpus al
    que pertenece, el texto y sus offsets de caracteres en el documento.
    """
    rows: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        text = row.model_output_text
        text_length = len(text)
        intervals = normalize_intervals(row.hard_labels, text_length)
        complement = complement_intervals(intervals, text_length)

        for start, end in intervals:
            trimmed = trim_span(text, start, end)
            if trimmed is not None:
                s, e = trimmed
                rows.append(
                    {
                        "doc_id": row.doc_id,
                        "split": row.split,
                        "corpus": "A",
                        "text": text[s:e],
                        "char_start": s,
                        "char_end": e,
                    }
                )

        for start, end in complement:
            trimmed = trim_span(text, start, end)
            if trimmed is not None:
                s, e = trimmed
                rows.append(
                    {
                        "doc_id": row.doc_id,
                        "split": row.split,
                        "corpus": "B",
                        "text": text[s:e],
                        "char_start": s,
                        "char_end": e,
                    }
                )

    segments = pd.DataFrame(rows)
    if segments.empty:
        raise RuntimeError(
            "No se generó ningún segmento. Revisá el formato de hard_labels."
        )

    segments["text_clean"] = (
        segments["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    segments["has_content"] = segments["text_clean"].str.contains(
        CONTENT_PATTERN, regex=True
    )
    segments["text_key"] = segments["text_clean"].str.lower()

    # Descartar segmentos sin contenido alfanumérico y duplicados exactos
    # dentro del mismo corpus. Los duplicados entre A y B se conservan.
    segments = segments[segments["has_content"]].copy()
    segments = segments.drop_duplicates(subset=["corpus", "text_key"], keep="first")
    segments = segments.reset_index(drop=True)

    if segments.empty:
        raise RuntimeError("Todos los segmentos fueron descartados por falta de contenido.")

    return segments


# ---------------------------------------------------------------------------
# 2. Análisis léxico: TF-IDF + LogisticRegression
# ---------------------------------------------------------------------------

def make_bow_vectorizer() -> CountVectorizer:
    """Vectorizador Bag-of-Words con la misma configuración que el TF-IDF."""
    return CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50_000,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        dtype=np.int64,
    )


def make_tfidf_vectorizer() -> TfidfVectorizer:
    """Vectorizador TF-IDF reproducible y adecuado para textos cortos."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50_000,
        sublinear_tf=True,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        dtype=np.float64,
    )


def make_classifier(seed: int = SEED) -> LogisticRegression:
    """Regresión logística balanceada y determinista."""
    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
    )


def prepare_lexical_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Conserva los segmentos con al menos una feature léxica.

    Se usa el vocabulario del CountVectorizer, que es el mismo que el de
    TF-IDF (mismo tokenizador, stopwords, ``min_df`` y ``max_df``). Así BoW y
    TF-IDF se entrenan y evalúan exactamente sobre las mismas filas.
    """
    vectorizer = make_bow_vectorizer()
    try:
        matrix = vectorizer.fit_transform(segments["text_clean"])
    except ValueError:
        raise RuntimeError(
            "No quedaron segmentos con tokens útiles para el análisis léxico."
        ) from None

    mask = matrix.getnnz(axis=1) > 0
    data = segments.loc[mask].reset_index(drop=True)

    if data.empty:
        raise RuntimeError("No quedaron segmentos con tokens útiles para el análisis léxico.")
    return data


@dataclass
class LexicalResult:
    name: str
    vectorizer_name: str
    segments: pd.DataFrame
    vectorizer: Any
    classifier: LogisticRegression
    coefficients: np.ndarray
    feature_names: np.ndarray
    top_features: pd.DataFrame
    evaluation: dict[str, Any]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Métricas binarias: F1 macro, MCC, precisión/recall por clase y matriz."""
    labels = [0, 1]
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n": int(len(y_true)),
        "n_negative": int(np.sum(y_true == 0)),
        "n_positive": int(np.sum(y_true == 1)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision_no_hallucination": float(precision[0]),
        "recall_no_hallucination": float(recall[0]),
        "precision_hallucination": float(precision[1]),
        "recall_hallucination": float(recall[1]),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }


def make_grouped_split(
    data: pd.DataFrame, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Split train/test agrupado por documento, compartido por BoW y TF-IDF."""
    y = (data["corpus"] == "A").astype(int).to_numpy()
    groups = data["doc_id"].to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Se necesitan las dos clases (A y B) para entrenar.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    return next(splitter.split(data, y, groups=groups))


def evaluate_lexical_model(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    vectorizer_factory: Any,
    name: str,
    vectorizer_name: str,
    seed: int = SEED,
    top_k: int = 15,
) -> LexicalResult:
    """Entrena y evalúa un modelo léxico con el split agrupado ya fijado.

    El modelo de evaluación se ajusta solo con train. Para interpretar los
    coeficientes se ajusta un modelo final con todos los segmentos, igual para
    BoW y TF-IDF.
    """
    train_df = data.iloc[train_idx]
    test_df = data.iloc[test_idx]

    y_train = (train_df["corpus"] == "A").astype(int).to_numpy()
    y_test = (test_df["corpus"] == "A").astype(int).to_numpy()

    eval_vectorizer = vectorizer_factory()
    x_train = eval_vectorizer.fit_transform(train_df["text_clean"])
    x_test = eval_vectorizer.transform(test_df["text_clean"])

    eval_classifier = make_classifier(seed)
    eval_classifier.fit(x_train, y_train)
    y_pred = eval_classifier.predict(x_test)

    evaluation = {
        "n_train_segments": int(len(train_df)),
        "n_test_segments": int(len(test_df)),
        "n_train_documents": int(train_df["doc_id"].nunique()),
        "n_test_documents": int(test_df["doc_id"].nunique()),
        "vocabulary_size_eval": int(len(eval_vectorizer.get_feature_names_out())),
        "metrics_test": classification_metrics(y_test, y_pred),
    }

    # Modelo final para interpretar los coeficientes.
    vectorizer = vectorizer_factory()
    x_all = vectorizer.fit_transform(data["text_clean"])
    classifier = make_classifier(seed)
    classifier.fit(x_all, (data["corpus"] == "A").astype(int))

    coefficients = classifier.coef_[0].astype(float)
    feature_names = vectorizer.get_feature_names_out()

    order = np.argsort(np.abs(coefficients))[::-1][:top_k]
    top = pd.DataFrame(
        {
            "feature": feature_names[order],
            "coefficient": coefficients[order],
            "abs_coefficient": np.abs(coefficients[order]),
        }
    )
    top["direction"] = np.where(
        top["coefficient"] > 0, "A (alucinación)", "B (veraz)"
    )
    evaluation["n_features_full_fit"] = int(len(coefficients))

    return LexicalResult(
        name=name,
        vectorizer_name=vectorizer_name,
        segments=data,
        vectorizer=vectorizer,
        classifier=classifier,
        coefficients=coefficients,
        feature_names=feature_names,
        top_features=top.reset_index(drop=True),
        evaluation=evaluation,
    )


def run_lexical_analysis(
    segments: pd.DataFrame, seed: int = SEED, top_k: int = 15
) -> dict[str, LexicalResult]:
    """Corre BoW y TF-IDF con el mismo conjunto de segmentos y el mismo split."""
    data = prepare_lexical_segments(segments)
    train_idx, test_idx = make_grouped_split(data, seed=seed)

    configs = [
        ("bow", make_bow_vectorizer, "Bag-of-Words", "CountVectorizer"),
        ("tfidf", make_tfidf_vectorizer, "TF-IDF", "TfidfVectorizer"),
    ]

    return {
        key: evaluate_lexical_model(
            data=data,
            train_idx=train_idx,
            test_idx=test_idx,
            vectorizer_factory=factory,
            name=name,
            vectorizer_name=vectorizer_name,
            seed=seed,
            top_k=top_k,
        )
        for key, factory, name, vectorizer_name in configs
    }


# ---------------------------------------------------------------------------
# 3. Análisis de perplejidad por tokens
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str) -> Any:
    """Resuelve 'auto' a cuda/mps/cpu."""
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_language_model(model_name: str, device: Any, hf_token: str | None = None) -> Any:
    """Carga tokenizer y modelo causal en el dispositivo indicado."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import __version__ as transformers_version
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    token_kwargs = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, **token_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    major_version = int(transformers_version.split(".")[0])

    if major_version >= 5:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, **token_kwargs
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, **token_kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, **token_kwargs
            )

    model.to(device)
    model.eval()
    return tokenizer, model


def assign_token_to_corpus(
    start: int,
    end: int,
    intervals: Sequence[tuple[int, int]],
    text: str,
) -> str | None:
    """Asigna un token a A/B según el solapamiento de caracteres con A.

    Devuelve None para tokens sin caracteres alfanuméricos (espacios,
    puntuación, tokens especiales con offset (0, 0), etc.).
    """
    if end <= start:
        return None
    if not CONTENT_PATTERN.search(text[start:end]):
        return None

    token_length = end - start
    overlap = 0
    for interval_start, interval_end in intervals:
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        overlap += min(end, interval_end) - max(start, interval_start)

    return "A" if overlap >= 0.5 * token_length else "B"


def compute_token_nlls(
    text: str,
    intervals: Sequence[tuple[int, int]],
    tokenizer: Any,
    model: Any,
    device: Any,
    max_length: int = 1_024,
) -> dict[str, Any]:
    """Calcula la NLL por token del texto completo y la separa en A/B.

    La NLL del token j se calcula condicionando en todos los tokens previos
    del documento original. Luego se etiqueta el token según su posición de
    caracteres. De este modo, la comparación A vs B es intra-documento.
    """
    import torch

    full_ids = tokenizer(
        text, add_special_tokens=True, truncation=False
    )["input_ids"]
    truncated = len(full_ids) > max_length

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0]  # (T, vocab)
    # logits[t] predice el token t+1. Por eso se shiftea una posición.
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    token_nll = -log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    token_nll = token_nll.detach().cpu().numpy()

    nll_a: list[float] = []
    nll_b: list[float] = []
    n_tokens_skipped = 0

    # token_nll[i] corresponde al token en la posición i+1 de offset_mapping.
    for index, nll in enumerate(token_nll):
        token_index = index + 1
        if token_index >= len(offset_mapping):
            break
        start, end = offset_mapping[token_index]
        corpus = assign_token_to_corpus(start, end, intervals, text)
        if corpus is None:
            n_tokens_skipped += 1
        elif corpus == "A":
            nll_a.append(float(nll))
        else:
            nll_b.append(float(nll))

    return {
        "nll_a": nll_a,
        "nll_b": nll_b,
        "n_tokens_a": len(nll_a),
        "n_tokens_b": len(nll_b),
        "n_tokens_total": len(token_nll),
        "n_tokens_skipped": n_tokens_skipped,
        "truncated": bool(truncated),
    }


def compute_perplexity_table(
    df: pd.DataFrame,
    model_name: str = "gpt2",
    device_arg: str = "auto",
    hf_token: str | None = None,
    max_length: int = 1_024,
    min_tokens_per_corpus: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recorre los documentos y arma la tabla de NLL/perplejidad por doc."""
    device = resolve_device(device_arg)
    print(f"  Dispositivo: {device} | Modelo: {model_name}")

    tokenizer, model = load_language_model(model_name, device, hf_token=hf_token)

    records: list[dict[str, Any]] = []
    truncated_docs = 0
    skipped_low_tokens = 0

    total = len(df)
    for position, row in enumerate(df.itertuples(index=False), start=1):
        text = row.model_output_text
        intervals = normalize_intervals(row.hard_labels, len(text))
        result = compute_token_nlls(
            text=text,
            intervals=intervals,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
        )

        if result["truncated"]:
            truncated_docs += 1

        if (
            result["n_tokens_a"] < min_tokens_per_corpus
            or result["n_tokens_b"] < min_tokens_per_corpus
        ):
            skipped_low_tokens += 1
            continue

        mean_nll_a = float(np.mean(result["nll_a"]))
        mean_nll_b = float(np.mean(result["nll_b"]))

        records.append(
            {
                "doc_id": row.doc_id,
                "split": row.split,
                "n_tokens_a": result["n_tokens_a"],
                "n_tokens_b": result["n_tokens_b"],
                "mean_nll_a": mean_nll_a,
                "mean_nll_b": mean_nll_b,
                "ppl_a": float(math.exp(mean_nll_a)),
                "ppl_b": float(math.exp(mean_nll_b)),
                "delta_nll": mean_nll_a - mean_nll_b,
                "truncated": result["truncated"],
            }
        )

        if position % 25 == 0 or position == total:
            print(f"    Procesados {position}/{total} documentos...")

    table = pd.DataFrame(records)
    metadata = {
        "lm": model_name,
        "device": str(device),
        "n_documents_processed": int(len(df)),
        "n_documents_evaluated": int(len(table)),
        "n_documents_skipped_min_tokens": int(skipped_low_tokens),
        "n_documents_truncated": int(truncated_docs),
        "max_length": int(max_length),
        "min_tokens_per_corpus": int(min_tokens_per_corpus),
    }

    if table.empty:
        raise RuntimeError(
            "Ningún documento tiene suficientes tokens A y B para el test apareado. "
            "Probá con --min-tokens 1 o con más documentos."
        )

    return table, metadata


def paired_statistics(table: pd.DataFrame) -> dict[str, Any]:
    """NLL/perplejidad media y t-test apareado a nivel documento (A vs B)."""
    a = table["mean_nll_a"].to_numpy(dtype=float)
    b = table["mean_nll_b"].to_numpy(dtype=float)

    t_stat, t_pvalue = stats.ttest_rel(a, b, nan_policy="omit")

    return {
        "n_documents_paired": int(len(a)),
        "n_tokens_a_total": int(table["n_tokens_a"].sum()),
        "n_tokens_b_total": int(table["n_tokens_b"].sum()),
        "mean_nll_a": float(np.mean(a)),
        "mean_nll_b": float(np.mean(b)),
        "mean_ppl_a": float(np.exp(np.mean(a))),
        "mean_ppl_b": float(np.exp(np.mean(b))),
        "paired_ttest": {
            "t_statistic": float(t_stat),
            "p_value": float(t_pvalue),
            "degrees_of_freedom": int(len(a) - 1),
        },
    }


# ---------------------------------------------------------------------------
# 4. Gráficos y exportación de resultados
# ---------------------------------------------------------------------------

def save_chart(chart: alt.Chart, stem: str, graphs_dir: Path, save_png: bool = False) -> Path:
    """Guarda un gráfico Altair en HTML (y opcionalmente PNG)."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    html_path = graphs_dir / f"{stem}.html"
    chart.save(html_path)

    if save_png:
        try:
            chart.save(graphs_dir / f"{stem}.png", scale_factor=2.0)
        except Exception as exc:  # vl-convert-python puede no estar instalado
            print(
                f"  [aviso] No se pudo guardar {stem}.png ({exc}). "
                "Instalá 'vl-convert-python' para exportar PNG."
            )

    return html_path


def build_top_features_chart(top: pd.DataFrame, model_name: str) -> alt.Chart:
    """Barras horizontales con el Top 15 de coeficientes de LogisticRegression."""
    order = top["feature"].tolist()
    color_scale = alt.Scale(
        domain=["A (alucinación)", "B (veraz)"],
        range=[COLOR_HALLUCINATION, COLOR_TRUTHFUL],
    )

    bars = alt.Chart(top).mark_bar(size=18).encode(
        y=alt.Y("feature:N", sort=order, title=None),
        x=alt.X("coefficient:Q", title="Coeficiente LogisticRegression (log-odds)"),
        color=alt.Color("direction:N", scale=color_scale, title=None),
        tooltip=[
            alt.Tooltip("feature:N", title="Palabra"),
            alt.Tooltip("coefficient:Q", format=".4f", title="Coeficiente"),
            alt.Tooltip("direction:N", title="Dirección"),
        ],
    )
    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color="gray", strokeDash=[4, 4]
    ).encode(x="x:Q")

    return (
        (bars + zero_rule)
        .properties(
            title=alt.TitleParams(
                text=f"Exp. 1 — {model_name}: Top 15 palabras por |coeficiente|",
                subtitle=(
                    "Positivo = empuja hacia Corpus A (alucinación) · "
                    "negativo = empuja hacia Corpus B (veraz)"
                ),
            ),
            width=640,
            height=420,
        )
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=11)
    )


def build_lexical_metrics_chart(
    lexical_results: dict[str, LexicalResult],
) -> alt.Chart:
    """Compara F1 macro, MCC y precisión/recall por clase entre BoW y TF-IDF."""
    metric_specs = [
        ("f1_macro", "F1 macro"),
        ("mcc", "MCC"),
        ("precision_hallucination", "Precisión · alucinación"),
        ("recall_hallucination", "Recall · alucinación"),
        ("precision_no_hallucination", "Precisión · veraz"),
        ("recall_no_hallucination", "Recall · veraz"),
    ]

    rows = [
        {
            "model": result.name,
            "metric": label,
            "value": float(result.evaluation["metrics_test"][key]),
        }
        for result in lexical_results.values()
        for key, label in metric_specs
    ]
    data = pd.DataFrame(rows)

    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "metric:N",
                sort=[label for _, label in metric_specs],
                title=None,
                axis=alt.Axis(labelAngle=-25),
            ),
            xOffset=alt.X(
                "model:N", sort=[result.name for result in lexical_results.values()]
            ),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="Valor en test"),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=["Bag-of-Words", "TF-IDF"],
                    range=[COLOR_BOW, COLOR_TFIDF],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Modelo"),
                alt.Tooltip("metric:N", title="Métrica"),
                alt.Tooltip("value:Q", format=".4f", title="Valor"),
            ],
        )
        .properties(
            title="Exp. 1 — Comparación léxica: Bag-of-Words vs. TF-IDF",
            width=640,
            height=360,
        )
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=11)
    )


def build_perplexity_plots(
    table: pd.DataFrame, stats_result: dict[str, Any]
) -> list[tuple[str, alt.Chart]]:
    """Genera el boxplot de sorpresa (NLL) y el histograma de su diferencia."""
    rng = np.random.default_rng(SEED)

    long = table.melt(
        id_vars=["doc_id"],
        value_vars=["mean_nll_a", "mean_nll_b"],
        var_name="corpus",
        value_name="mean_nll",
    )
    long["corpus"] = long["corpus"].map(
        {"mean_nll_a": "A · alucinado", "mean_nll_b": "B · veraz"}
    )
    long["jitter"] = rng.uniform(-0.28, 0.28, size=len(long))

    corpus_scale = alt.Scale(
        domain=["A · alucinado", "B · veraz"],
        range=[COLOR_HALLUCINATION, COLOR_TRUTHFUL],
    )

    box = alt.Chart(long).mark_boxplot(
        extent="min-max", size=46, median={"color": "black"}
    ).encode(
        x=alt.X("corpus:N", title=None),
        y=alt.Y("mean_nll:Q", title="Sorpresa del modelo (NLL por token)"),
        color=alt.Color("corpus:N", scale=corpus_scale, legend=None),
    )
    points = alt.Chart(long).mark_circle(size=18, opacity=0.35).encode(
        x=alt.X("corpus:N", title=None),
        xOffset=alt.X("jitter:Q", scale=alt.Scale(domain=[-0.35, 0.35])),
        y=alt.Y("mean_nll:Q"),
        color=alt.Color("corpus:N", scale=corpus_scale, legend=None),
        tooltip=[
            alt.Tooltip("doc_id:N", title="Documento"),
            alt.Tooltip("corpus:N", title="Corpus"),
            alt.Tooltip("mean_nll:Q", format=".4f", title="Sorpresa (NLL media)"),
        ],
    )
    boxplot_chart = (box + points).properties(
        title=alt.TitleParams(
            text="Exp. 1 — Sorpresa del modelo por documento: alucinación vs. veraz",
            subtitle=(
                f"n={stats_result['n_documents_paired']} documentos · "
                f"Perplejidad media A={stats_result['mean_ppl_a']:.2f} vs "
                f"B={stats_result['mean_ppl_b']:.2f} · "
                f"t apareado p={stats_result['paired_ttest']['p_value']:.3e}"
            ),
        ),
        width=460,
        height=380,
    )

    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color="black", strokeWidth=1.5
    ).encode(x="x:Q")
    histogram = alt.Chart(table).mark_bar(color=COLOR_NEUTRAL, opacity=0.85).encode(
        x=alt.X(
            "delta_nll:Q",
            bin=alt.Bin(maxbins=40),
            title="Diferencia de sorpresa = Alucinación − Veraz (NLL por token)",
        ),
        y=alt.Y("count():Q", title="Documentos"),
        tooltip=[alt.Tooltip("count():Q", title="Documentos")],
    )
    histogram_chart = (histogram + zero_rule).properties(
        title=alt.TitleParams(
            text="Exp. 1 — Diferencia de sorpresa (Alucinación − Veraz)",
            subtitle=(
                "Diferencia media="
                f"{float(table['delta_nll'].mean()):+.4f} NLL por token"
            ),
        ),
        width=560,
        height=340,
    )

    return [
        ("exp1_perplexity_boxplot", boxplot_chart),
        ("exp1_perplexity_delta_histogram", histogram_chart),
    ]


def lexical_result_summary(result: LexicalResult) -> dict[str, Any]:
    """Resume un modelo léxico para el JSON, sin estadística extra."""
    return {
        "model": result.name,
        "vectorizer": result.vectorizer_name,
        **result.evaluation,
        "top_features": result.top_features[
            ["feature", "coefficient", "abs_coefficient", "direction"]
        ].to_dict(orient="records"),
    }


def export_results(
    df: pd.DataFrame,
    segments: pd.DataFrame,
    lexical_results: dict[str, LexicalResult],
    perplexity_table: pd.DataFrame | None,
    perplexity_stats: dict[str, Any] | None,
    perplexity_metadata: dict[str, Any] | None,
    graphs_dir: Path,
    results_dir: Path,
    save_png: bool,
) -> dict[str, Any]:
    """Guarda CSV/JSON y gráficos; devuelve el resumen consolidado."""
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # --- CSVs --------------------------------------------------------------
    segments.to_csv(
        results_dir / "exp1_segments.csv", index=False, encoding="utf-8-sig"
    )
    for key, result in lexical_results.items():
        result.top_features.to_csv(
            results_dir / f"exp1_top_features_{key}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --- Gráficos léxicos --------------------------------------------------
    generated_charts: list[str] = []

    metrics_chart = build_lexical_metrics_chart(lexical_results)
    save_chart(metrics_chart, "exp1_lexical_metrics", graphs_dir, save_png=save_png)
    generated_charts.append("exp1_lexical_metrics.html")

    for key, result in lexical_results.items():
        top_chart = build_top_features_chart(result.top_features, result.name)
        stem = f"exp1_top_features_{key}"
        save_chart(top_chart, stem, graphs_dir, save_png=save_png)
        generated_charts.append(f"{stem}.html")

    # --- Perplejidad -------------------------------------------------------
    if perplexity_table is not None and perplexity_stats is not None:
        perplexity_table.to_csv(
            results_dir / "exp1_perplexity_per_document.csv",
            index=False,
            encoding="utf-8-sig",
        )
        for stem, chart in build_perplexity_plots(perplexity_table, perplexity_stats):
            save_chart(chart, stem, graphs_dir, save_png=save_png)
            generated_charts.append(f"{stem}.html")

    first_result = next(iter(lexical_results.values()))

    # --- Resumen JSON ------------------------------------------------------
    summary: dict[str, Any] = {
        "experiment": "exp1_lexical_perplexity",
        "seed": SEED,
        "dataset": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "splits": list(DEFAULT_SPLITS),
            "n_documents": int(len(df)),
            "n_documents_with_hallucination": int(
                df["hard_labels"].apply(count_hallucination_spans).gt(0).sum()
            ),
        },
        "corpora": {
            "n_segments_a": int((segments["corpus"] == "A").sum()),
            "n_segments_b": int((segments["corpus"] == "B").sum()),
            "n_segments_used_lexical": int(len(first_result.segments)),
            "n_documents_used_lexical": int(first_result.segments["doc_id"].nunique()),
        },
        "lexical": {
            "bow": lexical_result_summary(lexical_results["bow"]),
            "tfidf": lexical_result_summary(lexical_results["tfidf"]),
        },
        "perplexity": {
            **(perplexity_metadata or {}),
            **(perplexity_stats or {}),
        },
        "outputs": {
            "graphs_dir": str(graphs_dir),
            "results_dir": str(results_dir),
            "charts": generated_charts,
        },
    }

    with open(results_dir / "exp1_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)

    return summary


# ---------------------------------------------------------------------------
# 5. Presentación por consola
# ---------------------------------------------------------------------------

def print_lexical_results(lexical_results: dict[str, LexicalResult]) -> None:
    print_section("[2/3] ANÁLISIS LÉXICO — BAG-OF-WORDS Y TF-IDF")

    for result in lexical_results.values():
        evaluation = result.evaluation
        metrics = evaluation["metrics_test"]
        matrix = metrics["confusion_matrix"]

        print(f"\n  === {result.name} ({result.vectorizer_name}) ===")
        print(
            f"  Split agrupado por documento: "
            f"{evaluation['n_train_documents']} docs train / "
            f"{evaluation['n_test_documents']} docs test"
        )
        print(
            f"  Segmentos: {evaluation['n_train_segments']} train / "
            f"{evaluation['n_test_segments']} test"
        )
        print(
            f"  Vocabulario (fit de evaluación): "
            f"{evaluation['vocabulary_size_eval']:,} features"
        )
        print(f"  F1 macro en test: {metrics['f1_macro']:.4f}")
        print(f"  MCC en test: {metrics['mcc']:.4f}")
        print(
            f"  Alucinación: precisión={metrics['precision_hallucination']:.4f}, "
            f"recall={metrics['recall_hallucination']:.4f}"
        )
        print(
            f"  Veraz: precisión={metrics['precision_no_hallucination']:.4f}, "
            f"recall={metrics['recall_no_hallucination']:.4f}"
        )
        print(
            f"  Matriz de confusión: TN={matrix['tn']}, FP={matrix['fp']}, "
            f"FN={matrix['fn']}, TP={matrix['tp']}"
        )

        display = result.top_features.copy()
        display["coef"] = display["coefficient"].map(lambda value: f"{value:+.4f}")
        print("\n  Top 15 palabras por |coeficiente|:")
        print(
            display[["feature", "coef", "direction"]]
            .rename(columns={"feature": "palabra", "direction": "dirección"})
            .to_string(index=False)
        )


def print_perplexity_results(
    metadata: dict[str, Any], stats_result: dict[str, Any]
) -> None:
    print("\n  Resultados de perplejidad:")

    print(
        f"  Documentos procesados: {metadata['n_documents_processed']} | "
        f"evaluados: {metadata['n_documents_evaluated']} | "
        f"descartados por pocos tokens: {metadata['n_documents_skipped_min_tokens']} | "
        f"truncados: {metadata['n_documents_truncated']}"
    )
    print(
        f"  Tokens evaluados: A={stats_result['n_tokens_a_total']:,} | "
        f"B={stats_result['n_tokens_b_total']:,}"
    )
    print(
        f"  NLL media: A={stats_result['mean_nll_a']:.4f} | "
        f"B={stats_result['mean_nll_b']:.4f}"
    )
    print(
        f"  Perplejidad media: A={stats_result['mean_ppl_a']:.2f} | "
        f"B={stats_result['mean_ppl_b']:.2f}"
    )
    print(
        f"  T-test apareado: t={stats_result['paired_ttest']['t_statistic']:.4f}, "
        f"gl={stats_result['paired_ttest']['degrees_of_freedom']}, "
        f"p={stats_result['paired_ttest']['p_value']:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Main / CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Experimento 1: análisis léxico (Bag-of-Words y TF-IDF + LogReg) y "
            "de perplejidad (LM causal) de segmentos alucinados vs. veraces "
            "en Mu-SHROOM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla global.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits etiquetados de Mu-SHROOM a combinar.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Submuestrea N documentos (0 = usar todos). Útil para smoke tests.",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="gpt2",
        help="Modelo causal de HuggingFace para perplejidad.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda o mps.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token de HuggingFace (o usá `huggingface-cli login`).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1_024,
        help="Máximo de tokens de contexto para el LM.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Mínimo de tokens A y B por documento para el test apareado.",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Ejecuta solo la parte léxica (no requiere transformers ni descargar LM).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=root / "graphs" / "exp1",
        help="Directorio de salida de gráficos.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "exp1",
        help="Directorio de salida de CSV/JSON.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Caché de HuggingFace para el dataset.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="También exporta PNG (requiere vl-convert-python).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    print_header("EXPERIMENTO 1 — ANÁLISIS LÉXICO Y DE PERPLEJIDAD (NIVEL SEGMENTO)")
    print(f"Seed: {args.seed} | Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"Splits: {' + '.join(args.splits)}")

    # ------------------------------------------------------------------ [1/3]
    print_section("[1/3] CARGA DEL DATASET Y EXTRACCIÓN DE SEGMENTOS")

    df = load_mushroom(
        splits=args.splits,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    n_with_hallucination = int(
        df["hard_labels"].apply(count_hallucination_spans).gt(0).sum()
    )
    mean_words = float(df["model_output_text"].str.split().str.len().mean())

    print(f"  Documentos cargados: {len(df)}")
    print(
        f"  Con al menos una alucinación: {n_with_hallucination} "
        f"({n_with_hallucination / len(df):.1%})"
    )
    print(f"  Sin alucinación: {len(df) - n_with_hallucination}")
    print(f"  Longitud media: {mean_words:.1f} palabras")

    segments = build_segment_dataframe(df)
    n_a = int((segments["corpus"] == "A").sum())
    n_b = int((segments["corpus"] == "B").sum())
    print(f"  Spans alucinados (Corpus A): {n_a}")
    print(f"  Spans veraces (Corpus B): {n_b}")
    print(f"  Documentos representados: {segments['doc_id'].nunique()}")

    # ------------------------------------------------------------------ [2/3]
    lexical_results = run_lexical_analysis(
        segments=segments,
        seed=args.seed,
        top_k=15,
    )
    print_lexical_results(lexical_results)

    # ------------------------------------------------------------------ [3/3]
    perplexity_table: pd.DataFrame | None = None
    perplexity_stats: dict[str, Any] | None = None
    perplexity_metadata: dict[str, Any] | None = None

    if args.skip_perplexity:
        print_section("[3/3] ANÁLISIS DE PERPLEJIDAD — OMITIDO (--skip-perplexity)")
    else:
        print_section("[3/3] ANÁLISIS DE PERPLEJIDAD POR TOKENS")
        perplexity_table, perplexity_metadata = compute_perplexity_table(
            df=df,
            model_name=args.lm,
            device_arg=args.device,
            hf_token=args.hf_token,
            max_length=args.max_length,
            min_tokens_per_corpus=args.min_tokens,
        )
        perplexity_stats = paired_statistics(perplexity_table)
        print_perplexity_results(perplexity_metadata, perplexity_stats)

    # --------------------------------------------------------------- Salidas
    summary = export_results(
        df=df,
        segments=segments,
        lexical_results=lexical_results,
        perplexity_table=perplexity_table,
        perplexity_stats=perplexity_stats,
        perplexity_metadata=perplexity_metadata,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        save_png=args.save_png,
    )

    print_section("ARCHIVOS GENERADOS")
    print(f"  Gráficos: {summary['outputs']['graphs_dir']}")
    for chart in summary["outputs"]["charts"]:
        print(f"    - {chart}")
    print(f"  Resultados: {summary['outputs']['results_dir']}")
    for filename in (
        "exp1_segments.csv",
        "exp1_top_features_bow.csv",
        "exp1_top_features_tfidf.csv",
        "exp1_perplexity_per_document.csv",
        "exp1_summary.json",
    ):
        if (args.results_dir / filename).exists():
            print(f"    - {filename}")

    print_header("EXPERIMENTO 1 FINALIZADO")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# INSTRUCCIONES DE EJECUCIÓN (resumen)
# -----------------------------------------------------------------------------
# 1) Crear/activar el entorno e instalar dependencias:
#
#       python3 -m venv .venv
#       source .venv/bin/activate
#       pip install -r requirements-exp1.txt
#
# 2) Ejecución completa (dataset completo + GPT-2):
#
#       python experiments/exp1_lexical_perplexity.py
#
# 3) Smoke test rápido (sin descargar el LM):
#
#       python experiments/exp1_lexical_perplexity.py --skip-perplexity
#
# 4) Prueba con pocos documentos:
#
#       python experiments/exp1_lexical_perplexity.py --max-examples 20 --min-tokens 1
#
# 5) Modelo más grande / gated (requiere `huggingface-cli login`):
#
#       python experiments/exp1_lexical_perplexity.py --lm google/gemma-2-2b \
#           --max-length 512 --hf-token "$HF_TOKEN"
#
# 6) Exportar también PNG para diapositivas:
#
#       pip install vl-convert-python
#       python experiments/exp1_lexical_perplexity.py --save-png
#
# Salidas:
#   graphs/exp1/exp1_lexical_metrics.html
#   graphs/exp1/exp1_top_features_bow.html
#   graphs/exp1/exp1_top_features_tfidf.html
#   graphs/exp1/exp1_perplexity_boxplot.html
#   graphs/exp1/exp1_perplexity_delta_histogram.html
#   results/exp1/exp1_segments.csv
#   results/exp1/exp1_top_features_bow.csv
#   results/exp1/exp1_top_features_tfidf.csv
#   results/exp1/exp1_perplexity_per_document.csv
#   results/exp1/exp1_summary.json
# -----------------------------------------------------------------------------
