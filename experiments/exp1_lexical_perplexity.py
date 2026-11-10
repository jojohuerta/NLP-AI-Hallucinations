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

1. Léxica: TF-IDF + LogisticRegression. Se inspeccionan los coeficientes del
   modelo y se estima su incertidumbre mediante bootstrap agrupado por
   documento. También se reporta un t-test de una muestra sobre los pesos.
2. De perplejidad: se pasa el texto ORIGINAL completo por un LM causal (GPT-2
   por defecto) y se calcula la NLL (negative log-likelihood) únicamente sobre
   los tokens que caen dentro de los spans alucinados y sobre los que caen
   fuera. Luego se hace un t-test apareado a nivel documento.

Decisiones metodológicas relevantes
-----------------------------------
* Los offsets de ``hard_labels`` son pares [start, end) a nivel carácter.
* Los spans se normalizan (clamp, orden, merge de solapamientos/adyacencias).
* El split train/test de TF-IDF es GroupShuffleSplit por ``doc_id`` para evitar
  leakage entre segmentos del mismo documento.
* La perplejidad se calcula con el contexto original: cada token se predice
  usando todos los tokens anteriores del documento, y después se etiqueta como
  A/B según el solapamiento de su span de caracteres con los ``hard_labels``.
* Un token se asigna a A si al menos el 50 % de sus caracteres cae dentro de
  un span alucinado; en caso contrario se asigna a B. Se descartan tokens sin
  caracteres alfanuméricos.
* Todos los remuestreos y splits usan una semilla fija (``--seed``).

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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

def make_vectorizer() -> TfidfVectorizer:
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


def filter_empty_tfidf_rows(segments: pd.DataFrame) -> pd.DataFrame:
    """Elimina segmentos que quedan sin features tras quitar stopwords."""
    vectorizer = make_vectorizer()
    try:
        matrix = vectorizer.fit_transform(segments["text_clean"])
    except ValueError:
        return segments.iloc[0:0].copy()

    mask = matrix.getnnz(axis=1) > 0
    filtered = segments.loc[mask].reset_index(drop=True)
    return filtered


@dataclass
class LexicalResult:
    segments: pd.DataFrame
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    coefficients: np.ndarray
    feature_names: np.ndarray
    top_features: pd.DataFrame
    coef_ttest: dict[str, float]
    evaluation: dict[str, Any]


def fit_and_interpret_tfidf(
    segments: pd.DataFrame,
    seed: int = SEED,
    n_bootstrap: int = 200,
    top_k: int = 15,
) -> LexicalResult:
    """Entrena TF-IDF + LogReg, evalúa con split agrupado y extrae coeficientes.

    El modelo final para interpretación se entrena con todos los segmentos.
    La evaluación supervisada usa un GroupShuffleSplit por ``doc_id`` para que
    ningún documento aporte segmentos a train y test a la vez.
    """
    data = filter_empty_tfidf_rows(segments)
    if data.empty:
        raise RuntimeError("No quedaron segmentos con features TF-IDF.")

    y = (data["corpus"] == "A").astype(int).to_numpy()
    groups = data["doc_id"].to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Se necesitan las dos clases (A y B) para entrenar.")

    # --- Evaluación con split agrupado -------------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, y, groups=groups))

    train_df = data.iloc[train_idx]
    test_df = data.iloc[test_idx]

    eval_vectorizer = make_vectorizer()
    x_train = eval_vectorizer.fit_transform(train_df["text_clean"])
    x_test = eval_vectorizer.transform(test_df["text_clean"])

    eval_classifier = make_classifier(seed)
    eval_classifier.fit(x_train, (train_df["corpus"] == "A").astype(int))
    y_pred = eval_classifier.predict(x_test)
    y_test = (test_df["corpus"] == "A").astype(int).to_numpy()

    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=[0, 1], zero_division=0)

    evaluation = {
        "n_train_segments": int(len(train_df)),
        "n_test_segments": int(len(test_df)),
        "n_train_documents": int(train_df["doc_id"].nunique()),
        "n_test_documents": int(test_df["doc_id"].nunique()),
        "macro_f1_test": float(macro_f1),
        "vocabulary_size_eval": int(len(eval_vectorizer.get_feature_names_out())),
    }

    # --- Modelo final para interpretación ----------------------------------
    vectorizer = make_vectorizer()
    x_all = vectorizer.fit_transform(data["text_clean"])
    classifier = make_classifier(seed)
    classifier.fit(x_all, y)

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

    # --- Test t de una muestra sobre TODOS los pesos -----------------------
    # NOTA: este test describe si la distribución de coeficientes está
    # centrada fuera de cero. Es el test solicitado en el enunciado, pero no
    # debe interpretarse como un p-valor por feature: los coeficientes no son
    # observaciones i.i.d. La evidencia por feature se reporta con el IC
    # bootstrap de la sección siguiente.
    t_stat, p_value = stats.ttest_1samp(coefficients, popmean=0.0, nan_policy="omit")
    coef_ttest = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_coefficient": float(np.mean(coefficients)),
        "std_coefficient": float(np.std(coefficients, ddof=1)),
        "n_features": int(len(coefficients)),
    }

    # --- Bootstrap agrupado por documento para los top features ------------
    top = bootstrap_top_coefficients(
        data=data,
        top_features=top["feature"].tolist(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    ).merge(top, on="feature", how="left")
    top = top.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    top["direction"] = np.where(
        top["coefficient"] > 0, "A (alucinación)", "B (veraz)"
    )
    top["significant_95ci"] = (top["ci_low"] > 0) | (top["ci_high"] < 0)

    return LexicalResult(
        segments=data,
        vectorizer=vectorizer,
        classifier=classifier,
        coefficients=coefficients,
        feature_names=feature_names,
        top_features=top,
        coef_ttest=coef_ttest,
        evaluation=evaluation,
    )


def bootstrap_top_coefficients(
    data: pd.DataFrame,
    top_features: Sequence[str],
    n_bootstrap: int = 200,
    seed: int = SEED,
) -> pd.DataFrame:
    """IC bootstrap (percentil) para los coeficientes de los top features.

    Se remuestrean DOCUMENTOS completos con reemplazo (no segmentos), de modo
    que se preserva la dependencia entre segmentos del mismo documento.
    """
    features = list(dict.fromkeys(top_features))
    if n_bootstrap <= 0:
        return pd.DataFrame(
            {
                "feature": features,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "bootstrap_mean": np.nan,
                "n_bootstrap_ok": 0,
            }
        )

    rng = np.random.default_rng(seed)
    doc_ids = data["doc_id"].unique()
    groups = {
        doc_id: frame for doc_id, frame in data.groupby("doc_id", sort=False)
    }

    samples: dict[str, list[float]] = {feature: [] for feature in features}

    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(doc_ids, size=len(doc_ids), replace=True)
        boot = pd.concat(
            [groups[doc_id] for doc_id in sampled_ids], ignore_index=True
        )
        y_boot = (boot["corpus"] == "A").astype(int).to_numpy()
        if len(np.unique(y_boot)) < 2:
            continue

        vectorizer = make_vectorizer()
        try:
            x_boot = vectorizer.fit_transform(boot["text_clean"])
        except ValueError:
            continue

        classifier = make_classifier(seed)
        classifier.fit(x_boot, y_boot)
        coef_map = dict(zip(vectorizer.get_feature_names_out(), classifier.coef_[0]))
        for feature in features:
            samples[feature].append(float(coef_map.get(feature, 0.0)))

    records = []
    for feature in features:
        values = np.asarray(samples[feature], dtype=float)
        if values.size == 0:
            records.append(
                {
                    "feature": feature,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_mean": np.nan,
                    "n_bootstrap_ok": 0,
                }
            )
        else:
            records.append(
                {
                    "feature": feature,
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_mean": float(np.mean(values)),
                    "n_bootstrap_ok": int(values.size),
                }
            )

    return pd.DataFrame(records)


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
    """T-test apareado + Wilcoxon + tamaño del efecto sobre NLL A vs B."""
    a = table["mean_nll_a"].to_numpy(dtype=float)
    b = table["mean_nll_b"].to_numpy(dtype=float)
    diff = a - b

    t_stat, t_pvalue = stats.ttest_rel(a, b, nan_policy="omit")
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    cohens_d = mean_diff / std_diff if std_diff and not math.isnan(std_diff) else float("nan")

    # IC 95 % de la diferencia media (aproximación normal).
    n = len(diff)
    if n > 1 and std_diff and not math.isnan(std_diff):
        standard_error = std_diff / math.sqrt(n)
        critical = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean_diff - critical * standard_error
        ci_high = mean_diff + critical * standard_error
    else:
        ci_low = ci_high = float("nan")

    wilcoxon_pvalue = float("nan")
    if n >= 10 and np.any(diff != 0):
        try:
            wilcoxon_pvalue = float(
                stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue
            )
        except ValueError:
            wilcoxon_pvalue = float("nan")

    weights_a = table["n_tokens_a"].to_numpy(dtype=float)
    weights_b = table["n_tokens_b"].to_numpy(dtype=float)

    return {
        "n_documents_paired": int(n),
        "n_tokens_a_total": int(weights_a.sum()),
        "n_tokens_b_total": int(weights_b.sum()),
        "mean_nll_a": float(np.mean(a)),
        "mean_nll_b": float(np.mean(b)),
        "mean_ppl_a": float(np.exp(np.mean(a))),
        "mean_ppl_b": float(np.exp(np.mean(b))),
        "pooled_ppl_a": float(np.exp(np.average(a, weights=weights_a))),
        "pooled_ppl_b": float(np.exp(np.average(b, weights=weights_b))),
        "median_ppl_a": float(np.median(np.exp(a))),
        "median_ppl_b": float(np.median(np.exp(b))),
        "mean_delta_nll": mean_diff,
        "delta_nll_ci95_low": float(ci_low),
        "delta_nll_ci95_high": float(ci_high),
        "paired_ttest": {
            "t_statistic": float(t_stat),
            "p_value": float(t_pvalue),
            "degrees_of_freedom": int(n - 1),
        },
        "cohens_d_paired": float(cohens_d),
        "wilcoxon_p_value": wilcoxon_pvalue,
        "fraction_docs_a_more_surprising": float(np.mean(diff > 0)),
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


def build_top_features_chart(top: pd.DataFrame, coef_ttest: dict[str, Any], n_bootstrap: int) -> alt.Chart:
    """Barras horizontales de coeficientes con IC 95 % bootstrap."""
    order = top["feature"].tolist()
    color_scale = alt.Scale(
        domain=["A (alucinación)", "B (veraz)"],
        range=[COLOR_HALLUCINATION, COLOR_TRUTHFUL],
    )

    base = alt.Chart(top).encode(
        y=alt.Y("feature:N", sort=order, title=None),
    )
    bars = base.mark_bar(size=18).encode(
        x=alt.X("coefficient:Q", title="Coeficiente LogisticRegression (log-odds)"),
        color=alt.Color("direction:N", scale=color_scale, title=None),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("coefficient:Q", format=".4f", title="Coef."),
            alt.Tooltip("ci_low:Q", format=".4f", title="IC 95% inf."),
            alt.Tooltip("ci_high:Q", format=".4f", title="IC 95% sup."),
            alt.Tooltip("significant_95ci:N", title="IC excluye 0"),
        ],
    )
    error_bars = base.mark_rule(strokeWidth=2, color="black").encode(
        x=alt.X("ci_low:Q"),
        x2="ci_high:Q",
    )
    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color="gray", strokeDash=[4, 4]
    ).encode(x="x:Q")

    subtitle = (
        f"Top 15 por |coeficiente| · t-test de pesos: t={coef_ttest['t_statistic']:.2f}, "
        f"p={coef_ttest['p_value']:.2e} · IC 95 % bootstrap agrupado ({n_bootstrap} remuestreos)"
    )
    return (
        (bars + error_bars + zero_rule)
        .properties(
            title=alt.TitleParams(
                text="Exp. 1 — Términos asociados a alucinación vs. texto veraz",
                subtitle=subtitle,
            ),
            width=640,
            height=420,
        )
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=11)
    )


def build_perplexity_plots(table: pd.DataFrame, stats_result: dict[str, Any]) -> list[tuple[str, alt.Chart]]:
    """Genera boxplot, scatter apareado e histograma del delta de NLL."""
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
        y=alt.Y("mean_nll:Q", title="NLL medio por token (nats) — menor = más predecible"),
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
            alt.Tooltip("mean_nll:Q", format=".4f", title="NLL medio"),
        ],
    )
    boxplot_chart = (box + points).properties(
        title=alt.TitleParams(
            text="Exp. 1 — Perplejidad por token: alucinación vs. texto veraz",
            subtitle=(
                f"n={stats_result['n_documents_paired']} documentos · "
                f"PPL media A={stats_result['mean_ppl_a']:.1f} vs B={stats_result['mean_ppl_b']:.1f} · "
                f"t apareado p={stats_result['paired_ttest']['p_value']:.2e}"
            ),
        ),
        width=460,
        height=380,
    )

    min_value = float(min(table["mean_nll_a"].min(), table["mean_nll_b"].min()))
    max_value = float(max(table["mean_nll_a"].max(), table["mean_nll_b"].max()))
    identity = pd.DataFrame({"x": [min_value, max_value], "y": [min_value, max_value]})

    scatter = alt.Chart(table).mark_circle(size=38, opacity=0.65).encode(
        x=alt.X("mean_nll_b:Q", title="NLL medio · Corpus B (veraz)"),
        y=alt.Y("mean_nll_a:Q", title="NLL medio · Corpus A (alucinado)"),
        color=alt.condition(
            alt.datum.delta_nll > 0, alt.value(COLOR_HALLUCINATION), alt.value(COLOR_TRUTHFUL)
        ),
        tooltip=[
            alt.Tooltip("doc_id:N", title="Documento"),
            alt.Tooltip("mean_nll_a:Q", format=".4f", title="NLL A"),
            alt.Tooltip("mean_nll_b:Q", format=".4f", title="NLL B"),
            alt.Tooltip("delta_nll:Q", format=".4f", title="Δ NLL (A-B)"),
        ],
    )
    identity_line = alt.Chart(identity).mark_line(
        color="gray", strokeDash=[5, 5]
    ).encode(x="x:Q", y="y:Q")
    scatter_chart = (scatter + identity_line).properties(
        title=alt.TitleParams(
            text="Exp. 1 — NLL apareada por documento",
            subtitle="Puntos por encima de la diagonal: tokens alucinados menos predecibles que los veraces",
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
            title="Δ NLL = A − B (nats/token)",
        ),
        y=alt.Y("count():Q", title="Documentos"),
        tooltip=[alt.Tooltip("count():Q", title="Documentos")],
    )
    histogram_chart = (histogram + zero_rule).properties(
        title=alt.TitleParams(
            text="Exp. 1 — Distribución del Δ NLL por documento",
            subtitle=(
                f"Δ medio={stats_result['mean_delta_nll']:.4f} nats/token "
                f"(IC 95 %: {stats_result['delta_nll_ci95_low']:.4f} a "
                f"{stats_result['delta_nll_ci95_high']:.4f})"
            ),
        ),
        width=560,
        height=340,
    )

    return [
        ("exp1_perplexity_boxplot", boxplot_chart),
        ("exp1_perplexity_paired_scatter", scatter_chart),
        ("exp1_perplexity_delta_histogram", histogram_chart),
    ]


def export_results(
    df: pd.DataFrame,
    segments: pd.DataFrame,
    lexical: LexicalResult,
    perplexity_table: pd.DataFrame | None,
    perplexity_stats: dict[str, Any] | None,
    perplexity_metadata: dict[str, Any] | None,
    graphs_dir: Path,
    results_dir: Path,
    n_bootstrap: int,
    save_png: bool,
) -> dict[str, Any]:
    """Guarda CSV/JSON y gráficos; devuelve el resumen consolidado."""
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # --- CSVs --------------------------------------------------------------
    segments.to_csv(
        results_dir / "exp1_segments.csv", index=False, encoding="utf-8-sig"
    )
    lexical.top_features.to_csv(
        results_dir / "exp1_top_features.csv", index=False, encoding="utf-8-sig"
    )

    # --- Gráficos léxicos --------------------------------------------------
    top_chart = build_top_features_chart(
        lexical.top_features, lexical.coef_ttest, n_bootstrap
    )
    save_chart(top_chart, "exp1_top_features", graphs_dir, save_png=save_png)

    generated_charts = ["exp1_top_features.html"]

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
            "n_segments_used_tfidf": int(len(lexical.segments)),
            "n_documents_used_tfidf": int(lexical.segments["doc_id"].nunique()),
        },
        "tfidf": {
            **lexical.evaluation,
            "n_features_full_fit": int(len(lexical.coefficients)),
            "coefficient_ttest": lexical.coef_ttest,
            "top_features": lexical.top_features[
                [
                    "feature",
                    "coefficient",
                    "ci_low",
                    "ci_high",
                    "direction",
                    "significant_95ci",
                ]
            ].to_dict(orient="records"),
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

def print_lexical_results(lexical: LexicalResult, n_bootstrap: int) -> None:
    print_section("[2/3] ANÁLISIS LÉXICO — TF-IDF + LOGISTIC REGRESSION")

    evaluation = lexical.evaluation
    print(
        f"  Split agrupado por documento: "
        f"{evaluation['n_train_documents']} docs train / "
        f"{evaluation['n_test_documents']} docs test"
    )
    print(
        f"  Segmentos: {evaluation['n_train_segments']} train / "
        f"{evaluation['n_test_segments']} test"
    )
    print(f"  Vocabulario (fit de evaluación): {evaluation['vocabulary_size_eval']:,} features")
    print(f"  F1 macro en test: {evaluation['macro_f1_test']:.4f}")
    print(
        "  Test t de una muestra sobre TODOS los coeficientes "
        f"(H0: media=0): t={lexical.coef_ttest['t_statistic']:.3f}, "
        f"p={lexical.coef_ttest['p_value']:.3e}, "
        f"n_features={lexical.coef_ttest['n_features']:,}"
    )
    print(
        "  Nota: el t-test describe la distribución global de pesos; "
        "la evidencia por feature son los IC bootstrap."
    )

    display = lexical.top_features.copy()
    display["coef"] = display["coefficient"].map(lambda value: f"{value:+.4f}")
    display["IC 95%"] = display.apply(
        lambda row: f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]", axis=1
    )
    display["sig."] = np.where(display["significant_95ci"], "sí", "no")
    print(f"\n  Top 15 features por |coeficiente| "
          f"(bootstrap={n_bootstrap} remuestreos):")
    print(
        display[["feature", "coef", "IC 95%", "direction", "sig."]]
        .rename(columns={"feature": "feature", "direction": "dirección"})
        .to_string(index=False)
    )


def print_perplexity_results(metadata: dict[str, Any], stats_result: dict[str, Any]) -> None:
    print("\n  Resultados del test apareado:")

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
        f"  Perplejidad media por documento (exp de la NLL media): "
        f"A={stats_result['mean_ppl_a']:.2f} | B={stats_result['mean_ppl_b']:.2f}"
    )
    print(
        f"  Perplejidad agregada por token: "
        f"A={stats_result['pooled_ppl_a']:.2f} | B={stats_result['pooled_ppl_b']:.2f}"
    )
    print(
        f"  Perplejidad mediana: "
        f"A={stats_result['median_ppl_a']:.2f} | B={stats_result['median_ppl_b']:.2f}"
    )
    print(
        f"  Δ NLL medio (A-B): {stats_result['mean_delta_nll']:+.4f} nats/token "
        f"(IC 95 %: {stats_result['delta_nll_ci95_low']:+.4f} a "
        f"{stats_result['delta_nll_ci95_high']:+.4f})"
    )
    print(
        f"  T-test apareado: t={stats_result['paired_ttest']['t_statistic']:.3f}, "
        f"gl={stats_result['paired_ttest']['degrees_of_freedom']}, "
        f"p={stats_result['paired_ttest']['p_value']:.3e}"
    )
    print(
        f"  Cohen's d (pareado): {stats_result['cohens_d_paired']:.3f} | "
        f"Wilcoxon p={stats_result['wilcoxon_p_value']:.3e}"
    )
    print(
        f"  Documentos con A más sorprendente que B: "
        f"{stats_result['fraction_docs_a_more_surprising']:.1%}"
    )


# ---------------------------------------------------------------------------
# 6. Main / CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Experimento 1: análisis léxico (TF-IDF + LogReg) y de perplejidad "
            "(LM causal) de segmentos alucinados vs. veraces en Mu-SHROOM."
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
        "--bootstrap",
        type=int,
        default=200,
        help="Remuestreos bootstrap para los coeficientes (0 = desactivar).",
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
    lexical = fit_and_interpret_tfidf(
        segments=segments,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        top_k=15,
    )
    print_lexical_results(lexical, args.bootstrap)

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
        lexical=lexical,
        perplexity_table=perplexity_table,
        perplexity_stats=perplexity_stats,
        perplexity_metadata=perplexity_metadata,
        graphs_dir=args.graphs_dir,
        results_dir=args.results_dir,
        n_bootstrap=args.bootstrap,
        save_png=args.save_png,
    )

    print_section("ARCHIVOS GENERADOS")
    print(f"  Gráficos: {summary['outputs']['graphs_dir']}")
    for chart in summary["outputs"]["charts"]:
        print(f"    - {chart}")
    print(f"  Resultados: {summary['outputs']['results_dir']}")
    for filename in (
        "exp1_segments.csv",
        "exp1_top_features.csv",
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
#   graphs/exp1/exp1_top_features.html
#   graphs/exp1/exp1_perplexity_boxplot.html
#   graphs/exp1/exp1_perplexity_paired_scatter.html
#   graphs/exp1/exp1_perplexity_delta_histogram.html
#   results/exp1/exp1_segments.csv
#   results/exp1/exp1_top_features.csv
#   results/exp1/exp1_perplexity_per_document.csv
#   results/exp1/exp1_summary.json
# -----------------------------------------------------------------------------
