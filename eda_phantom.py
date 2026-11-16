#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA auxiliar — Phantom Hallucination Detection
==============================================

Genera dos gráficos pensados para diapositivas, a partir del split ``train`` de
``seyled/Phantom_Hallucination_Detection``:

1. ``phantom_class_balance``      : balance 50/50 de ``ground_truth_label``.
2. ``phantom_length_distribution``: densidad de palabras de ``answer`` por clase,
   recortada en el percentil 98 y con las medianas anotadas.

Cada gráfico se exporta a ``graphs/eda_phantom/`` en HTML y PNG. El PNG requiere
``vl-convert-python`` (``pip install vl-convert-python``). Los colores y el
estilo siguen la paleta de ``graphs/exp2`` y ``graphs/exp3``:

* verde  ``#54A24B`` → Sin alucinación
* rojo   ``#E45756`` → Alucinación

Uso
---
    python eda_phantom.py
    python eda_phantom.py --output-dir graphs/eda_phantom
"""

from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import gaussian_kde

DATASET_NAME = "seyled/Phantom_Hallucination_Detection"
DEFAULT_OUTPUT_DIR = Path("graphs/eda_phantom")

# Paleta consistente con experiments/exp2 y experiments/exp3
COLOR_TRUTHFUL = "#54A24B"       # verde
COLOR_HALLUCINATION = "#E45756"  # rojo
LABEL_TRUTHFUL = "Sin alucinación"
LABEL_HALLUCINATION = "Alucinación"
LABEL_MAP = {
    "not hallucination": LABEL_TRUTHFUL,
    "hallucination": LABEL_HALLUCINATION,
}
LABEL_ORDER = [LABEL_TRUTHFUL, LABEL_HALLUCINATION]
COLOR_SCALE = alt.Scale(
    domain=LABEL_ORDER, range=[COLOR_TRUTHFUL, COLOR_HALLUCINATION]
)

PERCENTILE_CUT = 0.98

# Tipografías grandes para proyección en diapositivas
FONT_AXIS_LABEL = 18
FONT_AXIS_TITLE = 19
FONT_LEGEND = 17
FONT_TITLE = 28
FONT_SUBTITLE = 19


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def format_number(value: int | float) -> str:
    """Formatea un número con punto de miles (estilo español)."""
    return f"{value:,.0f}".replace(",", ".")


def format_percent(value: float) -> str:
    """Formatea una proporción como porcentaje con coma decimal."""
    return f"{value * 100:.1f} %".replace(".", ",")


def load_phantom() -> pd.DataFrame:
    """Carga Phantom y agrega etiqueta legible y cantidad de palabras."""
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    required = {"answer", "ground_truth_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {DATASET_NAME}: {sorted(missing)}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df["label"] = (
        df["ground_truth_label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    )
    unknown = df.loc[df["label"].isna(), "ground_truth_label"].unique()
    if len(unknown) > 0:
        raise ValueError(f"Etiquetas desconocidas en ground_truth_label: {unknown}")

    df["n_words"] = (
        df["answer"].fillna("").astype(str).str.split().str.len().astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def build_class_balance_chart(df: pd.DataFrame) -> alt.Chart:
    """Barras horizontales con conteo exacto y porcentaje por clase."""
    counts = (
        df.groupby("label", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total = int(counts["count"].sum())
    counts["share"] = counts["count"] / total
    counts["bar_text"] = counts.apply(
        lambda row: f"{format_number(row['count'])}  ·  {format_percent(row['share'])}",
        axis=1,
    )

    x_max = float(counts["count"].max()) * 1.34

    base = alt.Chart(counts).encode(
        y=alt.Y(
            "label:N",
            title=None,
            sort=LABEL_ORDER,
            axis=alt.Axis(labelAngle=0),
        ),
        x=alt.X(
            "count:Q",
            title=f"Cantidad de respuestas (total: {format_number(total)})",
            scale=alt.Scale(domain=[0, x_max]),
            axis=alt.Axis(
                labelAngle=0,
                values=[0, 5000, 10000, 15000],
                format="~s",
                tickCount=4,
            ),
        ),
        color=alt.Color("label:N", scale=COLOR_SCALE, legend=None),
    )

    bars = base.mark_bar(size=72, cornerRadiusTopRight=5, cornerRadiusBottomRight=5)

    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=10,
        fontSize=21,
        fontWeight="bold",
        color="#333333",
    ).encode(text=alt.Text("bar_text:N"))

    return (
        (bars + labels)
        .properties(
            title=alt.TitleParams(
                text="Phantom — Balance de clases (50% / 50%)",
                subtitle=(
                    "A diferencia de Mu-SHROOM, cuenta con 15.443 pares exactos "
                    "limpio/alucinado"
                ),
                anchor="start",
            ),
            width=780,
            height=240,
        )
        .configure_axis(
            labelFontSize=FONT_AXIS_LABEL,
            titleFontSize=FONT_AXIS_TITLE,
            titlePadding=12,
            gridOpacity=0.25,
        )
        .configure_title(
            fontSize=FONT_TITLE,
            subtitleFontSize=FONT_SUBTITLE,
            subtitlePadding=8,
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def build_length_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """Densidades superpuestas de palabras por clase, recortadas al p98."""
    cut = float(df["n_words"].quantile(PERCENTILE_CUT))
    plot_df = df[df["n_words"] <= cut].copy()

    medians = (
        df.groupby("label")["n_words"].median().reindex(LABEL_ORDER).reset_index()
    )

    # KDE explícita: así controlamos la escala y podemos anotar las medianas
    grid = np.linspace(0, cut, 400)
    curves = []
    peak = 0.0
    for label in LABEL_ORDER:
        values = plot_df.loc[plot_df["label"] == label, "n_words"].to_numpy()
        density = gaussian_kde(values)(grid)
        peak = max(peak, float(density.max()))
        curves.append(
            pd.DataFrame({"words": grid, "density": density, "label": label})
        )

    density_df = pd.concat(curves, ignore_index=True)
    density_df["density"] = density_df["density"] / peak  # pico global = 1

    y_annot = 1.10
    y_max = 1.24
    line_width = 3.5

    base = alt.Chart(density_df).encode(
        x=alt.X(
            "words:Q",
            title=(
                "Palabras en la respuesta "
                f"(recorte en el percentil 98 = {format_number(cut)})"
            ),
            scale=alt.Scale(domain=[0, cut]),
            axis=alt.Axis(
                labelAngle=0,
                values=list(range(0, int(cut) + 1, 20)),
                tickCount=9,
            ),
        ),
        y=alt.Y(
            "density:Q",
            title="Concentración de respuestas",
            scale=alt.Scale(domain=[0, y_max]),
            axis=alt.Axis(labels=False, ticks=False, gridOpacity=0.25),
        ),
        color=alt.Color("label:N", title=None, scale=COLOR_SCALE, legend=None),
    )

    lines = base.mark_line(strokeWidth=line_width, interpolate="monotone").encode(
        color=alt.Color(
            "label:N",
            title=None,
            scale=COLOR_SCALE,
            legend=alt.Legend(
                orient="top",
                direction="horizontal",
                labelFontSize=FONT_LEGEND,
                symbolSize=260,
                symbolStrokeWidth=line_width,
                title=None,
            ),
        )
    )

    median_rules = (
        alt.Chart(medians)
        .mark_rule(strokeDash=[7, 5], strokeWidth=2.5, opacity=0.9)
        .encode(
            x=alt.X("n_words:Q"),
            color=alt.Color("label:N", scale=COLOR_SCALE, legend=None),
        )
    )

    medians["median_text"] = medians.apply(
        lambda row: f"{row['label']}\n{int(row['n_words'])} palabras", axis=1
    )
    median_labels = (
        alt.Chart(medians)
        .mark_text(
            align="left",
            baseline="top",
            dx=7,
            fontSize=18,
            fontWeight="bold",
            lineBreak="\n",
        )
        .encode(
            x=alt.X("n_words:Q"),
            y=alt.Y(datum=y_annot, title=None),
            text=alt.Text("median_text:N"),
            color=alt.Color("label:N", scale=COLOR_SCALE, legend=None),
        )
    )

    return (
        (lines + median_rules + median_labels)
        .properties(
            title=alt.TitleParams(
                text="Phantom — Longitud de respuesta por clase",
                subtitle=(
                    "Las respuestas alucinadas casi triplican la mediana de palabras "
                    "(67 vs. 25)"
                ),
                anchor="start",
            ),
            width=820,
            height=440,
            padding={"left": 22, "right": 14, "top": 5, "bottom": 5},
        )
        .configure_axis(
            labelFontSize=FONT_AXIS_LABEL,
            titleFontSize=FONT_AXIS_TITLE,
            titlePadding=12,
        )
        .configure_title(
            fontSize=FONT_TITLE,
            subtitleFontSize=FONT_SUBTITLE,
            subtitlePadding=8,
            anchor="start",
        )
        .configure_legend(
            labelFontSize=FONT_LEGEND, titleFontSize=FONT_LEGEND, padding=4
        )
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------

def save_chart(chart: alt.Chart, stem: str, output_dir: Path) -> None:
    """Guarda HTML y PNG (2x) del gráfico."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{stem}.html"
    chart.save(html_path)
    print(f"  ✓ {html_path}")

    png_path = output_dir / f"{stem}.png"
    try:
        chart.save(png_path, scale_factor=2.0)
        print(f"  ✓ {png_path}")
    except Exception as exc:  # vl-convert-python puede no estar instalado
        print(
            f"  [aviso] No se pudo guardar {png_path} ({exc}). "
            "Instalá 'vl-convert-python' para exportar PNG."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gráficos EDA de Phantom para diapositivas."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    alt.data_transformers.disable_max_rows()

    print(f"Cargando {DATASET_NAME} (split=train)…")
    df = load_phantom()

    counts = df["label"].value_counts().reindex(LABEL_ORDER)
    medians = df.groupby("label")["n_words"].median().reindex(LABEL_ORDER)
    cut = df["n_words"].quantile(PERCENTILE_CUT)

    print(f"  Total: {format_number(len(df))} respuestas")
    for label in LABEL_ORDER:
        print(
            f"  {label}: {format_number(counts[label])} "
            f"({format_percent(counts[label] / len(df))}) · "
            f"mediana {int(medians[label])} palabras"
        )
    print(f"  Percentil 98 de palabras: {format_number(cut)}")

    print(f"\nExportando gráficos en {args.output_dir}/")
    save_chart(
        build_class_balance_chart(df), "phantom_class_balance", args.output_dir
    )
    save_chart(
        build_length_distribution_chart(df),
        "phantom_length_distribution",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
