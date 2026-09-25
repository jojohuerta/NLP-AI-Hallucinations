import os
import pandas as pd
import numpy as np
import altair as alt
import torch

from datasets import load_dataset

# Load a specific language (we will use english)
dataset = load_dataset("Helsinki-NLP/mu-shroom", "en")

# Load all languages combined
full_dataset = load_dataset("Helsinki-NLP/mu-shroom", "all")

print("---------------------------------------")

print(dataset)

print("---------------------------------------")

print(full_dataset)

print("---------------------------------------")


# Preprocessing
cols = ["lang", "model_output_text", "hard_labels"]

df_train = full_dataset["train_unlabeled"].to_pandas()[cols]
df_val = full_dataset["validation"].to_pandas()[cols]
df_test = full_dataset["test"].to_pandas()[cols]

# Concatenation of all splits into a single df
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

df_all["output_words"] = df_all["model_output_text"].str.split().str.len()

# We add a type check because unlabeled data might contain None instead of empty lists
df_all["has_hallucination"] = df_all["hard_labels"].apply(
    lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False
)

# Altair serialization
df_vis = df_all[["lang", "output_words", "has_hallucination"]].copy()


# Chart A: Total Dataset Balance by Language (Across all splits)
chart_lang = alt.Chart(df_vis).mark_bar(color='#4C78A8').encode(
    x=alt.X("count():Q", title="Total Number of Entries (All Splits)"),
    y=alt.Y("lang:N", sort="-x", title="Language Code"),
    tooltip=["lang", "count()"]
).properties(
    title="Total Dataset Balance by Language",
    width=500,
    height=300
)

# Chart B: Hallucination correlation with response length (All splits)
chart_len = alt.Chart(df_vis[df_vis["has_hallucination"] == True]).mark_bar(opacity=0.8, color='#E45756').encode(
    x=alt.X("output_words:Q", bin=alt.Bin(maxbins=40), title="Response Length (Words)"),
    y=alt.Y("count():Q", title="Frequency"),
    tooltip=["count()"]
).properties(
    title="Does hallucination correlate with response length?",
    width=500,
    height=300
)

# Chart C: Global proportion (Donut chart - All splits)
chart_class = alt.Chart(df_vis).mark_arc(innerRadius=60).encode(
    theta=alt.Theta("count():Q"),
    color=alt.Color("has_hallucination:N", title="Hallucination"),
    tooltip=["has_hallucination", "count()"]
).properties(
    title="Global Proportion of Texts with Hallucinations",
    width=300,
    height=300
)


# Exporting
os.makedirs("graphs", exist_ok=True)

chart_lang.save("graphs/chart_lang.html")
chart_len.save("graphs/chart_len.html")
chart_class.save("graphs/chart_class.html")

print("Charts successfully saved in the /graphs folder!")