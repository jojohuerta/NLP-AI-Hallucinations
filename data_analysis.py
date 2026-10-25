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

# Concatenation for the language chart (All splits)
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_vis_all = df_all[["lang"]].copy()

# Concatenation for labeled data
df_labeled = pd.concat([df_val, df_test], ignore_index=True)

df_labeled["output_words"] = df_labeled["model_output_text"].str.split().str.len()
df_labeled["has_hallucination"] = df_labeled["hard_labels"].apply(
    lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False
)

# Altair serialization for labeled data
df_vis_labeled = df_labeled[["lang", "output_words", "has_hallucination"]].copy()

# Chart: Total Dataset Balance by Language (Across all splits)
chart_lang = alt.Chart(df_vis_all).mark_bar(color='#4C78A8').encode(
    x=alt.X("count():Q", title="Total Number of Entries (All Splits)"),
    y=alt.Y("lang:N", sort="-x", title="Language Code"),
    tooltip=["lang", "count()"]
).properties(
    title="Total Dataset Balance by Language",
    width=500,
    height=300
)

# Chart: Global proportion (Donut chart - All splits)
chart_class = alt.Chart(df_vis_labeled).mark_arc(innerRadius=60).encode(
    theta=alt.Theta("count():Q"),
    color=alt.Color("has_hallucination:N", 
                    title="Has Hallucination",
                    scale=alt.Scale(domain=[False, True], range=['#54A24B', '#E45756'])),
    tooltip=["has_hallucination", "count()"]
).properties(
    title="Global Proportion of Texts with Hallucinations (Labeled Data)",
    width=300,
    height=300
)
# Chart: Percentage Line Chart with Linear Scale
chart_combined = alt.Chart(df_vis_labeled).mark_line(point=True, color='#E45756', strokeWidth=3).encode(
    x=alt.X("output_words:Q", 
            bin=alt.Bin(maxbins=30), 
            title="Response Length (Words)"),
    y=alt.Y("mean(has_hallucination):Q", 
            title="Percentage of Hallucinations", 
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format='%')),
    tooltip=[alt.Tooltip("mean(has_hallucination):Q", format='.2%', title="% Hallucinations"), 
             alt.Tooltip("count():Q", title="Total Texts")]
).properties(
    title="Percentage of Hallucinations vs. Response Length",
    width=600,
    height=300
)

# Chart: Distribution of Response Length (Labeled Data - Linear Scale)
chart_histogram = alt.Chart(df_vis_labeled).mark_bar(color='#54A24B', opacity=0.8).encode(
    x=alt.X("output_words:Q", 
            bin=alt.Bin(maxbins=30), 
            title="Response Length (Words)"),
    y=alt.Y("count():Q", 
            title="Frequency (Number of Texts)"),
    tooltip=[alt.Tooltip("count():Q", title="Frequency")]
).properties(
    title="Distribution of Response Length (Labeled Data)",
    width=600,
    height=300
)

# Exporting
os.makedirs("graphs", exist_ok=True)

chart_lang.save("graphs/chart_lang.html")
chart_class.save("graphs/chart_class.html")
chart_combined.save("graphs/chart_combined.html")
chart_histogram.save("graphs/chart_histogram.html")

print("Charts successfully saved in the /graphs folder!")
