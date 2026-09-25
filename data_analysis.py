import numpy as np
import torch
import pandas as pd

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