
from pathlib import Path

import pandas as pd

# Locate train.csv relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Place train.csv inside the data folder."
    )

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("\n--- DATASET SIZE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

target = "Machine failure"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found.")

print("\n--- TARGET DISTRIBUTION ---")
print(df[target].value_counts(dropna=False))

print("\n--- TARGET PROPORTIONS (%) ---")
print(
    (df[target].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)

print("\n--- NUMERIC FEATURE SUMMARY ---")
print(df.describe().round(2))
