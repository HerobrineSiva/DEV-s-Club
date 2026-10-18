"""Shared data, model, and inference utilities for machine-failure prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
TARGET = "Machine failure"
FEATURE_COLUMNS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
NUMERIC_COLUMNS = FEATURE_COLUMNS[1:]
CATEGORICAL_COLUMNS = ["Type"]
VALID_TYPES = {"L", "M", "H"}
RANDOM_SEED = 42


def load_training_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load labelled data and enforce the expected schema."""
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")
    if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET!r} must contain only 0 and 1.")
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET].astype(int).copy()


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return inference features in the fitted feature order."""
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURE_COLUMNS)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected feature columns: {sorted(extra)}")
    validated = frame[FEATURE_COLUMNS].copy()
    if validated.empty:
        raise ValueError("At least one observation is required.")
    if validated["Type"].isna().any() or not validated["Type"].isin(VALID_TYPES).all():
        raise ValueError(f"Type must be one of {sorted(VALID_TYPES)}.")
    for column in NUMERIC_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        if validated[column].isna().any():
            raise ValueError(f"{column!r} must contain finite numeric values.")
        if not validated[column].map(lambda value: float("-inf") < value < float("inf")).all():
            raise ValueError(f"{column!r} must contain finite numeric values.")
    if (validated["Rotational speed [rpm]"] <= 0).any() or (validated["Tool wear [min]"] < 0).any():
        raise ValueError("Rotational speed must be positive and tool wear must be non-negative.")
    return validated


def make_preprocessor(scale_numeric: bool, dense: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]), CATEGORICAL_COLUMNS),
    ])


def make_models() -> dict[str, Pipeline]:
    """Return fresh pipelines; all preprocessing is fold-local during CV."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(scale_numeric=True, dense=False)),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False, dense=False)),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=1)),
        ]),
        "Extra Trees": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False, dense=False)),
            ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False, dense=True)),
            ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_SEED)),
        ]),
    }
