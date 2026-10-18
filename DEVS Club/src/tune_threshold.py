
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Load the dataset
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "train.csv")

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]


# 2. Keep the holdout set separate
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

numeric = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
categorical = ["Type"]


# 3. Build preprocessing pipelines
def make_preprocessor(dense=False):
    return ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric,
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=not dense,
                    ),
                ),
            ]),
            categorical,
        ),
    ])


# 4. Define candidate models
models = {
    "Random Forest": Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )),
    ]),
    "HistGradientBoosting": Pipeline([
        ("prep", make_preprocessor(dense=True)),
        ("model", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ]),
}


# 5. Generate out-of-fold probabilities
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = {}

for name, pipeline in models.items():
    oof_probabilities = np.zeros(len(X_dev))

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X_dev, y_dev), start=1
    ):
        X_fold_train = X_dev.iloc[train_idx]
        y_fold_train = y_dev.iloc[train_idx]
        X_fold_valid = X_dev.iloc[valid_idx]

        pipeline.fit(X_fold_train, y_fold_train)

        oof_probabilities[valid_idx] = (
            pipeline.predict_proba(X_fold_valid)[:, 1]
        )

        print(f"{name}: completed fold {fold}/5")

    # Every development row now has a prediction from
    # a model that did not train on that row.
    ap = average_precision_score(y_dev, oof_probabilities)

    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = []

    for threshold in thresholds:
        predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        scores.append(
            f1_score(y_dev, predictions, zero_division=0)
        )

    best_index = int(np.argmax(scores))
    best_threshold = float(thresholds[best_index])

    best_predictions = (
        oof_probabilities >= best_threshold
    ).astype(int)

    results[name] = {
        "AP": ap,
        "threshold": best_threshold,
        "F1": f1_score(
            y_dev, best_predictions, zero_division=0
        ),
        "precision": precision_score(
            y_dev, best_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_dev, best_predictions, zero_division=0
        ),
    }

    print(f"\n--- {name} ---")
    print(f"Out-of-fold Average Precision: {ap:.4f}")
    print(f"Best F1 threshold: {best_threshold:.2f}")
    print(f"F1 at selected threshold: {results[name]['F1']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")


print("\n--- Comparison ---")
for name, result in results.items():
    print(
        f"{name}: AP={result['AP']:.4f}, "
        f"F1={result['F1']:.4f}, "
        f"threshold={result['threshold']:.2f}"
    )

print(
    "\nHoldout remains untouched:",
    len(X_holdout),
    "rows; do not use its labels for threshold tuning."
)
