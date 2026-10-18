
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "train.csv")

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

numeric = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
categorical = ["Type"]

X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

def make_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

# HistGradientBoosting requires dense numeric input.
def make_dense_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]), categorical),
    ])

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
        ("prep", make_dense_preprocessor()),
        ("model", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in models.items():
    scores = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        scoring={
            "ap": "average_precision",
            "f1": make_scorer(f1_score, zero_division=0),
        },
        n_jobs=-1,
    )

    print(f"\n--- {name} ---")
    print(f"Mean AP: {scores['test_ap'].mean():.4f}")
    print(f"AP standard deviation: {scores['test_ap'].std():.4f}")
    print(f"Mean F1 at threshold 0.50: {scores['test_f1'].mean():.4f}")

print("\nHoldout remains untouched:", len(X_holdout), "rows")
