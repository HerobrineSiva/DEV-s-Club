
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

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

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

# Reserve a final holdout for a later, unbiased evaluation.
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    scores = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        scoring={
            "average_precision": "average_precision",
            "f1": "f1",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    print(f"\n--- {name} ---")
    print(
        "Mean average precision:",
        round(scores["test_average_precision"].mean(), 4),
    )
    print(
        "AP standard deviation:",
        round(scores["test_average_precision"].std(), 4),
    )
    print(
        "Mean F1 at default threshold:",
        round(scores["test_f1"].mean(), 4),
    )

print("\nHoldout rows reserved:", len(X_holdout))
print("Holdout failures reserved:", int(y_holdout.sum()))
print("Do not use holdout labels for model or threshold selection.")
