
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Load the data
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "Machine failure"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Separate categorical and numeric features
categorical_features = ["Type"]
numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# --------------------------------------------------
# 2. Split the data without losing failure examples
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training failures:", int(y_train.sum()))
print("Validation failures:", int(y_val.sum()))

# --------------------------------------------------
# 3. Prepare features
# --------------------------------------------------

numeric_preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessing, numeric_features),
    ("categorical", categorical_preprocessing, categorical_features),
])

# --------------------------------------------------
# 4. Define our first two models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}

# --------------------------------------------------
# 5. Train, evaluate and tune the threshold
# --------------------------------------------------

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    # Probability that each validation row is a failure
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    # PR-AUC-style metric: average precision
    ap = average_precision_score(y_val, probabilities)

    # Standard classification threshold
    default_predictions = (probabilities >= 0.5).astype(int)

    default_f1 = f1_score(
        y_val, default_predictions, zero_division=0
    )

    # Find the threshold with the best validation F1
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in [i / 100 for i in range(5, 96)]:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_val, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    tuned_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_val, tuned_predictions, zero_division=0
    )
    recall = recall_score(
        y_val, tuned_predictions, zero_division=0
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print(f"Average precision (PR metric): {ap:.4f}")
    print(f"F1 at threshold 0.50:          {default_f1:.4f}")
    print(f"Best validation threshold:     {best_threshold:.2f}")
    print(f"Best validation F1:            {best_f1:.4f}")
    print(f"Precision at tuned threshold:  {precision:.4f}")
    print(f"Recall at tuned threshold:     {recall:.4f}")

    print("\nConfusion matrix at tuned threshold:")
    print(confusion_matrix(y_val, tuned_predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            tuned_predictions,
            labels=[0, 1],
            target_names=["Healthy", "Failure"],
            zero_division=0,
        )
    )
