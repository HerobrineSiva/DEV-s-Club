
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load dataset
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "train.csv")

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Recreate exactly the same development/holdout split
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

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric,
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]),
        categorical,
    ),
])

# Selected model
model = Pipeline([
    ("prep", preprocessor),
    ("model", HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )),
])

# Train only on development data
model.fit(X_dev, y_dev)

# Evaluate on holdout exactly once
probabilities = model.predict_proba(X_holdout)[:, 1]
threshold = 0.26
predictions = (probabilities >= threshold).astype(int)

print("HOLDOUT RESULTS")
print("Holdout observations:", len(X_holdout))
print("Actual failures:", int(y_holdout.sum()))
print("Threshold:", threshold)
print(f"PR-AUC / Average Precision: {average_precision_score(y_holdout, probabilities):.4f}")
print(f"F1-Score: {f1_score(y_holdout, predictions, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_holdout, predictions, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_holdout, predictions, zero_division=0):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_holdout, predictions))

print("\nKeep these results as the independent evaluation.")
print("Do not tune the model or threshold using holdout results.")
