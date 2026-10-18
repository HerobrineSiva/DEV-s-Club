"""Reproducible development-CV selection followed by one fixed holdout evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from modeling import RANDOM_SEED, make_models, load_training_data

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "experiment.json"


def best_threshold(y_true, probabilities) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    candidates = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        candidates.append((f1_score(y_true, predicted, zero_division=0), float(threshold)))
    _, threshold = max(candidates, key=lambda item: (item[0], -item[1]))
    predicted = (probabilities >= threshold).astype(int)
    return threshold, {
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def metrics(y_true, probabilities, threshold: float) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def main() -> None:
    x, y = load_training_data()
    x_dev, x_holdout, y_dev, y_holdout = train_test_split(x, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    development: dict[str, dict[str, object]] = {}
    for name, model in make_models().items():
        probabilities = cross_val_predict(model, x_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold, threshold_metrics = best_threshold(y_dev, probabilities)
        development[name] = {
            "average_precision": float(average_precision_score(y_dev, probabilities)),
            "threshold": threshold,
            "threshold_metrics": threshold_metrics,
        }
        print(f"{name}: OOF AP={development[name]['average_precision']:.4f}, threshold={threshold:.2f}, OOF F1={threshold_metrics['f1']:.4f}")
    selected_name = max(development, key=lambda name: development[name]["average_precision"])
    selected_threshold = float(development[selected_name]["threshold"])
    selected_model = make_models()[selected_name]
    selected_model.fit(x_dev, y_dev)
    holdout = metrics(y_holdout, selected_model.predict_proba(x_holdout)[:, 1], selected_threshold)
    report = {
        "split": {"seed": RANDOM_SEED, "holdout_fraction": 0.20, "development_rows": len(x_dev), "holdout_rows": len(x_holdout), "holdout_failures": int(y_holdout.sum())},
        "failure_prevalence": float(y.mean()), "development_oof": development,
        "selected_model": selected_name, "selected_threshold": selected_threshold, "holdout": holdout,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSelected by OOF Average Precision: {selected_name}")
    print(f"Final threshold selected from OOF development predictions: {selected_threshold:.2f}")
    print("Holdout:", json.dumps(holdout))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
