"""Fit the selected deployment model on all labelled rows and serialize its contract."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from modeling import FEATURE_COLUMNS, RANDOM_SEED, make_models, load_training_data

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "results" / "experiment.json"
ARTIFACT = ROOT / "models" / "machine_failure_model.joblib"


def main() -> None:
    if not EXPERIMENT.exists():
        raise FileNotFoundError("Run src/run_experiment.py first to select a model and threshold.")
    experiment = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    name = experiment["selected_model"]
    threshold = float(experiment["selected_threshold"])
    x, y = load_training_data()
    model = make_models()[name]
    model.fit(x, y)
    artifact = {
        "model": model, "model_name": name, "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS, "positive_class": 1,
        "random_seed": RANDOM_SEED, "training_rows": len(x),
        "selection_experiment": "results/experiment.json",
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    print(f"Saved {name} deployment artifact to {ARTIFACT}")


if __name__ == "__main__":
    main()
