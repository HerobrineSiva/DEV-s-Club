"""Load a serialized model and predict machine-failure probabilities from a CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from modeling import PROJECT_ROOT, validate_features


def predict(frame: pd.DataFrame, artifact_path: Path = PROJECT_ROOT / "models" / "machine_failure_model.joblib") -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    validated = validate_features(frame)
    if artifact["feature_columns"] != list(validated.columns):
        raise ValueError("Artifact feature schema does not match the expected schema.")
    probability = artifact["model"].predict_proba(validated)[:, 1]
    return pd.DataFrame({"failure_probability": probability, "predicted_failure": (probability >= artifact["threshold"]).astype(int)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--artifact", type=Path, default=PROJECT_ROOT / "models" / "machine_failure_model.joblib")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(pd.read_csv(args.input_csv), args.artifact)
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(result.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
