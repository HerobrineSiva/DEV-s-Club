from pathlib import Path

import joblib
import pandas as pd
import pytest

from modeling import FEATURE_COLUMNS, RANDOM_SEED, make_models, load_training_data, validate_features
from predict import predict


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    x, y = load_training_data()
    model = make_models()["Logistic Regression"]
    model.fit(x.iloc[:400], y.iloc[:400])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "model_name": "Logistic Regression", "threshold": 0.5, "feature_columns": FEATURE_COLUMNS, "positive_class": 1, "random_seed": RANDOM_SEED}, path)
    return path


def test_feature_order_and_output_validity(artifact: Path):
    x, _ = load_training_data()
    result = predict(x.iloc[:3][list(reversed(FEATURE_COLUMNS))], artifact)
    assert result.shape == (3, 2)
    assert result["failure_probability"].between(0, 1).all()
    assert set(result["predicted_failure"]).issubset({0, 1})


def test_rejects_bad_schema_and_machine_type():
    x, _ = load_training_data()
    with pytest.raises(ValueError, match="Missing"):
        validate_features(x.drop(columns=[FEATURE_COLUMNS[0]]))
    invalid = x.iloc[:1].copy()
    invalid["Type"] = "unknown"
    with pytest.raises(ValueError, match="Type"):
        validate_features(invalid)
