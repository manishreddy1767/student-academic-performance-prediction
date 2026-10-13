"""Tests for the data, preprocessing, training, and inference contracts."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.predict import predict
from src.preprocessing import build_model, feature_columns, split_features_target, validate_prediction_record


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET = PROJECT_DIR / "data" / "raw" / "student-mat.csv"


@pytest.fixture(scope="module")
def frame():
    return load_dataset(DATASET)


def test_dataset_schema_and_target(frame):
    assert len(frame) > 300
    assert "G3" in frame.columns
    assert frame["G3"].between(0, 20).all()


def test_feature_target_separation(frame):
    features, target = split_features_target(frame, include_previous_grades=False)
    assert "G3" not in features
    assert len(features) == len(target)
    assert set(feature_columns(False)) == set(features.columns)


def test_preprocessing_and_training(frame):
    features, target = split_features_target(frame, include_previous_grades=True)
    model = build_model(True)
    model.fit(features.iloc[:300], target.iloc[:300])
    output = model.predict(features.iloc[300:305])
    assert output.shape == (5,)
    assert output.dtype.kind in "fc"


def test_invalid_prediction_record_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        validate_prediction_record({}, False)


def test_persisted_prediction_has_numeric_output():
    model_path = PROJECT_DIR / "models" / "without_previous_grades.joblib"
    if not model_path.exists():
        pytest.skip("Run the training command before this integration assertion.")
    frame = load_dataset(DATASET)
    record = frame.drop(columns=["G3", "G1", "G2"]).iloc[0].to_dict()
    value = predict(PROJECT_DIR, record, "without_previous_grades")
    assert isinstance(value, float)
    assert value == value
