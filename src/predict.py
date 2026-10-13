"""Load persisted models and make validated predictions."""

import json
from pathlib import Path
from typing import Any

import joblib

from .preprocessing import validate_prediction_record


def predict(
    project_dir: Path, record: dict[str, Any], model_name: str = "without_previous_grades"
) -> float:
    """Return one predicted G3 using a persisted model."""
    model_path = project_dir / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python main.py train` first."
        )
    bundle = joblib.load(model_path)
    features = validate_prediction_record(record, bundle["include_previous_grades"])
    return float(bundle["model"].predict(features)[0])


def load_json_record(value: str) -> dict[str, Any]:
    """Parse a JSON object supplied by the command line."""
    record = json.loads(value)
    if not isinstance(record, dict):
        raise ValueError("Prediction input must be a JSON object")
    return record
