"""Train, compare, persist, and report the project models."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import DATASET_FILENAME, download_dataset, load_dataset
from .evaluate import regression_metrics, save_eda_figures, save_evaluation_figures
from .preprocessing import build_model, split_features_target


def run_training(project_dir: Path) -> dict[str, dict[str, float]]:
    """Run both experiments on one shared, reproducible train/test split."""
    csv_path = download_dataset(project_dir / "data" / "raw")
    frame = load_dataset(csv_path)
    save_eda_figures(frame, project_dir / "reports" / "figures")

    indices = range(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_frame = frame.iloc[list(train_idx)]
    test_frame = frame.iloc[list(test_idx)]
    predictions: dict[str, object] = {}
    metrics: dict[str, dict[str, float]] = {}
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, include_grades in (
        ("with_previous_grades", True),
        ("without_previous_grades", False),
    ):
        x_train, y_train = split_features_target(train_frame, include_grades)
        x_test, y_test = split_features_target(test_frame, include_grades)
        model = build_model(include_grades)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[name] = predicted
        metrics[name] = regression_metrics(y_test, predicted)
        joblib.dump(
            {"model": model, "include_previous_grades": include_grades},
            models_dir / f"{name}.joblib",
        )

    baseline = Pipeline([("regressor", DummyRegressor(strategy="mean"))])
    _, y_train = split_features_target(train_frame, False)
    _, y_test = split_features_target(test_frame, False)
    baseline.fit(pd.DataFrame(index=y_train.index), y_train)
    baseline_pred = baseline.predict(pd.DataFrame(index=y_test.index))
    predictions["dummy_baseline"] = baseline_pred
    metrics["dummy_baseline"] = regression_metrics(y_test, baseline_pred)
    joblib.dump(
        {"model": baseline, "include_previous_grades": False},
        models_dir / "dummy_baseline.joblib",
    )

    save_evaluation_figures(
        y_test.reset_index(drop=True),
        {key: value for key, value in predictions.items()},
        project_dir / "reports" / "figures",
    )
    result = {
        "dataset": {
            "source": "UCI Student Performance dataset, student-mat.csv",
            "filename": DATASET_FILENAME,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "missing_values": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
        },
        "metrics": metrics,
        "split": {"test_size": 0.2, "random_state": 42},
    }
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return metrics
