"""Feature definitions and leakage-safe preprocessing pipelines."""

from numbers import Real
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "G3"
PREVIOUS_GRADES = ["G1", "G2"]
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]


def feature_columns(include_previous_grades: bool) -> list[str]:
    """Return the feature schema for one of the two experiments."""
    return (PREVIOUS_GRADES if include_previous_grades else []) + (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )


def split_features_target(
    frame: pd.DataFrame, include_previous_grades: bool
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the selected predictors from the final grade target."""
    columns = feature_columns(include_previous_grades)
    missing = set(columns + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[columns].copy(), frame[TARGET].copy()


def build_model(include_previous_grades: bool) -> Pipeline:
    """Build an imputation, encoding, and linear regression pipeline."""
    numeric = NUMERIC_FEATURES + (PREVIOUS_GRADES if include_previous_grades else [])
    categorical = CATEGORICAL_FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )


def validate_prediction_record(
    record: dict[str, Any], include_previous_grades: bool
) -> pd.DataFrame:
    """Validate and normalize one prediction record to the training schema."""
    columns = feature_columns(include_previous_grades)
    missing = [column for column in columns if column not in record]
    if missing:
        raise ValueError(f"Prediction input is missing: {', '.join(missing)}")
    unexpected = sorted(set(record) - set(columns))
    if unexpected:
        raise ValueError(f"Prediction input contains unsupported fields: {unexpected}")
    row = pd.DataFrame([{column: record[column] for column in columns}])
    numeric = NUMERIC_FEATURES + (PREVIOUS_GRADES if include_previous_grades else [])
    for column in numeric:
        value = row.at[0, column]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"{column} must be numeric")
        if column in {"age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                      "famrel", "freetime", "goout", "Dalc", "Walc", "health"} and value < 0:
            raise ValueError(f"{column} cannot be negative")
        if column in PREVIOUS_GRADES and not 0 <= value <= 20:
            raise ValueError(f"{column} must be between 0 and 20")
    if row.at[0, "absences"] < 0:
        raise ValueError("absences cannot be negative")
    return row
