from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import calculate_classification_metrics
from src.preprocessing import (
    build_preprocessor,
    identify_feature_types,
    introduce_missing_values,
    split_features_target,
)


DATA_PATH = Path("data/raw/employee_attrition.csv")
CONFIG_PATH = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def model_test_data() -> tuple[
    Pipeline,
    pd.DataFrame,
    pd.Series,
]:
    """Train a small model and return it with its test data."""
    if not DATA_PATH.exists():
        pytest.fail(
            "Dataset is missing. Run 'dvc pull' before running tests."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataframe = pd.read_csv(DATA_PATH)

    dataframe = introduce_missing_values(
        df=dataframe,
        columns=[
            "MonthlyIncome",
            "DistanceFromHome",
            "JobSatisfaction",
            "OverTime",
        ],
        missing_rate=0.03,
        random_state=42,
    )

    features, target = split_features_target(
        dataframe,
        target_column="Attrition",
    )

    target = target.map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=config["project"]["random_state"],
        stratify=target,
    )

    numeric_features, categorical_features = identify_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


def test_model_prediction_shape_and_type(
    model_test_data: tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    """Predictions should have the expected type, length, and classes."""
    pipeline, X_test, _ = model_test_data

    predictions = pipeline.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_probability_shape_and_range(
    model_test_data: tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    """Predicted probabilities should have valid shape and values."""
    pipeline, X_test, _ = model_test_data

    probabilities = pipeline.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_meets_minimum_performance(
    model_test_data: tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    """The model should meet stable minimum evaluation thresholds."""
    pipeline, X_test, y_test = model_test_data

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        y_probability=probabilities,
    )

    assert metrics["accuracy"] >= 0.75
    assert metrics["f1"] >= 0.35
    assert metrics["roc_auc"] >= 0.65