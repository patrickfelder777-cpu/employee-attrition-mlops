from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import (
    calculate_classification_metrics,
    validate_performance_thresholds,
)
from src.preprocessing import (
    build_preprocessor,
    identify_feature_types,
    introduce_missing_values,
    split_features_target,
)


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML dictionary.")

    return config


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the training dataset."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run 'dvc pull' first."
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError("Dataset cannot be empty.")

    return dataframe


def build_model(config: dict[str, Any]) -> Any:
    """Create a classification model from configuration values."""
    model_config = config["model"]
    model_type = model_config["type"]
    random_state = config["project"]["random_state"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 1000),
            class_weight=model_config.get("class_weight", "balanced"),
            random_state=random_state,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 200),
            max_depth=model_config.get("max_depth"),
            min_samples_split=model_config.get(
                "min_samples_split",
                2,
            ),
            class_weight=model_config.get(
                "class_weight",
                "balanced",
            ),
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=model_config.get("n_estimators", 100),
            learning_rate=model_config.get("learning_rate", 0.1),
            max_depth=model_config.get("max_depth", 3),
            random_state=random_state,
        )

    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=model_config.get("n_estimators", 200),
            max_depth=model_config.get("max_depth"),
            min_samples_split=model_config.get(
                "min_samples_split",
                2,
            ),
            class_weight=model_config.get(
                "class_weight",
                "balanced",
            ),
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def prepare_data(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare training and testing data."""
    data_config = config["data"]
    random_state = config["project"]["random_state"]

    target_column = data_config["target"]

    missing_value_columns = [
        "MonthlyIncome",
        "DistanceFromHome",
        "JobSatisfaction",
        "OverTime",
    ]

    available_columns = [
        column
        for column in missing_value_columns
        if column in dataframe.columns
    ]

    dataframe_with_missing_values = introduce_missing_values(
        df=dataframe,
        columns=available_columns,
        missing_rate=data_config["missing_value_rate"],
        random_state=random_state,
    )

    features, target = split_features_target(
        dataframe_with_missing_values,
        target_column,
    )

    target = target.map({"No": 0, "Yes": 1})

    if target.isna().any():
        raise ValueError(
            "Target must contain only 'Yes' and 'No' values."
        )

    return train_test_split(
        features,
        target,
        test_size=data_config["test_size"],
        random_state=random_state,
        stratify=target,
    )


def save_outputs(
    pipeline: Pipeline,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Save the trained model and metrics."""
    model_path = Path(config["artifacts"]["model_path"])
    metrics_path = Path(config["artifacts"]["metrics_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def train(config_path: str) -> dict[str, float]:
    """Train, evaluate, log, and save the model."""
    config = load_config(config_path)

    dataframe = load_dataset(config["data"]["raw_path"])

    X_train, X_test, y_train, y_test = prepare_data(
        dataframe,
        config,
    )

    numeric_features, categorical_features = identify_feature_types(
        X_train
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = build_model(config)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment(config["project"]["experiment_name"])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=predictions,
            y_probability=probabilities,
        )

        model_parameters = config["model"].copy()

        mlflow.log_params(model_parameters)
        mlflow.log_param(
            "random_state",
            config["project"]["random_state"],
        )
        mlflow.log_param(
            "test_size",
            config["data"]["test_size"],
        )
        mlflow.log_param(
            "missing_value_rate",
            config["data"]["missing_value_rate"],
        )
        mlflow.log_param(
            "data_version",
            config["data"]["data_version"],
        )

        mlflow.log_metrics(metrics)

        signature = infer_signature(
            X_train,
            pipeline.predict(X_train),
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="attrition_model",
            signature=signature,
            input_example=X_train.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        save_outputs(
            pipeline=pipeline,
            metrics=metrics,
            config=config,
        )

        print("\nTraining completed")
        print(f"MLflow Run ID: {run.info.run_id}")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        validate_performance_thresholds(
            metrics=metrics,
            minimum_f1=config["evaluation"]["minimum_f1"],
            minimum_accuracy=config["evaluation"][
                "minimum_accuracy"
            ],
            minimum_roc_auc=config["evaluation"][
                "minimum_roc_auc"
            ],
        )

    return metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the employee attrition model."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    train(arguments.config)