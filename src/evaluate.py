from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: Any | None = None,
) -> dict[str, float]:
    """Calculate classification metrics for a binary classifier."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
    }

    if y_probability is not None:
        probabilities = np.asarray(y_probability)

        if probabilities.ndim != 1:
            raise ValueError("y_probability must be one-dimensional.")

        if len(probabilities) != len(y_true):
            raise ValueError(
                "y_probability and y_true must have the same length."
            )

        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

    return metrics


def validate_performance_thresholds(
    metrics: dict[str, float],
    minimum_f1: float,
    minimum_accuracy: float,
    minimum_roc_auc: float,
) -> None:
    """Raise an error when model performance is below a required threshold."""
    required_metrics = {"f1", "accuracy", "roc_auc"}
    missing_metrics = required_metrics.difference(metrics)

    if missing_metrics:
        raise ValueError(
            f"Missing required metrics: {sorted(missing_metrics)}"
        )

    failures = []

    if metrics["f1"] < minimum_f1:
        failures.append(
            f"F1 {metrics['f1']:.4f} is below {minimum_f1:.4f}"
        )

    if metrics["accuracy"] < minimum_accuracy:
        failures.append(
            "Accuracy "
            f"{metrics['accuracy']:.4f} is below {minimum_accuracy:.4f}"
        )

    if metrics["roc_auc"] < minimum_roc_auc:
        failures.append(
            "ROC-AUC "
            f"{metrics['roc_auc']:.4f} is below {minimum_roc_auc:.4f}"
        )

    if failures:
        raise RuntimeError("; ".join(failures))