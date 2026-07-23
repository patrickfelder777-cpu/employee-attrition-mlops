from __future__ import annotations

import mlflow
import pandas as pd


EXPERIMENT_NAME = "employee_attrition_experiments"
PRIMARY_METRIC = "f1"


def compare_experiments() -> pd.DataFrame:
    """Find and print the best MLflow run based on F1 score."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
    )

    if runs.empty:
        raise RuntimeError(
            f"No runs found for experiment '{EXPERIMENT_NAME}'."
        )

    columns = [
        "run_id",
        "params.type",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
    ]

    available_columns = [
        column for column in columns if column in runs.columns
    ]

    results = runs[available_columns].copy()

    best_run = results.iloc[0]

    print("\nBest MLflow run")
    print("----------------")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Model: {best_run.get('params.type', 'unknown')}")
    print(f"F1: {best_run.get('metrics.f1', float('nan')):.4f}")
    print(
        "Accuracy: "
        f"{best_run.get('metrics.accuracy', float('nan')):.4f}"
    )
    print(
        "ROC-AUC: "
        f"{best_run.get('metrics.roc_auc', float('nan')):.4f}"
    )

    print("\nAll runs ranked by F1")
    print(results.to_string(index=False))

    return results


if __name__ == "__main__":
    compare_experiments()