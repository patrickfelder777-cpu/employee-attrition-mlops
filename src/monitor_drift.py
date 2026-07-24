from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset



def load_config(config_path: str) -> dict[str, Any]:
    """Load the YAML project configuration."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML dictionary.")

    if "data" not in config or "monitoring" not in config:
        raise ValueError(
            "Configuration must contain data and monitoring sections."
        )

    return config


def load_reference_data(
    data_path: str,
    target_column: str,
) -> pd.DataFrame:
    """Load reference features from the training dataset."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run 'dvc pull' first."
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError("Reference dataset cannot be empty.")

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing."
        )

    return dataframe.drop(columns=[target_column]).copy()


def create_production_data(
    reference_data: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create simulated production data containing intentional drift."""
    if reference_data.empty:
        raise ValueError("Reference data cannot be empty.")

    rng = np.random.default_rng(random_state)

    production_data = reference_data.sample(
        frac=0.50,
        random_state=random_state,
    ).copy()

    if "MonthlyIncome" in production_data.columns:
        production_data["MonthlyIncome"] = (
            production_data["MonthlyIncome"] * 1.45
        ).round()

    if "DistanceFromHome" in production_data.columns:
        production_data["DistanceFromHome"] = (
            production_data["DistanceFromHome"] + 10
        )

    if "TotalWorkingYears" in production_data.columns:
        production_data["TotalWorkingYears"] = (
            production_data["TotalWorkingYears"] + 5
        )

    if "YearsAtCompany" in production_data.columns:
        production_data["YearsAtCompany"] = (
            production_data["YearsAtCompany"] * 0.55
        ).round()

    if "JobSatisfaction" in production_data.columns:
        production_data["JobSatisfaction"] = rng.choice(
            [1, 2, 3, 4],
            size=len(production_data),
            p=[0.55, 0.25, 0.15, 0.05],
        )

    if "EnvironmentSatisfaction" in production_data.columns:
        production_data["EnvironmentSatisfaction"] = rng.choice(
            [1, 2, 3, 4],
            size=len(production_data),
            p=[0.50, 0.30, 0.15, 0.05],
        )

    if "OverTime" in production_data.columns:
        production_data["OverTime"] = rng.choice(
            ["Yes", "No"],
            size=len(production_data),
            p=[0.80, 0.20],
        )

    if "BusinessTravel" in production_data.columns:
        production_data["BusinessTravel"] = rng.choice(
            [
                "Travel_Frequently",
                "Travel_Rarely",
                "Non-Travel",
            ],
            size=len(production_data),
            p=[0.70, 0.20, 0.10],
        )

    if "Department" in production_data.columns:
        production_data["Department"] = rng.choice(
            [
                "Sales",
                "Research & Development",
                "Human Resources",
            ],
            size=len(production_data),
            p=[0.65, 0.25, 0.10],
        )

    return production_data


def extract_drift_results(
    report_results: dict[str, Any],
) -> tuple[list[str], float]:
    """Extract drifted columns and drift share from Evidently output."""
    drifted_features: list[str] = []
    value_drift_count = 0

    metrics = report_results.get("metrics", [])

    for metric in metrics:
        metric_id = str(metric.get("id", ""))
        value = metric.get("value")
        config = metric.get("config", {})

        if "ValueDrift" not in metric_id:
            continue

        value_drift_count += 1

        column_name = None

        if isinstance(config, dict):
            column_name = (
                config.get("column")
                or config.get("column_name")
            )

        if column_name is None:
            continue

        drift_detected = False

        if isinstance(value, bool):
            drift_detected = value

        elif isinstance(value, dict):
            drift_detected = bool(
                value.get("drift_detected")
                or value.get("drifted")
                or value.get("detected")
            )

        if drift_detected:
            drifted_features.append(str(column_name))

    drifted_features = sorted(set(drifted_features))

    drift_share = (
        len(drifted_features) / value_drift_count
        if value_drift_count
        else 0.0
    )

    return drifted_features, drift_share

def monitor_drift(config_path: str) -> float:
    """Generate the Evidently report and enforce the drift threshold."""
    config = load_config(config_path)

    data_path = config["data"]["raw_path"]
    target_column = config["data"]["target"]

    threshold = float(
        config["monitoring"]["drift_share_threshold"]
    )

    report_path = Path(
        config["monitoring"]["report_path"]
    )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Drift threshold must be between 0 and 1."
        )

    reference_data = load_reference_data(
        data_path=data_path,
        target_column=target_column,
    )

    production_data = create_production_data(
        reference_data=reference_data,
        random_state=config["project"]["random_state"],
    )

    drift_metrics = [
            DataDriftPreset(),
    ]

    for column in reference_data.columns:
        drift_metrics.append(
            ValueDrift(column=column)
    )

        report = Report(drift_metrics)

        snapshot = report.run(
        current_data=production_data,
        reference_data=reference_data,
    )

    snapshot.save_html(str(report_path))

    results = snapshot.dict()

    import pprint
    pprint.pp(results)

    drifted_features, drift_share = extract_drift_results(
        results
    )

    print("\nData Drift Summary")
    print("-" * 50)
    print(f"Reference rows: {len(reference_data)}")
    print(f"Production rows: {len(production_data)}")
    print(f"Features evaluated: {len(reference_data.columns)}")
    print(f"Drifted features: {len(drifted_features)}")
    print(f"Overall drift share: {drift_share:.2%}")
    print(f"Configured threshold: {threshold:.2%}")

    if drifted_features:
        print("\nFeatures with detected drift:")

        for feature in drifted_features:
            print(f"- {feature}")
    else:
        print("\nNo feature drift was detected.")

    print(f"\nHTML report saved to: {report_path}")

    if drift_share > threshold:
        print(
            "\nSTATUS: FAILED — drift exceeded the threshold."
        )
    else:
        print(
            "\nSTATUS: PASSED — drift is within the threshold."
        )

    return drift_share


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run Evidently data drift monitoring."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the project configuration.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    try:
        drift_share = monitor_drift(arguments.config)
        config = load_config(arguments.config)

        threshold = float(
            config["monitoring"]["drift_share_threshold"]
        )

        if drift_share > threshold:
            sys.exit(1)

    except Exception as error:
        print(f"\nDrift monitoring error: {error}")
        sys.exit(1)