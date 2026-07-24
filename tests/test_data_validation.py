from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path("data/raw/employee_attrition.csv")


@pytest.fixture(scope="module")
def attrition_data() -> pd.DataFrame:
    """Load the actual employee attrition dataset."""
    if not DATA_PATH.exists():
        pytest.fail(
            "Dataset is missing. Run 'dvc pull' before running tests."
        )

    return pd.read_csv(DATA_PATH)


def test_dataset_has_minimum_number_of_rows(
    attrition_data: pd.DataFrame,
) -> None:
    """The project dataset must contain at least 1,000 rows."""
    assert len(attrition_data) >= 1000


def test_dataset_has_minimum_number_of_features(
    attrition_data: pd.DataFrame,
) -> None:
    """The project dataset must contain at least eight feature columns."""
    feature_count = len(attrition_data.columns) - 1

    assert feature_count >= 8


def test_expected_columns_are_present(
    attrition_data: pd.DataFrame,
) -> None:
    """Important dataset columns must be available."""
    expected_columns = {
        "Age",
        "Attrition",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "Education",
        "JobRole",
        "JobSatisfaction",
        "MonthlyIncome",
        "OverTime",
        "TotalWorkingYears",
        "YearsAtCompany",
    }

    missing_columns = expected_columns.difference(attrition_data.columns)

    assert not missing_columns, (
        f"Missing expected columns: {sorted(missing_columns)}"
    )


def test_target_contains_only_expected_values(
    attrition_data: pd.DataFrame,
) -> None:
    """Attrition must contain only the expected classification labels."""
    actual_values = set(
        attrition_data["Attrition"].dropna().unique()
    )

    assert actual_values == {"Yes", "No"}


def test_target_has_no_missing_values(
    attrition_data: pd.DataFrame,
) -> None:
    """The target column must not contain missing values."""
    assert attrition_data["Attrition"].isna().sum() == 0


def test_age_is_within_expected_range(
    attrition_data: pd.DataFrame,
) -> None:
    """Employee age values should be realistic."""
    assert attrition_data["Age"].between(18, 70).all()


def test_monthly_income_is_positive(
    attrition_data: pd.DataFrame,
) -> None:
    """Monthly income should always be greater than zero."""
    assert (attrition_data["MonthlyIncome"] > 0).all()


def test_distance_from_home_is_within_expected_range(
    attrition_data: pd.DataFrame,
) -> None:
    """Distance from home should remain within a reasonable range."""
    assert attrition_data["DistanceFromHome"].between(1, 100).all()


def test_dataset_contains_numeric_and_categorical_features(
    attrition_data: pd.DataFrame,
) -> None:
    """The dataset must contain both numeric and categorical features."""
    numeric_columns = attrition_data.select_dtypes(
        include="number"
    ).columns

    categorical_columns = attrition_data.select_dtypes(
        exclude="number"
    ).columns

    assert len(numeric_columns) > 0
    assert len(categorical_columns) > 0
