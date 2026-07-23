from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessor,
    identify_feature_types,
    introduce_missing_values,
    split_features_target,
    validate_dataframe,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a small mixed-type DataFrame for testing."""
    return pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45],
            "MonthlyIncome": [3000, 4000, 5000, 6000, 7000],
            "Department": [
                "Sales",
                "Research",
                "Sales",
                "Human Resources",
                "Research",
            ],
            "OverTime": ["Yes", "No", "No", "Yes", "No"],
            "Attrition": ["No", "Yes", "No", "No", "Yes"],
        }
    )


def test_validate_dataframe_accepts_valid_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    """A valid non-empty DataFrame should not raise an exception."""
    validate_dataframe(sample_dataframe)


def test_validate_dataframe_rejects_non_dataframe() -> None:
    """A non-DataFrame input should raise TypeError."""
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_dataframe([1, 2, 3])


def test_validate_dataframe_rejects_empty_dataframe() -> None:
    """An empty DataFrame should raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_dataframe(pd.DataFrame())


def test_introduce_missing_values_adds_missing_values(
    sample_dataframe: pd.DataFrame,
) -> None:
    """The function should introduce missing values in selected columns."""
    result = introduce_missing_values(
        df=sample_dataframe,
        columns=["MonthlyIncome", "OverTime"],
        missing_rate=0.40,
        random_state=42,
    )

    assert result["MonthlyIncome"].isna().sum() > 0
    assert result["OverTime"].isna().sum() > 0


def test_introduce_missing_values_does_not_modify_original(
    sample_dataframe: pd.DataFrame,
) -> None:
    """The function should return a copy and preserve the original data."""
    original = sample_dataframe.copy(deep=True)

    introduce_missing_values(
        df=sample_dataframe,
        columns=["MonthlyIncome"],
        missing_rate=0.40,
        random_state=42,
    )

    pd.testing.assert_frame_equal(sample_dataframe, original)


def test_introduce_missing_values_rejects_invalid_rate(
    sample_dataframe: pd.DataFrame,
) -> None:
    """A missing rate outside zero to one should raise ValueError."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        introduce_missing_values(
            df=sample_dataframe,
            columns=["MonthlyIncome"],
            missing_rate=1.5,
        )


def test_introduce_missing_values_rejects_unknown_column(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Unknown columns should raise ValueError."""
    with pytest.raises(ValueError, match="Columns not found"):
        introduce_missing_values(
            df=sample_dataframe,
            columns=["UnknownColumn"],
            missing_rate=0.10,
        )


def test_split_features_target_returns_correct_outputs(
    sample_dataframe: pd.DataFrame,
) -> None:
    """The target should be removed from features and returned separately."""
    features, target = split_features_target(
        sample_dataframe,
        target_column="Attrition",
    )

    assert "Attrition" not in features.columns
    assert target.name == "Attrition"
    assert len(features) == len(target)


def test_split_features_target_rejects_missing_target(
    sample_dataframe: pd.DataFrame,
) -> None:
    """A missing target column should raise ValueError."""
    with pytest.raises(ValueError, match="Target column"):
        split_features_target(
            sample_dataframe,
            target_column="MissingTarget",
        )


def test_identify_feature_types(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Numeric and categorical feature types should be identified correctly."""
    features = sample_dataframe.drop(columns=["Attrition"])

    numeric_features, categorical_features = identify_feature_types(features)

    assert set(numeric_features) == {"Age", "MonthlyIncome"}
    assert set(categorical_features) == {"Department", "OverTime"}


def test_preprocessor_handles_missing_values_and_categories(
    sample_dataframe: pd.DataFrame,
) -> None:
    """The transformer should impute missing values and encode categories."""
    features = sample_dataframe.drop(columns=["Attrition"]).copy()

    features.loc[0, "MonthlyIncome"] = np.nan
    features.loc[1, "Department"] = np.nan

    numeric_features, categorical_features = identify_feature_types(features)

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unknown_category(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Unseen categories should not cause transformation failures."""
    training_features = sample_dataframe.drop(columns=["Attrition"]).copy()

    numeric_features, categorical_features = identify_feature_types(
        training_features
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    preprocessor.fit(training_features)

    new_data = training_features.iloc[[0]].copy()
    new_data["Department"] = "New Department"

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 1