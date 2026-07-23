from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")


def introduce_missing_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    missing_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of the DataFrame with simulated missing values."""
    validate_dataframe(df)

    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between 0 and 1.")

    columns = list(columns)

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    result = df.copy(deep=True)

    for index, column in enumerate(columns):
        sampled_indices = result.sample(
            frac=missing_rate,
            random_state=random_state + index,
        ).index

        result.loc[sampled_indices, column] = np.nan

    return result


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target without modifying the input."""
    validate_dataframe(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].copy()

    return features, target


def identify_feature_types(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return lists of numeric and categorical feature names."""
    validate_dataframe(features)

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical data."""
    if not numeric_features and not categorical_features:
        raise ValueError("At least one feature column must be provided.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor