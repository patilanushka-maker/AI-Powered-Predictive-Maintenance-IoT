"""Telemetry Cleaning, Transformation, and Preprocessing Module.

This module provides reusable scikit-learn preprocessing pipelines for
industrial machine sensor telemetry (UCI AI4I 2020 Predictive Maintenance).
It enforces strict column separation, prevents target leakage, and handles
missing values and unseen categories robustly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target Column Definition
TARGET_COLUMN: str = "Machine failure"

# Columns Strictly Excluded from Baseline Predictors
EXCLUDED_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# Numerical Sensor Telemetry Features
NUMERICAL_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Categorical Machinery Metadata Features
CATEGORICAL_FEATURES: list[str] = [
    "Type",
]

# Full Permitted Baseline Feature List
BASELINE_PREDICTORS: list[str] = (
    CATEGORICAL_FEATURES + NUMERICAL_FEATURES
)


def validate_input_data(
    df: pd.DataFrame,
    require_target: bool = True,
    target_column: str = TARGET_COLUMN,
    required_features: list[str] | None = None,
) -> None:
    """Validate input DataFrame for structure, presence of features, and valid target.

    Args:
        df: Input DataFrame to validate.
        require_target: Whether the target column must be present and verified.
        target_column: Name of the target column.
        required_features: List of required predictor columns (defaults to BASELINE_PREDICTORS).

    Raises:
        ValueError: If DataFrame is empty, missing required columns, or has invalid target values.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a non-null pandas DataFrame.")

    if df.empty or len(df) == 0:
        raise ValueError("Input DataFrame contains zero rows.")

    features_to_check = required_features if required_features is not None else BASELINE_PREDICTORS
    missing_features = [col for col in features_to_check if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required predictor columns: {missing_features}")

    if require_target:
        if target_column not in df.columns:
            raise ValueError(f"Missing required target column '{target_column}' in DataFrame.")

        target_series = df[target_column]
        if target_series.isnull().any():
            raise ValueError(f"Target column '{target_column}' contains null or missing values.")

        unique_vals = set(target_series.unique())
        # Target must be strictly binary {0, 1} (or a subset thereof)
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Target column '{target_column}' must contain strictly binary values (0 and 1). "
                f"Observed invalid values: {unique_vals - {0, 1}}"
            )


def separate_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate baseline predictor features and target label from raw or cleaned DataFrame.

    Ensures forbidden identifier and leakage columns are strictly excluded.
    Does not modify the original DataFrame in-place.

    Args:
        df: Input DataFrame.
        target_column: Name of target column.
        feature_columns: Optional explicit list of feature names. Defaults to BASELINE_PREDICTORS.
        exclude_columns: Optional list of columns to strictly prohibit. Defaults to EXCLUDED_COLUMNS.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Predictor DataFrame (X) and target Series (y).

    Raises:
        ValueError: If validation fails or forbidden columns are inadvertently requested.
    """
    validate_input_data(df, require_target=True, target_column=target_column)

    features = feature_columns if feature_columns is not None else BASELINE_PREDICTORS
    forbidden = set(exclude_columns if exclude_columns is not None else EXCLUDED_COLUMNS)

    # Ensure no forbidden columns leak into predictor set
    leakage_detected = [col for col in features if col in forbidden]
    if leakage_detected:
        raise ValueError(
            f"Target leakage hazard: Forbidden columns detected in requested predictors: {leakage_detected}"
        )

    X = df[features].copy()
    y = df[target_column].copy()

    return X, y


def create_numeric_transformer() -> Pipeline:
    """Construct standard scikit-learn pipeline for numerical sensor features.

    Implements:
    1. SimpleImputer with median strategy.
    2. StandardScaler for zero-mean, unit-variance scaling.

    Returns:
        Pipeline: Numeric feature processing pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def create_categorical_transformer() -> Pipeline:
    """Construct standard scikit-learn pipeline for categorical features.

    Implements:
    1. SimpleImputer with most_frequent strategy.
    2. OneHotEncoder with handle_unknown='ignore' to gracefully handle unseen categories.

    Returns:
        Pipeline: Categorical feature processing pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def create_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Construct composite ColumnTransformer combining numeric and categorical pipelines.

    Args:
        numeric_features: List of numeric column names. Defaults to NUMERICAL_FEATURES.
        categorical_features: List of categorical column names. Defaults to CATEGORICAL_FEATURES.

    Returns:
        ColumnTransformer: Preprocessing transformer ready for fitting on training data.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERICAL_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", create_numeric_transformer(), num_cols),
            ("cat", create_categorical_transformer(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform reproducible train/test split with stratification by default.

    Note:
        The 80/20 stratified random split provides a baseline evaluation setup.
        In production systems with temporal drift or serial correlation, time-based
        or machine-held-out splits should be considered.

    Args:
        X: Predictor DataFrame.
        y: Target Series.
        test_size: Proportion of dataset allocated to test split (default 0.2).
        random_state: Random seed for reproducibility (default 42).
        stratify: Whether to preserve class proportions across splits (default True).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if len(X) != len(y):
        raise ValueError(
            f"Length mismatch: X has {len(X)} rows while y has {len(y)} rows."
        )

    stratify_target = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Execute complete end-to-end preprocessing workflow without data leakage.

    Steps:
    1. Validates input schema and target constraints.
    2. Separates baseline predictors from target (excluding identifiers and failure modes).
    3. Splits data into train and test subsets using stratified splitting.
    4. Fits the ColumnTransformer STRICTLY on X_train.
    5. Transforms X_train and X_test using the fitted transformer.

    Args:
        df: Input DataFrame containing raw or loaded telemetry.
        test_size: Test split proportion (default 0.2).
        random_state: Random state seed (default 42).
        target_column: Primary target column name.
        feature_columns: Optional explicit list of feature names.

    Returns:
        dict: Preprocessing artifacts including train/test splits, transformed matrices,
              fitted preprocessor, feature names, and class distribution metadata.
    """
    X, y = separate_features_and_target(
        df,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    preprocessor = create_preprocessor()

    # Fit transformer STRICTLY on training predictors
    preprocessor.fit(X_train)

    # Transform both splits using parameters learned only from X_train
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract output feature names
    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names_out = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

    train_counts = {int(k): int(v) for k, v in y_train.value_counts().items()}
    test_counts = {int(k): int(v) for k, v in y_test.value_counts().items()}

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "preprocessor": preprocessor,
        "feature_names_out": feature_names_out,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "transformed_train_shape": X_train_transformed.shape,
        "transformed_test_shape": X_test_transformed.shape,
    }
