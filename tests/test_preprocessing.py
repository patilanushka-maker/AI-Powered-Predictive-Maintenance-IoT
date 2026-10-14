"""Unit and Integration tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src import preprocessing

# Sample schema matching UCI AI4I 2020 layout
FULL_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Generate a clean synthetic DataFrame with known distribution for unit testing."""
    np.random.seed(42)
    n = 20
    data = {
        "UDI": list(range(1, n + 1)),
        "Product ID": [f"M_{i:04d}" for i in range(n)],
        "Type": ["L"] * 12 + ["M"] * 6 + ["H"] * 2,
        "Air temperature [K]": np.linspace(298.0, 302.0, n),
        "Process temperature [K]": np.linspace(308.0, 312.0, n),
        "Rotational speed [rpm]": np.linspace(1400, 1600, n),
        "Torque [Nm]": np.linspace(35.0, 45.0, n),
        "Tool wear [min]": list(range(0, n * 5, 5)),
        "Machine failure": [0] * 16 + [1] * 4,  # 20% positive class for test stability
        "TWF": [0] * n,
        "HDF": [0] * n,
        "PWF": [0] * n,
        "OSF": [0] * n,
        "RNF": [0] * n,
    }
    return pd.DataFrame(data)


def test_validate_input_data_empty():
    """Verify ValueError is raised on empty or non-DataFrame input."""
    with pytest.raises(ValueError, match="non-null pandas DataFrame"):
        preprocessing.validate_input_data(None)

    with pytest.raises(ValueError, match="zero rows"):
        preprocessing.validate_input_data(pd.DataFrame())


def test_validate_input_data_missing_predictor(synthetic_sample_df: pd.DataFrame):
    """Verify ValueError when required predictor column is missing."""
    df_missing = synthetic_sample_df.drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="Missing required predictor columns.*Torque \\[Nm\\]"):
        preprocessing.validate_input_data(df_missing)


def test_validate_input_data_missing_target(synthetic_sample_df: pd.DataFrame):
    """Verify ValueError when target column is missing."""
    df_missing = synthetic_sample_df.drop(columns=["Machine failure"])
    with pytest.raises(ValueError, match="Missing required target column"):
        preprocessing.validate_input_data(df_missing, require_target=True)


def test_validate_input_data_invalid_target_values(synthetic_sample_df: pd.DataFrame):
    """Verify ValueError when target contains non-binary values."""
    df_invalid = synthetic_sample_df.copy()
    df_invalid.loc[0, "Machine failure"] = 2
    with pytest.raises(ValueError, match="strictly binary values"):
        preprocessing.validate_input_data(df_invalid, require_target=True)

    df_null = synthetic_sample_df.copy()
    df_null.loc[0, "Machine failure"] = None
    with pytest.raises(ValueError, match="null or missing"):
        preprocessing.validate_input_data(df_null, require_target=True)


def test_separate_features_and_target_exclusion(synthetic_sample_df: pd.DataFrame):
    """Verify that all forbidden identifier and failure mode columns are excluded from X."""
    X, y = preprocessing.separate_features_and_target(synthetic_sample_df)

    assert set(X.columns) == set(preprocessing.BASELINE_PREDICTORS)
    assert len(X.columns) == 6
    assert len(y) == len(synthetic_sample_df)
    assert y.name == "Machine failure"

    # Strict check: None of the forbidden columns may exist in X
    for col in preprocessing.EXCLUDED_COLUMNS:
        assert col not in X.columns


def test_immutability_of_input_dataframe(synthetic_sample_df: pd.DataFrame):
    """Verify that separation does not mutate the original input DataFrame."""
    df_copy = synthetic_sample_df.copy()
    X, y = preprocessing.separate_features_and_target(synthetic_sample_df)

    pd.testing.assert_frame_equal(synthetic_sample_df, df_copy)


def test_split_data_reproducibility(synthetic_sample_df: pd.DataFrame):
    """Verify split reproducibility and dimension preservation with random_state=42."""
    X, y = preprocessing.separate_features_and_target(synthetic_sample_df)

    X_train1, X_test1, y_train1, y_test1 = preprocessing.split_data(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = preprocessing.split_data(X, y, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)

    # Dimensions on 20 rows with 20% test
    assert len(X_train1) == 16
    assert len(X_test1) == 4
    # Stratified: class 1 was 4/20 (20%), so 16*0.2 = 3.2 -> ~3 positive in train, ~1 in test
    assert (y_train1 == 1).sum() > 0
    assert (y_test1 == 1).sum() > 0


def test_numeric_and_categorical_pipelines_imputation():
    """Verify imputation behavior on synthetic data with missing values."""
    train_data = pd.DataFrame({
        "Type": ["L", "M", "L", "H"],
        "Air temperature [K]": [300.0, 302.0, np.nan, 304.0],  # Median of 300, 302, 304 is 302.0
        "Process temperature [K]": [310.0, 311.0, 312.0, 313.0],
        "Rotational speed [rpm]": [1500, 1550, 1600, 1650],
        "Torque [Nm]": [40.0, 42.0, 44.0, 46.0],
        "Tool wear [min]": [10, 20, 30, 40],
    })

    test_data = pd.DataFrame({
        "Type": [None],
        "Air temperature [K]": [None],
        "Process temperature [K]": [311.0],
        "Rotational speed [rpm]": [1550],
        "Torque [Nm]": [42.0],
        "Tool wear [min]": [20],
    })

    preprocessor = preprocessing.create_preprocessor()
    preprocessor.fit(train_data)
    transformed_test = preprocessor.transform(test_data)

    assert not np.isnan(transformed_test).any()
    assert transformed_test.shape == (1, 8)


def test_unseen_categorical_handling():
    """Verify that unseen categorical values do not crash and encode to zeros."""
    train_data = pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [300.0, 301.0, 302.0],
        "Process temperature [K]": [310.0, 311.0, 312.0],
        "Rotational speed [rpm]": [1500, 1520, 1540],
        "Torque [Nm]": [40.0, 41.0, 42.0],
        "Tool wear [min]": [10, 20, 30],
    })

    # Unseen category 'SPECIAL_EDITION'
    test_data = pd.DataFrame({
        "Type": ["SPECIAL_EDITION"],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]": [1500],
        "Torque [Nm]": [40.0],
        "Tool wear [min]": [10],
    })

    preprocessor = preprocessing.create_preprocessor()
    preprocessor.fit(train_data)
    transformed = preprocessor.transform(test_data)

    assert transformed.shape == (1, 8)
    # The last 3 columns correspond to OneHotEncoder categories ('Type_H', 'Type_L', 'Type_M')
    # Unseen category should result in all 0s for one-hot encoded columns
    cat_columns_transformed = transformed[0, -3:]
    np.testing.assert_array_equal(cat_columns_transformed, [0.0, 0.0, 0.0])


def test_no_data_leakage_scaler_fitted_on_train_only():
    """Verify that scaler parameters reflect training distribution only, not test."""
    train_data = pd.DataFrame({
        "Type": ["L", "L"],
        "Air temperature [K]": [300.0, 300.0],  # Constant in train (mean = 300, scale = 1 or std = 0)
        "Process temperature [K]": [310.0, 310.0],
        "Rotational speed [rpm]": [1500, 1500],
        "Torque [Nm]": [40.0, 40.0],
        "Tool wear [min]": [10, 10],
    })

    # Outlier test data
    test_data = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [9999.0],
        "Process temperature [K]": [9999.0],
        "Rotational speed [rpm]": [9999],
        "Torque [Nm]": [9999.0],
        "Tool wear [min]": [9999],
    })

    preprocessor = preprocessing.create_preprocessor()
    preprocessor.fit(train_data)

    # Inspect the internal numeric transformer scaler mean
    numeric_pipe = preprocessor.named_transformers_["num"]
    scaler = numeric_pipe.named_steps["scaler"]

    # Mean of training data must be 300.0, not affected by test data (9999)
    assert scaler.mean_[0] == 300.0


def test_preprocess_data_end_to_end(synthetic_sample_df: pd.DataFrame):
    """Verify full preprocess_data orchestration on synthetic dataset."""
    results = preprocessing.preprocess_data(synthetic_sample_df, test_size=0.2, random_state=42)

    assert results["train_shape"] == (16, 6)
    assert results["test_shape"] == (4, 6)
    assert results["transformed_train_shape"] == (16, 8)
    assert results["transformed_test_shape"] == (4, 8)
    assert len(results["feature_names_out"]) == 8
    assert results["preprocessor"] is not None


def test_real_dataset_preprocessing_integration():
    """Integration test verifying preprocessing pipeline on real data/raw/ai4i2020.csv."""
    from src.data_loader import get_raw_data_path, load_raw_data

    raw_path = get_raw_data_path()
    if not raw_path.is_file():
        pytest.skip("Official dataset data/raw/ai4i2020.csv not found.")

    df = load_raw_data(raw_path)
    results = preprocessing.preprocess_data(df, test_size=0.2, random_state=42)

    # Shape checks: 10,000 -> 8,000 train, 2,000 test
    assert results["train_shape"] == (8000, 6)
    assert results["test_shape"] == (2000, 6)

    # Transformed features: 5 numeric + 3 one-hot categories (H, L, M) = 8
    assert results["transformed_train_shape"] == (8000, 8)
    assert results["transformed_test_shape"] == (2000, 8)

    # Stratified target counts
    # Class 1: 339 total -> 271 train (3.39%), 68 test (3.40%)
    assert results["train_class_counts"] == {0: 7729, 1: 271}
    assert results["test_class_counts"] == {0: 1932, 1: 68}

    # Transformed arrays must have zero nulls and correct dtype
    assert not np.isnan(results["X_train_transformed"]).any()
    assert not np.isnan(results["X_test_transformed"]).any()
