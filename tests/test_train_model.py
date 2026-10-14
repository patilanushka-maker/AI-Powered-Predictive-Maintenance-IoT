"""Unit and Integration tests for baseline model training module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import train_model
from src.preprocessing import BASELINE_PREDICTORS, EXCLUDED_COLUMNS, TARGET_COLUMN


@pytest.fixture
def synthetic_training_df() -> pd.DataFrame:
    """Generate a clean synthetic DataFrame with known distribution for training tests."""
    np.random.seed(42)
    n = 40
    data = {
        "UDI": list(range(1, n + 1)),
        "Product ID": [f"M_{i:04d}" for i in range(n)],
        "Type": ["L"] * 24 + ["M"] * 12 + ["H"] * 4,
        "Air temperature [K]": np.linspace(298.0, 302.0, n),
        "Process temperature [K]": np.linspace(308.0, 312.0, n),
        "Rotational speed [rpm]": np.linspace(1400, 1600, n),
        "Torque [Nm]": np.linspace(35.0, 45.0, n),
        "Tool wear [min]": list(range(0, n * 5, 5)),
        "Machine failure": [0] * 32 + [1] * 8,  # 20% positive class for stratified split stability
        "TWF": [0] * n,
        "HDF": [0] * n,
        "PWF": [0] * n,
        "OSF": [0] * n,
        "RNF": [0] * n,
    }
    return pd.DataFrame(data)


def test_baseline_model_configs_factory():
    """Verify that model factory returns the expected four configurations and parameters."""
    configs = train_model.get_baseline_model_configs(random_state=42)

    assert set(configs.keys()) == {
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest",
        "random_forest_balanced",
    }

    # Verify Logistic Regression configurations
    lr_unweighted = configs["logistic_regression"]
    assert isinstance(lr_unweighted, LogisticRegression)
    assert lr_unweighted.class_weight is None
    assert lr_unweighted.max_iter >= 1000

    lr_balanced = configs["logistic_regression_balanced"]
    assert isinstance(lr_balanced, LogisticRegression)
    assert lr_balanced.class_weight == "balanced"
    assert lr_balanced.max_iter >= 1000

    # Verify Random Forest configurations
    rf_unweighted = configs["random_forest"]
    assert isinstance(rf_unweighted, RandomForestClassifier)
    assert rf_unweighted.class_weight is None
    assert rf_unweighted.random_state == 42

    rf_balanced = configs["random_forest_balanced"]
    assert isinstance(rf_balanced, RandomForestClassifier)
    assert rf_balanced.class_weight == "balanced"
    assert rf_balanced.random_state == 42


def test_build_model_pipeline():
    """Verify that build_model_pipeline bundles preprocessor ColumnTransformer and estimator."""
    clf = LogisticRegression()
    pipeline = train_model.build_model_pipeline(clf)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert pipeline.named_steps["classifier"] is clf


def test_train_single_pipeline_on_synthetic_data(synthetic_training_df: pd.DataFrame):
    """Verify that training completes on a synthetic dataset and produces valid predictions."""
    X = synthetic_training_df[BASELINE_PREDICTORS].copy()
    y = synthetic_training_df[TARGET_COLUMN].copy()

    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = train_model.train_single_pipeline(clf, X, y)

    # Predict on raw feature DataFrame
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert set(np.unique(preds)).issubset({0, 1})
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_training_reproducibility(synthetic_training_df: pd.DataFrame):
    """Verify that model training is deterministic with fixed random seeds."""
    X = synthetic_training_df[BASELINE_PREDICTORS].copy()
    y = synthetic_training_df[TARGET_COLUMN].copy()

    configs1 = train_model.get_baseline_model_configs(random_state=42)
    configs2 = train_model.get_baseline_model_configs(random_state=42)

    pipe1 = train_model.train_single_pipeline(configs1["random_forest"], X, y)
    pipe2 = train_model.train_single_pipeline(configs2["random_forest"], X, y)

    probs1 = pipe1.predict_proba(X)
    probs2 = pipe2.predict_proba(X)

    np.testing.assert_array_equal(probs1, probs2)


def test_forbidden_columns_rejected_during_training(synthetic_training_df: pd.DataFrame):
    """Verify that forbidden identifier or target leakage columns trigger ValueError."""
    # Attempt to pass forbidden columns into training
    X_leaked = synthetic_training_df[["Type", "Air temperature [K]", "TWF", "UDI"]].copy()
    y = synthetic_training_df[TARGET_COLUMN].copy()

    clf = LogisticRegression()
    with pytest.raises(ValueError, match="Forbidden columns detected"):
        train_model.train_single_pipeline(clf, X_leaked, y)


def test_save_and_load_pipeline_joblib(tmp_path: Path, synthetic_training_df: pd.DataFrame):
    """Verify serialization and deserialization of fitted pipelines with joblib."""
    X = synthetic_training_df[BASELINE_PREDICTORS].copy()
    y = synthetic_training_df[TARGET_COLUMN].copy()

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline = train_model.train_single_pipeline(clf, X, y)

    model_path = tmp_path / "test_rf_model.joblib"
    saved_path = train_model.save_pipeline(pipeline, model_path)
    assert saved_path.is_file()

    loaded_pipeline = train_model.load_pipeline(saved_path)
    assert isinstance(loaded_pipeline, Pipeline)
    assert "preprocessor" in loaded_pipeline.named_steps
    assert "classifier" in loaded_pipeline.named_steps

    # Test that loaded pipeline accepts raw permitted predictors
    preds_orig = pipeline.predict(X)
    preds_loaded = loaded_pipeline.predict(X)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_load_pipeline_error_handling(tmp_path: Path):
    """Verify proper error handling when loading non-existent or invalid model files."""
    missing_file = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        train_model.load_pipeline(missing_file)

    corrupt_file = tmp_path / "corrupt.joblib"
    corrupt_file.write_text("invalid content", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to deserialize"):
        train_model.load_pipeline(corrupt_file)


def test_train_all_baseline_models_end_to_end(tmp_path: Path, synthetic_training_df: pd.DataFrame):
    """Verify orchestration of all 4 baseline models on a temporary dataset."""
    csv_path = tmp_path / "synthetic_dataset.csv"
    synthetic_training_df.to_csv(csv_path, index=False)
    models_dir = tmp_path / "models"

    results = train_model.train_all_baseline_models(
        data_path=csv_path,
        models_dir=models_dir,
        random_state=42,
        test_size=0.2,
        save_artifacts=True,
    )

    assert len(results["models"]) == 4
    assert len(results["model_paths"]) == 4

    # Verify all four artifacts were created on disk
    for key, path in results["model_paths"].items():
        assert path.is_file()
        assert path.name.endswith(".joblib")

    # Verify train/test shapes on 40 rows (80% train = 32, 20% test = 8)
    assert len(results["X_train"]) == 32
    assert len(results["X_test"]) == 8
    assert results["training_metadata"]["train_sample_count"] == 32
    assert results["training_metadata"]["test_sample_count"] == 8

    # Verify no forbidden columns appear in feature list
    for col in EXCLUDED_COLUMNS:
        assert col not in results["X_train"].columns
