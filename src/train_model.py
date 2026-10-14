"""Model Training and Pipeline Serialization Module.

This module provides reusable functions for configuring, training, verifying,
and persisting baseline classification pipelines for the UCI AI4I 2020 dataset.
It enforces strict leakage prevention, pipeline encapsulation of preprocessing,
and reproducible model artifact serialization.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.preprocessing import (
    BASELINE_PREDICTORS,
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    create_preprocessor,
    separate_features_and_target,
    split_data,
    validate_input_data,
)

# Standard directory locations
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Baseline model identifiers and human-readable names
BASELINE_MODEL_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression (Unweighted)",
    "logistic_regression_balanced": "Logistic Regression (Balanced)",
    "random_forest": "Random Forest (Unweighted)",
    "random_forest_balanced": "Random Forest (Balanced)",
}

# Artifact file names
BASELINE_MODEL_FILENAMES: dict[str, str] = {
    "logistic_regression": "baseline_logistic_regression.joblib",
    "logistic_regression_balanced": "baseline_logistic_regression_balanced.joblib",
    "random_forest": "baseline_random_forest.joblib",
    "random_forest_balanced": "baseline_random_forest_balanced.joblib",
}


def get_baseline_model_configs(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Instantiate the four baseline estimator configurations.

    Configurations:
    1. Logistic Regression, no class weighting (class_weight=None).
    2. Logistic Regression, class_weight='balanced'.
    3. Random Forest, no class weighting (class_weight=None).
    4. Random Forest, class_weight='balanced'.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        dict[str, BaseEstimator]: Mapping of configuration key to instantiated estimator.
    """
    return {
        "logistic_regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
            class_weight=None,
        ),
        "logistic_regression_balanced": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight=None,
            n_jobs=-1,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def build_model_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Encapsulate the Phase 3 preprocessing ColumnTransformer and classifier into a Pipeline.

    This ensures that imputation, scaling, and one-hot encoding are coupled directly
    with the estimator for consistent inference and leak-free training.

    Args:
        classifier: A scikit-learn compatible classifier estimator.

    Returns:
        Pipeline: Full end-to-end pipeline with steps [('preprocessor', ...), ('classifier', ...)].
    """
    preprocessor = create_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_single_pipeline(
    classifier: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Construct and fit a single model pipeline strictly on training data.

    Args:
        classifier: Scikit-learn estimator instance.
        X_train: Training predictor DataFrame.
        y_train: Training target labels.

    Returns:
        Pipeline: Fitted pipeline ready for prediction.

    Raises:
        ValueError: If forbidden columns are detected in X_train.
    """
    forbidden_present = [col for col in X_train.columns if col in EXCLUDED_COLUMNS]
    if forbidden_present:
        raise ValueError(
            f"Target leakage hazard: Forbidden columns detected in X_train: {forbidden_present}"
        )

    pipeline = build_model_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline(
    pipeline: Pipeline,
    output_path: Path | str,
) -> Path:
    """Serialize a fitted pipeline to disk using joblib.

    Verifies that the object is a valid fitted Pipeline containing both
    preprocessor and classifier components before serializing.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        output_path: Destination file path.

    Returns:
        Path: Resolved path of saved pipeline artifact.

    Raises:
        ValueError: If pipeline is not a valid Pipeline or missing required steps.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a scikit-learn Pipeline, got {type(pipeline)}")

    required_steps = {"preprocessor", "classifier"}
    actual_steps = set(pipeline.named_steps.keys())
    if not required_steps.issubset(actual_steps):
        raise ValueError(
            f"Pipeline missing required steps. Expected {required_steps}, found {actual_steps}"
        )

    resolved_path = Path(output_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, resolved_path)
    return resolved_path


def load_pipeline(model_path: Path | str) -> Pipeline:
    """Load a serialized pipeline from disk and verify its integrity.

    Args:
        model_path: Path to serialized .joblib file.

    Returns:
        Pipeline: Loaded scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If loaded object is invalid or corrupted.
    """
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to deserialize pipeline from {path}: {exc}") from exc

    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Loaded artifact at {path} is not a Pipeline (type={type(pipeline)})")

    if "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError(
            f"Loaded pipeline from {path} does not contain required steps ('preprocessor', 'classifier')"
        )

    return pipeline


def train_all_baseline_models(
    data_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end training of the four baseline configurations.

    Workflow:
    1. Ingest raw dataset and validate schema.
    2. Separate baseline permitted predictors from target.
    3. Perform stratified 80/20 train/test split (test set held out untouched).
    4. For each baseline configuration:
       - Build full Pipeline(preprocessor, classifier).
       - Fit pipeline strictly on X_train and y_train.
       - Persist fitted pipeline to models_dir (if save_artifacts=True).
    5. Compile training metadata and return results.

    Args:
        data_path: Path to raw dataset CSV. Defaults to data/raw/ai4i2020.csv.
        models_dir: Directory where model artifacts are persisted. Defaults to models/.
        random_state: Random state seed for reproducible split and modeling (default 42).
        test_size: Proportion for held-out test split (default 0.2).
        save_artifacts: Whether to serialize trained models to disk.

    Returns:
        dict[str, Any]: Dictionary containing fitted models, artifact paths,
                        data splits (X_train, X_test, y_train, y_test), and metadata.
    """
    resolved_data_path = Path(data_path).resolve() if data_path else DEFAULT_DATA_PATH.resolve()
    resolved_models_dir = Path(models_dir).resolve() if models_dir else DEFAULT_MODELS_DIR.resolve()

    # Ingest and validate
    df = load_raw_data(resolved_data_path)
    validate_input_data(df, require_target=True, target_column=TARGET_COLUMN)

    # Separate permitted predictors and target
    X, y = separate_features_and_target(
        df,
        target_column=TARGET_COLUMN,
        feature_columns=BASELINE_PREDICTORS,
        exclude_columns=EXCLUDED_COLUMNS,
    )

    # Perform stratified split (strictly holding out test set)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    # Training class counts
    train_class_counts = {int(k): int(v) for k, v in y_train.value_counts().items()}
    test_class_counts = {int(k): int(v) for k, v in y_test.value_counts().items()}

    configs = get_baseline_model_configs(random_state=random_state)
    fitted_models: dict[str, Pipeline] = {}
    saved_paths: dict[str, Path] = {}
    model_metadata: dict[str, dict[str, Any]] = {}

    for key, estimator in configs.items():
        # Fit pipeline strictly on training data
        pipeline = train_single_pipeline(estimator, X_train, y_train)
        fitted_models[key] = pipeline

        # Verify no forbidden columns leaked into pipeline
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(fitted_preprocessor, "feature_names_in_"):
            leaked = [c for c in fitted_preprocessor.feature_names_in_ if c in EXCLUDED_COLUMNS]
            if leaked:
                raise ValueError(f"Leakage check failed for {key}: forbidden columns {leaked} in pipeline")

        # Save artifact if requested
        if save_artifacts:
            filename = BASELINE_MODEL_FILENAMES[key]
            out_file = resolved_models_dir / filename
            saved_path = save_pipeline(pipeline, out_file)
            saved_paths[key] = saved_path

        # Record estimator parameters
        classifier_step = pipeline.named_steps["classifier"]
        model_metadata[key] = {
            "model_name": BASELINE_MODEL_NAMES.get(key, key),
            "class_weight": getattr(classifier_step, "class_weight", None),
            "parameters": classifier_step.get_params(),
            "artifact_filename": BASELINE_MODEL_FILENAMES.get(key),
            "artifact_path": str(saved_paths.get(key, "")),
        }

    training_metadata: dict[str, Any] = {
        "dataset_path": str(resolved_data_path),
        "target_column": TARGET_COLUMN,
        "permitted_predictors": BASELINE_PREDICTORS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "random_state": random_state,
        "test_size": test_size,
        "train_sample_count": len(X_train),
        "test_sample_count": len(X_test),
        "train_class_distribution": train_class_counts,
        "test_class_distribution": test_class_counts,
        "models": model_metadata,
        "training_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    return {
        "models": fitted_models,
        "model_paths": saved_paths,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "training_metadata": training_metadata,
    }
