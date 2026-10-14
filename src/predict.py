"""Batch and Streaming Inference Engine for Predictive Maintenance.

This module provides reusable, testable prediction and inference functions for
evaluating raw IoT sensor telemetry against trained classification pipelines.
It validates incoming records against the permitted baseline feature schema,
enforces strict boundaries to prevent target leakage, and produces structured,
interpretable prediction outputs without retraining or mutating input data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    BASELINE_PREDICTORS,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.train_model import (
    BASELINE_MODEL_FILENAMES,
    BASELINE_MODEL_NAMES,
    DEFAULT_MODELS_DIR,
    load_pipeline,
)

# Reference baseline model selection (strongest F1-score and PR-AUC from Phase 4)
DEFAULT_REFERENCE_MODEL_KEY: str = "random_forest"
DEFAULT_REFERENCE_MODEL_FILENAME: str = BASELINE_MODEL_FILENAMES[DEFAULT_REFERENCE_MODEL_KEY]

# Permitted categorical categories for machine 'Type'
PERMITTED_MACHINE_TYPES: set[str] = {"L", "M", "H"}

# Strict label mapping for binary classification
LABEL_MAPPING: dict[int, str] = {
    0: "No failure label predicted",
    1: "Failure label predicted",
}

# Forbidden input fields: post-hoc failure modes, identifiers, and target label
FORBIDDEN_INPUT_COLUMNS: set[str] = set(EXCLUDED_COLUMNS) | {TARGET_COLUMN}


def validate_threshold(threshold: float) -> float:
    """Validate that the decision threshold is numeric, finite, and strictly in (0, 1).

    Args:
        threshold: Decision threshold value to validate.

    Returns:
        float: Validated threshold.

    Raises:
        ValueError: If threshold is not a finite float strictly between 0 and 1.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"Threshold must be a numeric float, got {type(threshold).__name__}: {threshold}")

    threshold_val = float(threshold)
    if not math.isfinite(threshold_val):
        raise ValueError(f"Threshold must be finite, got: {threshold}")

    if not (0.0 < threshold_val < 1.0):
        raise ValueError(
            f"Decision threshold must be strictly between 0.0 and 1.0 (exclusive). Got: {threshold_val}"
        )

    return threshold_val


def load_inference_pipeline(
    model: str | Path | Pipeline | None = None,
) -> tuple[Pipeline, str]:
    """Load or resolve an inference pipeline without retraining.

    Supports:
    - None: Defaults to the provisional reference model (Random Forest, unweighted).
    - Model key string: One of the 4 Phase 4 baseline model keys.
    - File path: Path or str pointing to a serialized .joblib Pipeline file.
    - Pipeline instance: Directly validates and reuses an already loaded pipeline,
      avoiding redundant disk reads across repeated inferences.

    Args:
        model: Model identifier string, file path, existing Pipeline, or None.

    Returns:
        tuple[Pipeline, str]: The loaded scikit-learn Pipeline and its human-readable name.

    Raises:
        FileNotFoundError: If the specified model file does not exist on disk.
        ValueError: If model key is unrecognized or the artifact is corrupted.
    """
    if isinstance(model, Pipeline):
        # Verify required pipeline steps
        if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
            raise ValueError(
                "Provided Pipeline instance must contain 'preprocessor' and 'classifier' steps."
            )
        model_name = type(model.named_steps["classifier"]).__name__
        return model, model_name

    if model is None:
        target_path = DEFAULT_MODELS_DIR / DEFAULT_REFERENCE_MODEL_FILENAME
        model_key = DEFAULT_REFERENCE_MODEL_KEY
    elif isinstance(model, str) and model in BASELINE_MODEL_FILENAMES:
        target_path = DEFAULT_MODELS_DIR / BASELINE_MODEL_FILENAMES[model]
        model_key = model
    else:
        target_path = Path(model).resolve()
        model_key = target_path.stem

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at: {target_path}. "
            "Please ensure model training was executed via: python main.py --mode train"
        )

    pipeline = load_pipeline(target_path)
    model_name = BASELINE_MODEL_NAMES.get(model_key, target_path.stem)
    return pipeline, model_name


def validate_prediction_record(
    record: dict[str, Any] | pd.Series,
    record_index: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a single raw sensor telemetry record against the inference contract.

    Validation Rules:
    1. Rejects missing required predictor columns.
    2. Rejects forbidden columns (identifiers, failure modes, target label).
    3. Rejects unexpected extra columns.
    4. Rejects invalid categorical values (only 'L', 'M', 'H' allowed for 'Type').
    5. Rejects non-numeric sensor readings (booleans, strings, None).
    6. Rejects non-finite values (NaN, positive/negative infinity).
    7. Rejects negative rotational speed, torque, or tool wear.
    8. Rejects non-positive temperatures (Kelvin scale must be > 0).
    9. Generates domain plausibility warnings for abnormal physical operating states.
    10. Does not mutate the caller's input object.

    Args:
        record: Dictionary or pandas Series representing a single telemetry sample.
        record_index: Optional row index for contextual error traceability.

    Returns:
        tuple[dict[str, Any], list[str]]: Validated record dictionary and list of warnings.

    Raises:
        ValueError: If any validation rule is violated.
    """
    idx_str = f" at index {record_index}" if record_index is not None else ""

    if record is None:
        raise ValueError(f"Input record cannot be None{idx_str}.")

    if isinstance(record, pd.Series):
        record_dict = record.to_dict()
    elif isinstance(record, dict):
        record_dict = dict(record)
    else:
        raise ValueError(
            f"Expected input record to be a dict or pandas Series{idx_str}, got {type(record).__name__}."
        )

    # 1. Check for forbidden columns (target leakage / identifiers)
    detected_forbidden = [col for col in record_dict if col in FORBIDDEN_INPUT_COLUMNS]
    if detected_forbidden:
        raise ValueError(
            f"Forbidden column(s) detected in prediction input{idx_str}: {detected_forbidden}. "
            "Identifiers, post-hoc failure modes, and target labels must never enter the inference engine."
        )

    # 2. Check for missing required predictors
    missing_fields = [col for col in BASELINE_PREDICTORS if col not in record_dict]
    if missing_fields:
        raise ValueError(
            f"Missing required predictor field(s){idx_str}: {missing_fields}. "
            f"Required predictors: {BASELINE_PREDICTORS}"
        )

    # 3. Check for unexpected extra columns
    unexpected_fields = [col for col in record_dict if col not in BASELINE_PREDICTORS]
    if unexpected_fields:
        raise ValueError(
            f"Unexpected field(s) in prediction input{idx_str}: {unexpected_fields}. "
            f"Inference engine accepts only the six permitted predictors: {BASELINE_PREDICTORS}"
        )

    validated: dict[str, Any] = {}
    warnings: list[str] = []

    # 4. Categorical feature validation ('Type')
    machine_type = record_dict["Type"]
    if not isinstance(machine_type, str):
        raise ValueError(
            f"Invalid machine 'Type' data type{idx_str}: expected str, got {type(machine_type).__name__} ({machine_type})."
        )
    machine_type_clean = machine_type.strip()
    if machine_type_clean not in PERMITTED_MACHINE_TYPES:
        raise ValueError(
            f"Invalid machine 'Type' value{idx_str}: '{machine_type}'. "
            f"Permitted categories are: {sorted(list(PERMITTED_MACHINE_TYPES))}"
        )
    validated["Type"] = machine_type_clean

    # 5. Numerical sensor feature validation
    for num_col in NUMERICAL_FEATURES:
        raw_val = record_dict[num_col]

        # Explicitly reject booleans (which inherit from int in Python)
        if isinstance(raw_val, bool):
            raise ValueError(
                f"Invalid boolean value for numerical sensor '{num_col}'{idx_str}: {raw_val}. "
                "Expected a finite numeric float or integer."
            )

        # Ensure value is convertible to float
        if raw_val is None:
            raise ValueError(
                f"Sensor '{num_col}' is None or missing{idx_str}. Missing values are not permitted at inference boundary."
            )

        try:
            val_float = float(raw_val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Non-numeric value for sensor '{num_col}'{idx_str}: {raw_val} ({type(raw_val).__name__})."
            ) from exc

        # Check for NaN / Infinity
        if not math.isfinite(val_float):
            raise ValueError(
                f"Non-finite value (NaN or Infinity) detected for sensor '{num_col}'{idx_str}: {raw_val}."
            )

        validated[num_col] = val_float

    # 6. Physical validity checks (hard domain constraints)
    if validated["Rotational speed [rpm]"] < 0:
        raise ValueError(
            f"Rotational speed cannot be negative{idx_str}: {validated['Rotational speed [rpm]']} rpm."
        )

    if validated["Torque [Nm]"] < 0:
        raise ValueError(
            f"Torque cannot be negative{idx_str}: {validated['Torque [Nm]']} Nm."
        )

    if validated["Tool wear [min]"] < 0:
        raise ValueError(
            f"Tool wear cannot be negative{idx_str}: {validated['Tool wear [min]']} min."
        )

    if validated["Air temperature [K]"] <= 0:
        raise ValueError(
            f"Air temperature in Kelvin must be strictly positive{idx_str}: {validated['Air temperature [K]']} K."
        )

    if validated["Process temperature [K]"] <= 0:
        raise ValueError(
            f"Process temperature in Kelvin must be strictly positive{idx_str}: {validated['Process temperature [K]']} K."
        )

    # 7. Domain plausibility warnings (distinguishing basic validity from warnings)
    if validated["Process temperature [K]"] < validated["Air temperature [K]"]:
        warnings.append(
            f"Inverted thermal gradient: Process temperature ({validated['Process temperature [K]']:.2f} K) "
            f"is lower than Air temperature ({validated['Air temperature [K]']:.2f} K)."
        )

    if validated["Rotational speed [rpm]"] > 3500:
        warnings.append(
            f"Unusually high rotational speed: {validated['Rotational speed [rpm]']} rpm exceeds typical operational envelope."
        )

    if validated["Torque [Nm]"] > 85.0:
        warnings.append(
            f"Unusually high torque: {validated['Torque [Nm]']} Nm exceeds typical operational envelope."
        )

    if validated["Tool wear [min]"] > 300:
        warnings.append(
            f"Excessive tool wear: {validated['Tool wear [min]']} min exceeds normal maintenance schedule limits."
        )

    return validated, warnings


def record_to_dataframe(validated_record: dict[str, Any]) -> pd.DataFrame:
    """Convert a validated single record dictionary into a 1-row DataFrame.

    Ensures column ordering strictly matches BASELINE_PREDICTORS.

    Args:
        validated_record: Validated dictionary containing the 6 baseline features.

    Returns:
        pd.DataFrame: A single-row DataFrame.
    """
    row_data = {col: [validated_record[col]] for col in BASELINE_PREDICTORS}
    return pd.DataFrame(row_data)


def extract_positive_class_score(
    pipeline: Pipeline,
    df: pd.DataFrame,
) -> np.ndarray:
    """Extract positive-class failure prediction scores using dynamic estimator class ordering.

    Inspects pipeline classifier classes_ to locate class 1 rather than assuming
    it is always column index 1.

    Args:
        pipeline: Trained scikit-learn Pipeline.
        df: Input DataFrame containing permitted features.

    Returns:
        np.ndarray: 1D array of estimated positive-class failure probabilities in [0.0, 1.0].

    Raises:
        AttributeError: If pipeline does not support probability prediction.
        ValueError: If positive class label (1) is not found in classifier classes.
    """
    classifier = pipeline.named_steps["classifier"]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df)
        classes = list(getattr(classifier, "classes_", [0, 1]))
        if 1 in classes:
            pos_idx = classes.index(1)
        elif len(classes) == 1 and classes[0] == 1:
            pos_idx = 0
        else:
            raise ValueError(f"Positive class label '1' not found in classifier classes: {classes}")
        return probabilities[:, pos_idx]

    if hasattr(pipeline, "decision_function"):
        decision_scores = pipeline.decision_function(df)
        # Apply standard sigmoid transformation to map margin scores into [0, 1]
        return 1.0 / (1.0 + np.exp(-decision_scores))

    raise AttributeError(
        f"Classifier {type(classifier).__name__} does not support predict_proba or decision_function."
    )


def predict_single(
    record: dict[str, Any] | pd.Series,
    model: str | Path | Pipeline | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute failure classification inference on a single raw telemetry record.

    Workflow:
    1. Validates the decision threshold.
    2. Validates the raw input features without modifying the caller's input.
    3. Loads or reuses the model pipeline without retraining.
    4. Computes the positive-class prediction score.
    5. Applies the decision threshold to produce the predicted class.
    6. Constructs and returns the structured prediction contract.

    Args:
        record: Raw input dictionary or pandas Series.
        model: Model identifier key, path to .joblib artifact, existing Pipeline, or None (default).
        threshold: Decision threshold for positive classification (default 0.5).

    Returns:
        dict[str, Any]: Structured prediction result.
    """
    valid_thresh = validate_threshold(threshold)
    validated_record, warnings = validate_prediction_record(record)
    pipeline, model_name = load_inference_pipeline(model)

    df_single = record_to_dataframe(validated_record)
    scores = extract_positive_class_score(pipeline, df_single)
    pos_score = float(scores[0])

    pred_class = int(pos_score >= valid_thresh)
    pred_label = LABEL_MAPPING[pred_class]

    return {
        "model_name": model_name,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "positive_class_score": round(pos_score, 6),
        "threshold": round(valid_thresh, 6),
        "status": "success",
        "warnings": warnings,
    }


def predict_batch(
    records: list[dict[str, Any]] | pd.DataFrame,
    model: str | Path | Pipeline | None = None,
    threshold: float = 0.5,
    error_policy: str = "raise",
) -> list[dict[str, Any]]:
    """Execute failure classification inference across multiple raw telemetry records.

    Preserves row order and utilizes vectorized pipeline execution for efficiency.

    Error Policies:
    - 'raise' (default): Validates every record first. If any record violates validation
      rules, raises ValueError detailing the first failing record index and reason.
    - 'per_record': Returns an explicit error dictionary for invalid records
      ({'status': 'error', 'row_index': i, 'error': ...}) while executing vectorized
      predictions on valid records.

    Args:
        records: List of dictionaries or a pandas DataFrame.
        model: Model key, artifact path, Pipeline, or None.
        threshold: Decision threshold in (0, 1) (default 0.5).
        error_policy: Error handling policy ('raise' or 'per_record').

    Returns:
        list[dict[str, Any]]: List of structured prediction dictionaries matching input order.

    Raises:
        ValueError: If error_policy is invalid, records list is empty, or validation fails under 'raise'.
    """
    if error_policy not in {"raise", "per_record"}:
        raise ValueError(f"Invalid error_policy: '{error_policy}'. Must be 'raise' or 'per_record'.")

    valid_thresh = validate_threshold(threshold)
    pipeline, model_name = load_inference_pipeline(model)

    if records is None:
        raise ValueError("Batch records input cannot be None.")

    if isinstance(records, pd.DataFrame):
        records_list: list[dict[str, Any]] = records.to_dict(orient="records")
    elif isinstance(records, list):
        records_list = records
    else:
        raise ValueError(
            f"Expected records to be a list of dicts or a pandas DataFrame, got {type(records).__name__}."
        )

    if len(records_list) == 0:
        raise ValueError("Cannot perform batch prediction on empty input records.")

    # Validate each record preserving order
    validated_records: list[dict[str, Any] | None] = []
    record_warnings: list[list[str]] = []
    validation_errors: dict[int, str] = {}

    for idx, rec in enumerate(records_list):
        try:
            val_rec, warns = validate_prediction_record(rec, record_index=idx)
            validated_records.append(val_rec)
            record_warnings.append(warns)
        except Exception as exc:
            if error_policy == "raise":
                raise ValueError(f"Batch validation failed at row {idx}: {exc}") from exc
            validated_records.append(None)
            record_warnings.append([])
            validation_errors[idx] = str(exc)

    # Perform vectorized prediction on valid records
    valid_indices = [i for i, r in enumerate(validated_records) if r is not None]
    results: list[dict[str, Any]] = [{}] * len(records_list)

    if valid_indices:
        valid_rows = [validated_records[i] for i in valid_indices if validated_records[i] is not None]
        # Construct DataFrame strictly with BASELINE_PREDICTORS
        df_valid = pd.DataFrame(valid_rows)[BASELINE_PREDICTORS]
        scores = extract_positive_class_score(pipeline, df_valid)

        for pos_in_valid, orig_idx in enumerate(valid_indices):
            score_val = float(scores[pos_in_valid])
            pred_class = int(score_val >= valid_thresh)
            results[orig_idx] = {
                "row_index": orig_idx,
                "model_name": model_name,
                "predicted_class": pred_class,
                "predicted_label": LABEL_MAPPING[pred_class],
                "positive_class_score": round(score_val, 6),
                "threshold": round(valid_thresh, 6),
                "status": "success",
                "warnings": record_warnings[orig_idx],
            }

    # Populate explicit error records for 'per_record' policy
    for err_idx, err_msg in validation_errors.items():
        results[err_idx] = {
            "row_index": err_idx,
            "model_name": model_name,
            "predicted_class": None,
            "predicted_label": None,
            "positive_class_score": None,
            "threshold": round(valid_thresh, 6),
            "status": "error",
            "error": err_msg,
            "warnings": [],
        }

    return results


def predict(
    input_data: dict[str, Any] | pd.Series | list[dict[str, Any]] | pd.DataFrame,
    model: str | Path | Pipeline | None = None,
    threshold: float = 0.5,
    error_policy: str = "raise",
) -> dict[str, Any] | list[dict[str, Any]]:
    """Unified entry-point for predictive maintenance telemetry inference.

    Automatically dispatches single records to predict_single and multiple
    records to predict_batch.

    Args:
        input_data: Single dictionary/Series, or list of dictionaries / DataFrame.
        model: Optional model key, file path, Pipeline, or None (default).
        threshold: Decision threshold (default 0.5).
        error_policy: Batch error handling policy ('raise' or 'per_record').

    Returns:
        dict[str, Any] for single record input, or list[dict[str, Any]] for batch input.
    """
    if isinstance(input_data, (list, pd.DataFrame)):
        return predict_batch(
            records=input_data,
            model=model,
            threshold=threshold,
            error_policy=error_policy,
        )

    if isinstance(input_data, (dict, pd.Series)):
        return predict_single(
            record=input_data,
            model=model,
            threshold=threshold,
        )

    raise ValueError(
        f"Unsupported input type for predict: {type(input_data).__name__}. "
        "Expected dict, pd.Series, list of dicts, or pd.DataFrame."
    )
