"""Model Evaluation, Diagnostics, and Benchmark Reporting Module.

This module computes performance metrics for baseline failure classification models
on the held-out test split of the UCI AI4I 2020 dataset. It persists reproducible
evaluation reports (JSON, CSV, confusion matrices) and enforces strict separation
between test evaluation and validation-based threshold analysis.
"""

from __future__ import annotations

import datetime
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.preprocessing import (
    BASELINE_PREDICTORS,
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    separate_features_and_target,
    split_data,
    validate_input_data,
)
from src.train_model import (
    BASELINE_MODEL_FILENAMES,
    BASELINE_MODEL_NAMES,
    DEFAULT_MODELS_DIR,
    load_pipeline,
)

# Standard evaluation output directory
DEFAULT_OUTPUT_METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive classification performance metrics for binary failure detection.

    Computes precision, recall, F1-score, ROC-AUC, average precision (PR-AUC),
    accuracy, and confusion matrix components with explicit positive class (1).

    Args:
        y_true: Ground truth binary target labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted failure probabilities (scores in [0.0, 1.0]).
        threshold: Decision threshold used to convert probabilities into binary labels.

    Returns:
        dict[str, Any]: Dictionary of metric values (all JSON-serializable standard types).

    Raises:
        ValueError: If array lengths do not match or inputs are empty.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Dimension mismatch: y_true has {len(y_true_arr)}, y_pred has {len(y_pred_arr)}, "
            f"and y_prob has {len(y_prob_arr)} elements."
        )

    # Confusion matrix extraction
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Positive-class metrics (failure = 1)
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0.0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0.0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0.0))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Area under curves (ROC-AUC and Average Precision / PR-AUC)
    # Check if both classes are present in y_true
    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) > 1:
        roc_auc: float | None = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc: float | None = float(average_precision_score(y_true_arr, y_prob_arr, pos_label=1))
    else:
        roc_auc = None
        pr_auc = None

    return {
        "threshold": float(threshold),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
        "average_precision": round(pr_auc, 6) if pr_auc is not None else None,
        "accuracy": round(acc, 6),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a single trained pipeline on held-out test data at a specified threshold.

    Args:
        pipeline: Trained scikit-learn Pipeline with steps [preprocessor, classifier].
        X_test: Held-out test predictor DataFrame.
        y_test: Held-out test target Series.
        threshold: Classification probability threshold (default 0.5).

    Returns:
        dict[str, Any]: Metric dictionary.

    Raises:
        ValueError: If input dimensions mismatch or forbidden columns are detected.
    """
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Length mismatch: X_test has {len(X_test)} rows but y_test has {len(y_test)} rows."
        )

    # Generate probabilities for positive class (1 = Machine failure)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        # Binary classifier: probability of failure is column index 1
        pos_class_idx = 1 if probabilities.shape[1] > 1 else 0
        y_prob = probabilities[:, pos_class_idx]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        # Sigmoid transform for ROC-AUC / PR-AUC score calculation
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise AttributeError("Pipeline classifier must implement predict_proba or decision_function.")

    # Apply classification threshold
    y_pred = (y_prob >= threshold).astype(int)

    return compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=threshold,
    )


def evaluate_all_models(
    models: dict[str, Pipeline] | dict[str, Path | str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Evaluate all baseline models on the exact same held-out test split.

    Args:
        models: Dictionary mapping model_key to either a fitted Pipeline or a Path to a .joblib file.
        X_test: Held-out test predictors.
        y_test: Held-out test target.
        threshold: Classification threshold (default 0.5).

    Returns:
        dict[str, dict[str, Any]]: Evaluation metrics keyed by model identifier.
    """
    results: dict[str, dict[str, Any]] = {}

    for key, model_or_path in models.items():
        if isinstance(model_or_path, (str, Path)):
            pipeline = load_pipeline(model_or_path)
        else:
            pipeline = model_or_path

        metrics = evaluate_pipeline(pipeline, X_test, y_test, threshold=threshold)
        metrics["model_name"] = BASELINE_MODEL_NAMES.get(key, key)
        results[key] = metrics

    return results


def format_metrics_summary_table(evaluation_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Format multiple model evaluation outputs into a comparative summary DataFrame.

    Args:
        evaluation_results: Dictionary of model evaluation results from evaluate_all_models.

    Returns:
        pd.DataFrame: Tabular comparison summary.
    """
    rows = []
    for key, m in evaluation_results.items():
        rows.append({
            "model_key": key,
            "model_name": m.get("model_name", BASELINE_MODEL_NAMES.get(key, key)),
            "threshold": m["threshold"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "average_precision": m["average_precision"],
            "accuracy": m["accuracy"],
            "true_positives": m["true_positives"],
            "false_positives": m["false_positives"],
            "true_negatives": m["true_negatives"],
            "false_negatives": m["false_negatives"],
        })

    df_summary = pd.DataFrame(rows)
    return df_summary.sort_values(by="f1_score", ascending=False).reset_index(drop=True)


def analyze_training_validation_thresholds(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    candidate_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Conduct training-only cross-validation threshold analysis.

    IMPORTANT: This threshold analysis is strictly performed using out-of-fold
    predictions on X_train. The test set is NOT touched during this analysis.

    Args:
        pipeline: Unfitted or fitted Pipeline.
        X_train: Training predictor DataFrame.
        y_train: Training target labels.
        cv: Number of stratified cross-validation folds.
        candidate_thresholds: List of candidate decision thresholds to evaluate.

    Returns:
        dict[str, Any]: Candidate thresholds, validation metrics per threshold,
                        and the threshold optimizing validation F1.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # Generate out-of-fold predicted probabilities strictly within training data
    oof_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_results = []
    best_f1 = -1.0
    best_threshold = 0.5

    for t in candidate_thresholds:
        y_pred = (oof_probs >= t).astype(int)
        m = compute_classification_metrics(y_train, y_pred, oof_probs, threshold=t)
        threshold_results.append({
            "threshold": t,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "true_positives": m["true_positives"],
            "false_positives": m["false_positives"],
        })
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_threshold = t

    return {
        "analysis_type": "training_only_stratified_cv_threshold_sweep",
        "cv_folds": cv,
        "sample_count": len(X_train),
        "best_f1_threshold": best_threshold,
        "best_f1_value": best_f1,
        "threshold_evaluations": threshold_results,
        "policy_note": (
            "Threshold selection is a decision policy balancing precision and recall. "
            "It does NOT represent a calibrated real-world failure probability."
        ),
    }


def save_evaluation_reports(
    evaluation_results: dict[str, dict[str, Any]],
    output_dir: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist evaluation results as JSON report, CSV summary, and confusion matrices JSON.

    Args:
        evaluation_results: Results from evaluate_all_models.
        output_dir: Output directory (defaults to outputs/metrics/).
        metadata: Additional metadata (split parameters, versions, seeds).

    Returns:
        dict[str, Path]: Paths to written report artifacts.
    """
    resolved_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_METRICS_DIR.resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON Report
    report_path = resolved_dir / "baseline_evaluation_report.json"
    full_report: dict[str, Any] = {
        "report_title": "AI4I 2020 Predictive Maintenance Baseline Model Evaluation",
        "evaluation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metadata": metadata or {},
        "models": evaluation_results,
    }

    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(full_report, fp, indent=2)

    # 2. CSV Summary
    csv_path = resolved_dir / "baseline_models_summary.csv"
    summary_df = format_metrics_summary_table(evaluation_results)
    summary_df.to_csv(csv_path, index=False)

    # 3. Confusion Matrices JSON
    cm_path = resolved_dir / "confusion_matrices.json"
    cm_data: dict[str, Any] = {}
    for key, m in evaluation_results.items():
        cm_data[key] = {
            "model_name": m.get("model_name", BASELINE_MODEL_NAMES.get(key, key)),
            "threshold": m["threshold"],
            "true_positives": m["true_positives"],
            "false_positives": m["false_positives"],
            "true_negatives": m["true_negatives"],
            "false_negatives": m["false_negatives"],
            "confusion_matrix_2x2": m["confusion_matrix"],
            "matrix_layout": [["TN", "FP"], ["FN", "TP"]],
        }

    with open(cm_path, "w", encoding="utf-8") as fp:
        json.dump(cm_data, fp, indent=2)

    return {
        "json_report": report_path,
        "csv_summary": csv_path,
        "confusion_matrices": cm_path,
    }


def run_baseline_evaluation(
    models_dir: Path | str | None = None,
    data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    threshold: float = 0.5,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Execute complete held-out evaluation workflow across all four baseline models.

    Workflow:
    1. Loads the validated raw dataset.
    2. Generates the reproducible 80/20 stratified split (seed 42).
    3. Verifies that all 4 model binaries exist in models_dir.
    4. Evaluates each model on the untouched test split at the standard 0.5 threshold.
    5. Saves full JSON report, CSV summary, and confusion matrices JSON.

    Args:
        models_dir: Directory containing serialized .joblib pipelines.
        data_path: Path to raw dataset CSV.
        output_dir: Directory to save evaluation reports.
        threshold: Classification threshold (default 0.5).
        random_state: Random state seed (default 42).
        test_size: Held-out test proportion (default 0.2).

    Returns:
        dict[str, Any]: Results dictionary with metrics, summary table, and report paths.

    Raises:
        FileNotFoundError: If any required model artifact is missing.
    """
    resolved_models_dir = Path(models_dir).resolve() if models_dir else DEFAULT_MODELS_DIR.resolve()
    resolved_data_path = Path(data_path).resolve() if data_path else DEFAULT_DATA_PATH.resolve()

    # Ingest and split dataset
    df = load_raw_data(resolved_data_path)
    validate_input_data(df, require_target=True, target_column=TARGET_COLUMN)

    X, y = separate_features_and_target(
        df,
        target_column=TARGET_COLUMN,
        feature_columns=BASELINE_PREDICTORS,
        exclude_columns=EXCLUDED_COLUMNS,
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    # Check for presence of all four model artifacts
    model_paths: dict[str, Path] = {}
    missing_models: list[str] = []

    for key, filename in BASELINE_MODEL_FILENAMES.items():
        m_path = resolved_models_dir / filename
        if not m_path.is_file():
            missing_models.append(filename)
        else:
            model_paths[key] = m_path

    if missing_models:
        raise FileNotFoundError(
            f"Missing required baseline model artifacts in {resolved_models_dir}: {missing_models}. "
            "Please execute model training first via: python main.py --mode train"
        )

    # Evaluate all models on untouched test split
    evaluation_results = evaluate_all_models(
        models=model_paths,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
    )

    # Compile environment metadata
    metadata: dict[str, Any] = {
        "dataset_path": str(resolved_data_path),
        "target_column": TARGET_COLUMN,
        "permitted_predictors": BASELINE_PREDICTORS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_class_distribution": {int(k): int(v) for k, v in y_test.value_counts().items()},
        "train_class_distribution": {int(k): int(v) for k, v in y_train.value_counts().items()},
        "random_state": random_state,
        "test_size": test_size,
        "classification_threshold": threshold,
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn_version,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }

    # Persist reports
    report_paths = save_evaluation_reports(
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        metadata=metadata,
    )

    summary_df = format_metrics_summary_table(evaluation_results)

    return {
        "evaluation_results": evaluation_results,
        "summary_table": summary_df,
        "report_paths": report_paths,
        "metadata": metadata,
    }
