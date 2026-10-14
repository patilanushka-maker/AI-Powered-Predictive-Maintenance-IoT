"""Unit and Integration tests for baseline model evaluation module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import evaluate_model, train_model
from src.preprocessing import BASELINE_PREDICTORS, TARGET_COLUMN


@pytest.fixture
def synthetic_eval_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic test predictors and target labels."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "Type": ["L"] * 18 + ["M"] * 9 + ["H"] * 3,
        "Air temperature [K]": np.linspace(298.0, 302.0, n),
        "Process temperature [K]": np.linspace(308.0, 312.0, n),
        "Rotational speed [rpm]": np.linspace(1400, 1600, n),
        "Torque [Nm]": np.linspace(35.0, 45.0, n),
        "Tool wear [min]": list(range(0, n * 5, 5)),
    })
    y = pd.Series([0] * 24 + [1] * 6, name=TARGET_COLUMN)
    return df, y


def test_compute_classification_metrics_known_example():
    """Verify metrics computation against a known hand-calculated example."""
    # Ground truth: 3 negatives (0), 2 positives (1)
    y_true = np.array([0, 0, 0, 1, 1])
    # Predictions: 1 true negative, 2 false positives, 1 false negative, 1 true positive
    y_pred = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.7, 0.4, 0.9])

    metrics = evaluate_model.compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=0.5,
    )

    # Hand-calculated components:
    # TN: y_true=0, y_pred=0 -> index 0 (1 count)
    # FP: y_true=0, y_pred=1 -> indices 1, 2 (2 counts)
    # FN: y_true=1, y_pred=0 -> index 3 (1 count)
    # TP: y_true=1, y_pred=1 -> index 4 (1 count)
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1

    # Precision = TP / (TP + FP) = 1 / (1 + 2) = 1/3 ~ 0.333333
    assert pytest.approx(metrics["precision"], abs=1e-4) == 1 / 3

    # Recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert metrics["recall"] == 0.5

    # F1 = 2 * (1/3 * 0.5) / (1/3 + 0.5) = (1/3) / (5/6) = 2/5 = 0.4
    assert metrics["f1_score"] == 0.4

    # Accuracy = (1 + 1) / 5 = 0.4
    assert metrics["accuracy"] == 0.4

    # Confusion matrix structure: [[TN, FP], [FN, TP]]
    assert metrics["confusion_matrix"] == [[1, 2], [1, 1]]


def test_compute_classification_metrics_dimension_mismatch():
    """Verify ValueError when array lengths do not match."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])
    y_prob = np.array([0.2, 0.8, 0.1])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        evaluate_model.compute_classification_metrics(y_true, y_pred, y_prob)


def test_compute_classification_metrics_empty():
    """Verify ValueError on empty array inputs."""
    with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
        evaluate_model.compute_classification_metrics(np.array([]), np.array([]), np.array([]))


def test_evaluate_pipeline_single(synthetic_eval_data: tuple[pd.DataFrame, pd.Series]):
    """Verify evaluation of a trained pipeline on test data."""
    X, y = synthetic_eval_data
    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = train_model.train_single_pipeline(clf, X, y)

    metrics = evaluate_model.evaluate_pipeline(pipeline, X, y, threshold=0.5)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["true_positives"] + metrics["false_negatives"] == 6  # Total positive cases
    assert metrics["true_negatives"] + metrics["false_positives"] == 24  # Total negative cases


def test_evaluate_pipeline_length_mismatch(synthetic_eval_data: tuple[pd.DataFrame, pd.Series]):
    """Verify ValueError when X_test and y_test row counts differ."""
    X, y = synthetic_eval_data
    clf = LogisticRegression()
    pipeline = train_model.build_model_pipeline(clf)

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_model.evaluate_pipeline(pipeline, X.iloc[:-2], y)


def test_save_evaluation_reports(tmp_path: Path):
    """Verify that reports are persisted as valid JSON, CSV, and confusion matrix files."""
    sample_eval_results = {
        "model_a": {
            "model_name": "Model Alpha",
            "threshold": 0.5,
            "precision": 0.85,
            "recall": 0.70,
            "f1_score": 0.77,
            "roc_auc": 0.92,
            "average_precision": 0.80,
            "accuracy": 0.95,
            "true_positives": 7,
            "false_positives": 1,
            "true_negatives": 19,
            "false_negatives": 3,
            "confusion_matrix": [[19, 1], [3, 7]],
        }
    }

    report_paths = evaluate_model.save_evaluation_reports(
        evaluation_results=sample_eval_results,
        output_dir=tmp_path,
        metadata={"test_run": "unit_test"},
    )

    assert report_paths["json_report"].is_file()
    assert report_paths["csv_summary"].is_file()
    assert report_paths["confusion_matrices"].is_file()

    # Verify JSON content is valid and contains no non-finite values
    with open(report_paths["json_report"], "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    assert json_data["models"]["model_a"]["f1_score"] == 0.77
    assert json_data["metadata"]["test_run"] == "unit_test"

    # Verify CSV summary content
    df_summary = pd.read_csv(report_paths["csv_summary"])
    assert len(df_summary) == 1
    assert df_summary.loc[0, "model_key"] == "model_a"
    assert df_summary.loc[0, "f1_score"] == 0.77


def test_training_validation_threshold_analysis(synthetic_eval_data: tuple[pd.DataFrame, pd.Series]):
    """Verify training-only cross-validation threshold sweep without touching test data."""
    X, y = synthetic_eval_data
    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = train_model.build_model_pipeline(clf)

    threshold_analysis = evaluate_model.analyze_training_validation_thresholds(
        pipeline=pipeline,
        X_train=X,
        y_train=y,
        cv=3,
        candidate_thresholds=[0.3, 0.5, 0.7],
    )

    assert threshold_analysis["analysis_type"] == "training_only_stratified_cv_threshold_sweep"
    assert threshold_analysis["sample_count"] == len(X)
    assert len(threshold_analysis["threshold_evaluations"]) == 3
    assert 0.0 <= threshold_analysis["best_f1_threshold"] <= 1.0


def test_run_baseline_evaluation_missing_models(tmp_path: Path):
    """Verify FileNotFoundError when required model artifacts are missing."""
    empty_models_dir = tmp_path / "empty_models"
    empty_models_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing required baseline model artifacts"):
        evaluate_model.run_baseline_evaluation(models_dir=empty_models_dir)
