"""Equipment Health Scoring and Alert Dispatching Module.

This module consumes Phase 5 prediction outputs and evaluates rule-based
maintenance alert levels (NORMAL, WARNING, CRITICAL, INVALID). It enforces
strict configuration validation, score boundary semantics, and generates
actionable, structured maintenance recommendations without modifying models
or input data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src import predict

# Standard Alert Level Identifiers
ALERT_NORMAL: str = "NORMAL"
ALERT_WARNING: str = "WARNING"
ALERT_CRITICAL: str = "CRITICAL"
ALERT_INVALID: str = "INVALID"

VALID_ALERT_LEVELS: set[str] = {
    ALERT_NORMAL,
    ALERT_WARNING,
    ALERT_CRITICAL,
    ALERT_INVALID,
}

# Configurable Illustrative Defaults (Not validated industrial safety limits)
DEFAULT_WARNING_THRESHOLD: float = 0.3
DEFAULT_CRITICAL_THRESHOLD: float = 0.7

# Recommended Action Templates
ACTION_NORMAL: str = (
    "Continue standard operational monitoring schedule. No maintenance intervention required."
)
ACTION_WARNING: str = (
    "Schedule non-urgent technical inspection. Review thermal and mechanical stress indicators."
)
ACTION_CRITICAL: str = (
    "Prioritize maintenance review. Verify sensor readings and operating conditions, and follow the site's approved safety procedures before taking operational action."
)
ACTION_INVALID: str = (
    "Inspect data quality and fix sensor telemetry input formatting before evaluating alerts."
)


def validate_alert_threshold(threshold: float, name: str = "Alert") -> float:
    """Validate that an alert threshold is a finite numeric float strictly within (0.0, 1.0).

    Args:
        threshold: Candidate threshold value.
        name: Threshold label for error messages.

    Returns:
        float: Validated threshold.

    Raises:
        ValueError: If threshold is non-numeric, boolean, non-finite, or outside (0.0, 1.0).
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"{name} threshold must be a numeric float or integer, got {type(threshold).__name__}: {threshold}"
        )

    t_val = float(threshold)
    if not math.isfinite(t_val):
        raise ValueError(f"{name} threshold must be finite, got: {threshold}")

    if not (0.0 < t_val < 1.0):
        raise ValueError(
            f"{name} threshold must be strictly between 0.0 and 1.0 (exclusive). Got: {t_val}"
        )

    return t_val


@dataclass(frozen=True)
class AlertConfig:
    """Configuration for rule-based equipment maintenance alerting.

    Attributes:
        warning_threshold: Failure risk score threshold for WARNING alert level.
        critical_threshold: Failure risk score threshold for CRITICAL alert level.

    Note:
        These thresholds are illustrative engineering defaults for system testing
        and demonstration. They do NOT represent certified industrial safety limits.
    """

    warning_threshold: float = DEFAULT_WARNING_THRESHOLD
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD

    def __post_init__(self) -> None:
        """Validate alert threshold bounds and relative ordering."""
        w_val = validate_alert_threshold(self.warning_threshold, name="Warning")
        c_val = validate_alert_threshold(self.critical_threshold, name="Critical")

        if c_val <= w_val:
            raise ValueError(
                f"Critical threshold ({c_val}) must be strictly greater than "
                f"warning threshold ({w_val})."
            )

        # Re-assign validated float values (object is frozen, use object.__setattr__)
        object.__setattr__(self, "warning_threshold", w_val)
        object.__setattr__(self, "critical_threshold", c_val)


def resolve_alert_config(config: AlertConfig | dict[str, Any] | None = None) -> AlertConfig:
    """Resolve and validate an AlertConfig from various candidate types.

    Args:
        config: AlertConfig instance, dictionary of threshold parameters, or None.

    Returns:
        AlertConfig: Validated immutable alert configuration.
    """
    if config is None:
        return AlertConfig()

    if isinstance(config, AlertConfig):
        return config

    if isinstance(config, dict):
        w = config.get("warning_threshold", DEFAULT_WARNING_THRESHOLD)
        c = config.get("critical_threshold", DEFAULT_CRITICAL_THRESHOLD)
        return AlertConfig(warning_threshold=w, critical_threshold=c)

    raise ValueError(
        f"Expected config to be an AlertConfig, dict, or None, got {type(config).__name__}."
    )


def classify_alert(
    prediction_result: dict[str, Any],
    config: AlertConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a single Phase 5 prediction result into a structured maintenance alert.

    Classification Semantics:
    - positive_class_score < warning_threshold: NORMAL
    - warning_threshold <= positive_class_score < critical_threshold: WARNING
    - positive_class_score >= critical_threshold: CRITICAL
    - Missing, malformed, non-numeric, or non-finite scores, or failed predictions: INVALID

    Boundary Rules:
    - Score exactly equal to warning_threshold -> WARNING
    - Score exactly equal to critical_threshold -> CRITICAL

    Args:
        prediction_result: Output dictionary from src.predict.predict_single or predict_batch.
        config: Optional AlertConfig or dictionary with thresholds.

    Returns:
        dict[str, Any]: Structured alert dictionary.
    """
    cfg = resolve_alert_config(config)

    # Base metadata extraction
    model_name: str | None = None
    record_index: int | None = None
    warnings: list[str] = []

    if isinstance(prediction_result, dict):
        model_name = prediction_result.get("model_name")
        record_index = prediction_result.get("row_index")
        raw_warns = prediction_result.get("warnings")
        if isinstance(raw_warns, list):
            warnings = list(raw_warns)

    thresholds_used = {
        "warning_threshold": cfg.warning_threshold,
        "critical_threshold": cfg.critical_threshold,
    }

    # 1. Guard against non-dict or failed inference status
    if not isinstance(prediction_result, dict):
        return {
            "status": "invalid",
            "alert_level": ALERT_INVALID,
            "reason": f"Prediction result must be a dictionary, got {type(prediction_result).__name__}.",
            "positive_class_score": None,
            "thresholds_used": thresholds_used,
            "recommended_action": ACTION_INVALID,
            "model_name": model_name,
            "record_index": record_index,
            "warnings": warnings,
        }

    if prediction_result.get("status") != "success":
        err_msg = prediction_result.get("error", "Unknown inference error")
        return {
            "status": "invalid",
            "alert_level": ALERT_INVALID,
            "reason": f"Prediction status indicates inference error: {err_msg}",
            "positive_class_score": None,
            "thresholds_used": thresholds_used,
            "recommended_action": ACTION_INVALID,
            "model_name": model_name,
            "record_index": record_index,
            "warnings": warnings,
        }

    # 2. Extract and validate positive_class_score
    raw_score = prediction_result.get("positive_class_score")

    if raw_score is None or isinstance(raw_score, bool):
        return {
            "status": "invalid",
            "alert_level": ALERT_INVALID,
            "reason": f"Positive-class prediction score is missing or boolean: {raw_score}.",
            "positive_class_score": None,
            "thresholds_used": thresholds_used,
            "recommended_action": ACTION_INVALID,
            "model_name": model_name,
            "record_index": record_index,
            "warnings": warnings,
        }

    try:
        score_val = float(raw_score)
    except (ValueError, TypeError):
        return {
            "status": "invalid",
            "alert_level": ALERT_INVALID,
            "reason": f"Positive-class prediction score is non-numeric: {raw_score}.",
            "positive_class_score": None,
            "thresholds_used": thresholds_used,
            "recommended_action": ACTION_INVALID,
            "model_name": model_name,
            "record_index": record_index,
            "warnings": warnings,
        }

    if not math.isfinite(score_val):
        return {
            "status": "invalid",
            "alert_level": ALERT_INVALID,
            "reason": f"Positive-class prediction score is non-finite (NaN or Infinity): {raw_score}.",
            "positive_class_score": None,
            "thresholds_used": thresholds_used,
            "recommended_action": ACTION_INVALID,
            "model_name": model_name,
            "record_index": record_index,
            "warnings": warnings,
        }

    if not (0.0 <= score_val <= 1.0):
        return {
            "status": "invalid",
            "alert_level": ALERT_INVALID,
            "reason": f"Positive-class prediction score ({score_val}) is outside probability interval [0.0, 1.0].",
            "positive_class_score": score_val,
            "thresholds_used": thresholds_used,
            "recommended_action": ACTION_INVALID,
            "model_name": model_name,
            "record_index": record_index,
            "warnings": warnings,
        }

    # 3. Rule-based alert classification
    if score_val >= cfg.critical_threshold:
        alert_level = ALERT_CRITICAL
        reason = (
            f"Failure risk score ({score_val:.4f}) reached or exceeded critical threshold "
            f"({cfg.critical_threshold:.4f}). Prioritized maintenance review recommended."
        )
        action = ACTION_CRITICAL

    elif score_val >= cfg.warning_threshold:
        alert_level = ALERT_WARNING
        reason = (
            f"Failure risk score ({score_val:.4f}) reached or exceeded warning threshold "
            f"({cfg.warning_threshold:.4f}) but is below critical threshold ({cfg.critical_threshold:.4f}). "
            f"Elevated component degradation risk."
        )
        action = ACTION_WARNING

    else:
        alert_level = ALERT_NORMAL
        reason = (
            f"Failure risk score ({score_val:.4f}) is below warning threshold "
            f"({cfg.warning_threshold:.4f}). Nominal operating condition."
        )
        action = ACTION_NORMAL

    return {
        "status": "success",
        "alert_level": alert_level,
        "reason": reason,
        "positive_class_score": score_val,
        "thresholds_used": thresholds_used,
        "recommended_action": action,
        "model_name": model_name,
        "record_index": record_index,
        "warnings": warnings,
    }


def classify_alert_batch(
    predictions: list[dict[str, Any]],
    config: AlertConfig | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Classify multiple prediction results into structured maintenance alerts.

    Preserves input order, handles each record independently, and prevents
    a single malformed record from interrupting batch evaluation.

    Args:
        predictions: List of prediction result dictionaries.
        config: Optional AlertConfig or threshold dictionary.

    Returns:
        list[dict[str, Any]]: List of alert dictionaries matching input order.

    Raises:
        ValueError: If predictions input is None or not a list.
    """
    if predictions is None:
        raise ValueError("Predictions batch input cannot be None.")

    if not isinstance(predictions, list):
        raise ValueError(
            f"Expected predictions to be a list of dicts, got {type(predictions).__name__}."
        )

    cfg = resolve_alert_config(config)
    alerts: list[dict[str, Any]] = []

    for idx, pred in enumerate(predictions):
        alert_res = classify_alert(pred, config=cfg)
        if alert_res.get("record_index") is None:
            alert_res["record_index"] = idx
        alerts.append(alert_res)

    return alerts


def evaluate_alerts_from_telemetry(
    records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    model: str | Path | Pipeline | None = None,
    config: AlertConfig | dict[str, Any] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Thin orchestration helper: run Phase 5 inference then classify alert levels.

    Reuses existing Phase 5 predict API without duplicating inference or validation logic.

    Args:
        records: Single record (dict/Series) or batch records (list of dicts/DataFrame).
        model: Optional model identifier or pipeline.
        config: Optional AlertConfig or threshold dictionary.

    Returns:
        dict[str, Any] for single record input, or list[dict[str, Any]] for batch input.
    """
    cfg = resolve_alert_config(config)

    if isinstance(records, (list, pd.DataFrame)):
        predictions = predict.predict_batch(
            records=records,
            model=model,
            threshold=cfg.warning_threshold,
            error_policy="per_record",
        )
        return classify_alert_batch(predictions, config=cfg)

    # Single record
    pred = predict.predict_single(
        record=records,
        model=model,
        threshold=cfg.warning_threshold,
    )
    return classify_alert(pred, config=cfg)
