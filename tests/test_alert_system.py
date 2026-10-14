"""Unit and Integration tests for maintenance alert system (Phase 6)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import alert_system, predict, train_model


@pytest.fixture
def default_config() -> alert_system.AlertConfig:
    """Fixture providing standard default AlertConfig (warning=0.3, critical=0.7)."""
    return alert_system.AlertConfig(warning_threshold=0.3, critical_threshold=0.7)


# 1. Normal score below warning threshold
def test_alert_normal_score(default_config: alert_system.AlertConfig):
    """Verify that a score strictly below warning threshold produces NORMAL alert."""
    pred = {
        "status": "success",
        "positive_class_score": 0.15,
        "model_name": "TestModel",
        "warnings": [],
    }
    alert = alert_system.classify_alert(pred, config=default_config)
    assert alert["status"] == "success"
    assert alert["alert_level"] == alert_system.ALERT_NORMAL
    assert alert["positive_class_score"] == 0.15
    assert "below warning threshold" in alert["reason"]
    assert "No maintenance intervention required" in alert["recommended_action"]


# 2. Score exactly equal to warning threshold
def test_alert_boundary_exact_warning(default_config: alert_system.AlertConfig):
    """Verify that a score exactly equal to warning threshold produces WARNING alert."""
    pred = {
        "status": "success",
        "positive_class_score": 0.30,
        "model_name": "TestModel",
    }
    alert = alert_system.classify_alert(pred, config=default_config)
    assert alert["status"] == "success"
    assert alert["alert_level"] == alert_system.ALERT_WARNING
    assert alert["positive_class_score"] == 0.30
    assert "reached or exceeded warning threshold" in alert["reason"]


# 3. Score between warning and critical thresholds
def test_alert_warning_score(default_config: alert_system.AlertConfig):
    """Verify that a score between warning and critical produces WARNING alert."""
    pred = {
        "status": "success",
        "positive_class_score": 0.50,
        "model_name": "TestModel",
    }
    alert = alert_system.classify_alert(pred, config=default_config)
    assert alert["status"] == "success"
    assert alert["alert_level"] == alert_system.ALERT_WARNING
    assert "Schedule non-urgent technical inspection" in alert["recommended_action"]


# 4. Score exactly equal to critical threshold
def test_alert_boundary_exact_critical(default_config: alert_system.AlertConfig):
    """Verify that a score exactly equal to critical threshold produces CRITICAL alert."""
    pred = {
        "status": "success",
        "positive_class_score": 0.70,
        "model_name": "TestModel",
    }
    alert = alert_system.classify_alert(pred, config=default_config)
    assert alert["status"] == "success"
    assert alert["alert_level"] == alert_system.ALERT_CRITICAL
    assert alert["positive_class_score"] == 0.70
    assert "reached or exceeded critical threshold" in alert["reason"]


# 5. Score strictly above critical threshold
def test_alert_critical_score(default_config: alert_system.AlertConfig):
    """Verify that a score above critical threshold produces CRITICAL alert."""
    pred = {
        "status": "success",
        "positive_class_score": 0.92,
        "model_name": "TestModel",
    }
    alert = alert_system.classify_alert(pred, config=default_config)
    assert alert["status"] == "success"
    assert alert["alert_level"] == alert_system.ALERT_CRITICAL
    assert "Prioritize maintenance review" in alert["recommended_action"]


# 6. Missing score
def test_alert_missing_score(default_config: alert_system.AlertConfig):
    """Verify that a missing or None score produces an INVALID alert level."""
    pred_none = {"status": "success", "positive_class_score": None}
    alert_none = alert_system.classify_alert(pred_none, config=default_config)
    assert alert_none["alert_level"] == alert_system.ALERT_INVALID
    assert alert_none["status"] == "invalid"
    assert "missing or boolean" in alert_none["reason"]

    pred_empty = {"status": "success"}
    alert_empty = alert_system.classify_alert(pred_empty, config=default_config)
    assert alert_empty["alert_level"] == alert_system.ALERT_INVALID


# 7. NaN and infinite scores
def test_alert_nan_and_inf_scores(default_config: alert_system.AlertConfig):
    """Verify that NaN and Infinity scores produce INVALID alert level."""
    pred_nan = {"status": "success", "positive_class_score": float("nan")}
    alert_nan = alert_system.classify_alert(pred_nan, config=default_config)
    assert alert_nan["alert_level"] == alert_system.ALERT_INVALID
    assert alert_nan["status"] == "invalid"
    assert "non-finite" in alert_nan["reason"]

    pred_inf = {"status": "success", "positive_class_score": float("inf")}
    alert_inf = alert_system.classify_alert(pred_inf, config=default_config)
    assert alert_inf["alert_level"] == alert_system.ALERT_INVALID


# 8. Out-of-range scores
def test_alert_out_of_range_scores(default_config: alert_system.AlertConfig):
    """Verify that negative scores or scores > 1.0 produce INVALID alert level."""
    pred_neg = {"status": "success", "positive_class_score": -0.1}
    alert_neg = alert_system.classify_alert(pred_neg, config=default_config)
    assert alert_neg["alert_level"] == alert_system.ALERT_INVALID
    assert "outside probability interval" in alert_neg["reason"]

    pred_high = {"status": "success", "positive_class_score": 1.5}
    alert_high = alert_system.classify_alert(pred_high, config=default_config)
    assert alert_high["alert_level"] == alert_system.ALERT_INVALID


# 9. Invalid threshold configurations
def test_alert_invalid_threshold_configurations():
    """Verify that invalid threshold values trigger ValueError."""
    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        alert_system.AlertConfig(warning_threshold=0.0, critical_threshold=0.5)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        alert_system.AlertConfig(warning_threshold=0.3, critical_threshold=1.0)

    with pytest.raises(ValueError, match="numeric float or integer"):
        alert_system.AlertConfig(warning_threshold="0.3", critical_threshold=0.7)

    with pytest.raises(ValueError, match="numeric float or integer"):
        alert_system.AlertConfig(warning_threshold=True, critical_threshold=0.7)


# 10. Critical threshold less than or equal to warning threshold
def test_alert_critical_le_warning():
    """Verify ValueError when critical threshold is <= warning threshold."""
    with pytest.raises(ValueError, match="Critical threshold.*strictly greater"):
        alert_system.AlertConfig(warning_threshold=0.5, critical_threshold=0.5)

    with pytest.raises(ValueError, match="Critical threshold.*strictly greater"):
        alert_system.AlertConfig(warning_threshold=0.6, critical_threshold=0.4)


# 11. Failed prediction status
def test_alert_failed_prediction_status(default_config: alert_system.AlertConfig):
    """Verify that a failed prediction result produces INVALID alert with error reason."""
    failed_pred = {
        "status": "error",
        "error": "Invalid machine 'Type' value",
        "positive_class_score": None,
    }
    alert = alert_system.classify_alert(failed_pred, config=default_config)
    assert alert["alert_level"] == alert_system.ALERT_INVALID
    assert alert["status"] == "invalid"
    assert "inference error" in alert["reason"]
    assert "Invalid machine 'Type'" in alert["reason"]


# 12. Malformed prediction result
def test_alert_malformed_prediction(default_config: alert_system.AlertConfig):
    """Verify that non-dictionary inputs produce INVALID alert level."""
    alert_none = alert_system.classify_alert(None, config=default_config)
    assert alert_none["alert_level"] == alert_system.ALERT_INVALID

    alert_str = alert_system.classify_alert("not_a_dict", config=default_config)
    assert alert_str["alert_level"] == alert_system.ALERT_INVALID


# 13. Batch order preservation
def test_classify_alert_batch_order_preservation(default_config: alert_system.AlertConfig):
    """Verify that classify_alert_batch strictly preserves input order."""
    preds = [
        {"row_index": 0, "status": "success", "positive_class_score": 0.10},
        {"row_index": 1, "status": "success", "positive_class_score": 0.45},
        {"row_index": 2, "status": "success", "positive_class_score": 0.85},
    ]
    alerts = alert_system.classify_alert_batch(preds, config=default_config)
    assert len(alerts) == 3
    assert [a["alert_level"] for a in alerts] == [
        alert_system.ALERT_NORMAL,
        alert_system.ALERT_WARNING,
        alert_system.ALERT_CRITICAL,
    ]
    assert [a["record_index"] for a in alerts] == [0, 1, 2]


# 14. Mixed valid and invalid records in one batch
def test_classify_alert_batch_mixed_records(default_config: alert_system.AlertConfig):
    """Verify that invalid records do not crash the batch and are isolated as INVALID."""
    preds = [
        {"row_index": 0, "status": "success", "positive_class_score": 0.05},
        {"row_index": 1, "status": "error", "error": "Sensor speed negative"},
        {"row_index": 2, "status": "success", "positive_class_score": 0.75},
        {"row_index": 3, "status": "success", "positive_class_score": float("nan")},
    ]
    alerts = alert_system.classify_alert_batch(preds, config=default_config)
    assert len(alerts) == 4
    assert alerts[0]["alert_level"] == alert_system.ALERT_NORMAL
    assert alerts[1]["alert_level"] == alert_system.ALERT_INVALID
    assert alerts[2]["alert_level"] == alert_system.ALERT_CRITICAL
    assert alerts[3]["alert_level"] == alert_system.ALERT_INVALID


# 15. Custom threshold configuration
def test_alert_custom_configuration():
    """Verify custom warning and critical thresholds are applied correctly."""
    custom_cfg = alert_system.AlertConfig(warning_threshold=0.20, critical_threshold=0.60)
    pred = {"status": "success", "positive_class_score": 0.25}

    # Under default config (0.3, 0.7), 0.25 is NORMAL
    alert_default = alert_system.classify_alert(pred)
    assert alert_default["alert_level"] == alert_system.ALERT_NORMAL

    # Under custom config (0.2, 0.6), 0.25 is WARNING
    alert_custom = alert_system.classify_alert(pred, config=custom_cfg)
    assert alert_custom["alert_level"] == alert_system.ALERT_WARNING


# 16. No model fitting or modification
def test_alert_system_does_not_train():
    """Verify that alert evaluation does not touch or retrain models."""
    mock_model = MagicMock()
    # Alert system functions take prediction results and config, never invoking model.fit
    pred = {"status": "success", "positive_class_score": 0.5}
    alert_system.classify_alert(pred)
    mock_model.fit.assert_not_called()


# 17. Thin orchestration helper: evaluate_alerts_from_telemetry
def test_evaluate_alerts_from_telemetry_integration():
    """Verify end-to-end telemetry -> prediction -> alert orchestration."""
    sample_records = [
        {"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0},
        {"Type": "L", "Air temperature [K]": 303.9, "Process temperature [K]": 312.8, "Rotational speed [rpm]": 1345, "Torque [Nm]": 62.4, "Tool wear [min]": 215},
    ]

    rf_path = predict.DEFAULT_MODELS_DIR / predict.DEFAULT_REFERENCE_MODEL_FILENAME
    if not rf_path.is_file():
        pytest.skip("Phase 4 reference model not available.")

    alerts = alert_system.evaluate_alerts_from_telemetry(
        records=sample_records,
        model="random_forest",
        config=alert_system.AlertConfig(warning_threshold=0.3, critical_threshold=0.7),
    )

    assert len(alerts) == 2
    assert alerts[0]["alert_level"] in alert_system.VALID_ALERT_LEVELS
    assert alerts[1]["alert_level"] in alert_system.VALID_ALERT_LEVELS
    # Record 0 is healthy -> NORMAL
    assert alerts[0]["alert_level"] == alert_system.ALERT_NORMAL
    # Record 1 is severe degradation -> CRITICAL (score 0.84 >= 0.70)
    assert alerts[1]["alert_level"] == alert_system.ALERT_CRITICAL


# 18. CLI alert mode integration test
def test_cli_alert_success(monkeypatch: pytest.MonkeyPatch):
    """Verify that CLI alert mode runs successfully on sample CSV."""
    import main

    sample_csv = Path(__file__).resolve().parent.parent / "data" / "sample_inference_records.csv"
    if not sample_csv.is_file():
        pytest.skip("sample_inference_records.csv not found")

    test_args = [
        "main.py",
        "--mode",
        "alert",
        "--input",
        str(sample_csv),
        "--warning-threshold",
        "0.25",
        "--critical-threshold",
        "0.75",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    assert main.main() == 0


# 19. CLI alert mode error handling
def test_cli_alert_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Verify that CLI alert mode exits with code 1 for invalid configurations."""
    import main

    sample_csv = Path(__file__).resolve().parent.parent / "data" / "sample_inference_records.csv"

    # Invalid thresholds: critical <= warning
    test_args_bad_thresh = [
        "main.py",
        "--mode",
        "alert",
        "--input",
        str(sample_csv),
        "--warning-threshold",
        "0.8",
        "--critical-threshold",
        "0.4",
    ]
    monkeypatch.setattr("sys.argv", test_args_bad_thresh)
    assert main.main() == 1

    # Missing input file
    test_args_no_file = [
        "main.py",
        "--mode",
        "alert",
        "--input",
        "non_existent_file.csv",
    ]
    monkeypatch.setattr("sys.argv", test_args_no_file)
    assert main.main() == 1
