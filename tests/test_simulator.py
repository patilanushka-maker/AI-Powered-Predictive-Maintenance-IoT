"""Unit and Integration tests for IoT Telemetry Simulator (Phase 7)."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import alert_system, predict, simulator
from src.preprocessing import BASELINE_PREDICTORS


# 1. Deterministic output with the same seed
def test_simulation_reproducibility_with_same_seed():
    """Verify that identical seed and config reproduce identical telemetry values."""
    records_1 = simulator.generate_telemetry_records(
        num_records=10,
        num_machines=2,
        seed=42,
    )
    records_2 = simulator.generate_telemetry_records(
        num_records=10,
        num_machines=2,
        seed=42,
    )

    assert len(records_1) == 10
    assert len(records_2) == 10
    assert records_1 == records_2


# 2. Different telemetry when the seed changes
def test_simulation_different_telemetry_with_different_seed():
    """Verify that changing the random seed generates different telemetry values."""
    records_a = simulator.generate_telemetry_records(num_records=5, seed=42)
    records_b = simulator.generate_telemetry_records(num_records=5, seed=999)

    assert len(records_a) == len(records_b) == 5
    # Numerical readings should differ between runs
    speeds_a = [r["Rotational speed [rpm]"] for r in records_a]
    speeds_b = [r["Rotational speed [rpm]"] for r in records_b]
    assert speeds_a != speeds_b


# 3. Valid feature names and data types
def test_simulation_valid_feature_names_and_types():
    """Verify generated records contain strictly valid baseline predictors and types."""
    records = simulator.generate_telemetry_records(num_records=15, num_machines=3, seed=123)

    for rec in records:
        for predictor in BASELINE_PREDICTORS:
            assert predictor in rec, f"Missing predictor '{predictor}'"

        assert rec["Type"] in {"L", "M", "H"}
        assert isinstance(rec["Air temperature [K]"], float)
        assert isinstance(rec["Process temperature [K]"], float)
        assert isinstance(rec["Rotational speed [rpm]"], float)
        assert isinstance(rec["Torque [Nm]"], float)
        assert isinstance(rec["Tool wear [min]"], float)

        # Physical domain boundary checks
        assert rec["Air temperature [K]"] > 0
        assert rec["Process temperature [K]"] > rec["Air temperature [K]"]
        assert rec["Rotational speed [rpm]"] > 0
        assert rec["Torque [Nm]"] > 0
        assert rec["Tool wear [min]"] >= 0


# 4. Correct record counts
def test_simulation_record_counts():
    """Verify simulator generates exactly the requested number of records, including empty."""
    # Empty simulation
    empty_records = simulator.generate_telemetry_records(num_records=0)
    assert empty_records == []

    # Single machine
    records_5 = simulator.generate_telemetry_records(num_records=5, num_machines=1)
    assert len(records_5) == 5

    # Multi-machine round-robin
    records_12 = simulator.generate_telemetry_records(num_records=12, num_machines=4)
    assert len(records_12) == 12


# 5. Correct machine identifiers and timestamps
def test_simulation_machine_identifiers_and_timestamps():
    """Verify machine IDs cycle correctly and timestamps advance by interval."""
    interval = 2.5
    records = simulator.generate_telemetry_records(
        num_records=6,
        num_machines=2,
        interval_seconds=interval,
        start_time="2026-06-01T12:00:00Z",
    )

    expected_ids = ["M-001", "M-002", "M-001", "M-002", "M-001", "M-002"]
    actual_ids = [r["machine_id"] for r in records]
    assert actual_ids == expected_ids

    # For M-001: records at index 0, 2, 4
    dt0 = datetime.fromisoformat(records[0]["timestamp"].replace("Z", "+00:00"))
    dt2 = datetime.fromisoformat(records[2]["timestamp"].replace("Z", "+00:00"))
    dt4 = datetime.fromisoformat(records[4]["timestamp"].replace("Z", "+00:00"))

    assert (dt2 - dt0).total_seconds() == interval
    assert (dt4 - dt2).total_seconds() == interval


# 6. Empty and invalid configurations
def test_simulation_empty_and_invalid_configurations():
    """Verify that invalid configuration values raise explicit ValueErrors."""
    # Negative records
    with pytest.raises(ValueError, match="num_records must be non-negative"):
        simulator.SimulationConfig(num_records=-1)

    # Zero or negative machines
    with pytest.raises(ValueError, match="num_machines must be a positive integer"):
        simulator.SimulationConfig(num_machines=0)

    # Zero or negative interval
    with pytest.raises(ValueError, match="interval_seconds must be a finite positive number"):
        simulator.SimulationConfig(interval_seconds=0.0)

    with pytest.raises(ValueError, match="interval_seconds must be a finite positive number"):
        simulator.SimulationConfig(interval_seconds=float("nan"))

    # Invalid scenario
    with pytest.raises(ValueError, match="Invalid simulation scenario"):
        simulator.SimulationConfig(scenario="catastrophic_meltdown")

    # Invalid seed type
    with pytest.raises(ValueError, match="seed must be an integer or None"):
        simulator.SimulationConfig(seed="not_a_seed")  # type: ignore

    # Invalid timestamp format
    with pytest.raises(ValueError, match="Invalid start_time format"):
        simulator.SimulationConfig(start_time="not-a-datetime")


# 7. Nominal and degradation scenario generation
def test_simulation_nominal_and_degradation_scenarios():
    """Verify differences between nominal and progressive degradation telemetry."""
    nominal_records = simulator.generate_telemetry_records(
        num_records=10,
        scenario="nominal",
        seed=42,
    )
    degraded_records = simulator.generate_telemetry_records(
        num_records=10,
        scenario="degradation",
        seed=42,
    )

    # In degradation scenario, tool wear and torque should be significantly elevated
    avg_nom_wear = sum(r["Tool wear [min]"] for r in nominal_records) / 10
    avg_deg_wear = sum(r["Tool wear [min]"] for r in degraded_records) / 10
    assert avg_deg_wear > avg_nom_wear + 100.0

    avg_nom_torque = sum(r["Torque [Nm]"] for r in nominal_records) / 10
    avg_deg_torque = sum(r["Torque [Nm]"] for r in degraded_records) / 10
    assert avg_deg_torque > avg_nom_torque + 10.0


# 8. Successful integration with mocked prediction and alert functions
def test_simulation_integration_with_mocked_functions():
    """Verify run_simulation orchestrates injected mock prediction and alert functions."""
    mock_predict = MagicMock(
        return_value={
            "model_name": "MockClassifier",
            "predicted_class": 1,
            "predicted_label": "Failure label predicted",
            "positive_class_score": 0.85,
            "threshold": 0.3,
            "status": "success",
            "warnings": ["Test advisory warning"],
        }
    )
    mock_alert = MagicMock(
        return_value={
            "status": "success",
            "alert_level": alert_system.ALERT_CRITICAL,
            "reason": "Mock critical alert triggered",
            "positive_class_score": 0.85,
            "recommended_action": alert_system.ACTION_CRITICAL,
            "model_name": "MockClassifier",
            "warnings": [],
        }
    )

    events = simulator.run_simulation(
        num_records=3,
        seed=42,
        predict_fn=mock_predict,
        alert_fn=mock_alert,
    )

    assert len(events) == 3
    assert mock_predict.call_count == 3
    assert mock_alert.call_count == 3

    for evt in events:
        assert evt["status"] == "success"
        assert evt["alert_level"] == alert_system.ALERT_CRITICAL
        assert evt["positive_class_score"] == 0.85
        assert evt["model_name"] == "MockClassifier"
        assert "Test advisory warning" in evt["warnings"]


# 9. Invalid record and inference error handling
def test_simulation_fault_injection_and_error_handling():
    """Verify that fault injection produces INVALID alert levels without crashing batch."""
    events = simulator.run_simulation(
        num_records=4,
        fault_injection=True,
        seed=42,
        predict_fn=lambda r: {
            "model_name": "Mock",
            "predicted_class": 0,
            "predicted_label": "No failure",
            "positive_class_score": 0.1,
            "status": "success",
            "warnings": [],
        },
        alert_fn=lambda p: {
            "status": "success",
            "alert_level": alert_system.ALERT_NORMAL,
            "positive_class_score": 0.1,
            "reason": "Nominal",
            "recommended_action": alert_system.ACTION_NORMAL,
            "warnings": [],
        },
    )

    assert len(events) == 4
    # Event 1 (target of fault injection) should be INVALID due to negative speed
    invalid_event = events[1]
    assert invalid_event["status"] == "invalid"
    assert invalid_event["alert_level"] == alert_system.ALERT_INVALID
    assert invalid_event["positive_class_score"] is None
    assert "validation failed" in invalid_event["reason"].lower()

    # Other events should be successful and NORMAL
    assert events[0]["status"] == "success"
    assert events[0]["alert_level"] == alert_system.ALERT_NORMAL
    assert events[2]["status"] == "success"


def test_simulation_handles_inference_exception():
    """Verify that an unexpected exception inside predict_fn fails safely as an INVALID alert."""
    def exploding_predict(rec):
        raise RuntimeError("Inference pipeline failure!")

    events = simulator.run_simulation(
        num_records=2,
        seed=42,
        predict_fn=exploding_predict,
    )

    assert len(events) == 2
    for evt in events:
        assert evt["status"] == "invalid"
        assert evt["alert_level"] == alert_system.ALERT_INVALID
        assert "Inference execution failed" in evt["reason"]


# 10. Event schema consistency
def test_simulation_event_schema_consistency():
    """Verify that every event dictionary strictly includes all required schema fields."""
    expected_fields = {
        "event_id",
        "timestamp",
        "machine_id",
        "scenario",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "model_name",
        "predicted_class",
        "predicted_label",
        "positive_class_score",
        "alert_level",
        "status",
        "reason",
        "recommended_action",
        "warnings",
    }

    events = simulator.run_simulation(
        num_records=3,
        scenario="nominal",
        seed=42,
        predict_fn=lambda r: {
            "model_name": "TestM",
            "predicted_class": 0,
            "predicted_label": "No failure",
            "positive_class_score": 0.05,
            "status": "success",
            "warnings": [],
        },
        alert_fn=lambda p: {
            "status": "success",
            "alert_level": alert_system.ALERT_NORMAL,
            "positive_class_score": 0.05,
            "reason": "OK",
            "recommended_action": alert_system.ACTION_NORMAL,
            "warnings": [],
        },
    )

    for evt in events:
        assert set(evt.keys()) == expected_fields


# 11. JSON and CSV output persistence
def test_simulation_json_and_csv_persistence(tmp_path: Path):
    """Verify saving simulation events to JSON and CSV files."""
    json_path = tmp_path / "sim_events.json"
    csv_path = tmp_path / "sim_events.csv"

    # Save to JSON
    events_json = simulator.run_simulation(
        num_records=5,
        output_path=json_path,
        seed=42,
        predict_fn=lambda r: {
            "status": "success",
            "positive_class_score": 0.1,
            "predicted_class": 0,
            "predicted_label": "No failure",
            "warnings": [],
        },
    )
    assert json_path.is_file()
    loaded_df = pd.read_json(json_path)
    assert len(loaded_df) == 5

    # Save to CSV
    events_csv = simulator.run_simulation(
        num_records=5,
        output_path=csv_path,
        seed=42,
        predict_fn=lambda r: {
            "status": "success",
            "positive_class_score": 0.1,
            "predicted_class": 0,
            "predicted_label": "No failure",
            "warnings": [],
        },
    )
    assert csv_path.is_file()
    loaded_csv = pd.read_csv(csv_path)
    assert len(loaded_csv) == 5
    assert "event_id" in loaded_csv.columns
    assert "alert_level" in loaded_csv.columns

    # Unsupported format
    with pytest.raises(ValueError, match="Unsupported simulation output format"):
        simulator.run_simulation(num_records=1, output_path=tmp_path / "out.parquet")


# 12. No model training during simulation
def test_simulation_does_not_train_models(monkeypatch: pytest.MonkeyPatch):
    """Verify that simulation never invokes fit or fit_transform on pipelines."""
    fit_mock = MagicMock(side_effect=AssertionError("Pipeline.fit must NOT be called!"))
    monkeypatch.setattr(Pipeline, "fit", fit_mock)

    events = simulator.run_simulation(
        num_records=3,
        seed=42,
        model="random_forest",
    )
    assert len(events) == 3
    fit_mock.assert_not_called()


# 13. Real model integration end-to-end
def test_simulation_real_random_forest_model_end_to_end():
    """Verify end-to-end simulation using real persisted random_forest baseline pipeline."""
    events = simulator.run_simulation(
        num_records=4,
        num_machines=2,
        scenario="nominal",
        seed=42,
        model="random_forest",
    )

    assert len(events) == 4
    for evt in events:
        assert evt["status"] == "success"
        assert evt["alert_level"] in {alert_system.ALERT_NORMAL, alert_system.ALERT_WARNING}
        assert 0.0 <= evt["positive_class_score"] <= 1.0
        assert evt["predicted_class"] in {0, 1}
        assert evt["model_name"] == "RandomForestClassifier"


# 14. CLI integration: simulate mode success and error handling
def test_cli_simulate_success():
    """Verify main.py CLI runs --mode simulate successfully."""
    from main import main
    import sys

    test_args = [
        "main.py",
        "--mode",
        "simulate",
        "--records",
        "5",
        "--machines",
        "1",
        "--seed",
        "42",
        "--scenario",
        "nominal",
    ]
    orig_argv = sys.argv
    sys.argv = test_args
    try:
        exit_code = main()
        assert exit_code == 0
    finally:
        sys.argv = orig_argv


def test_cli_simulate_error_handling():
    """Verify main.py CLI fails cleanly on invalid alert thresholds in simulate mode."""
    from main import main
    import sys

    # Invalid thresholds: critical <= warning
    test_args = [
        "main.py",
        "--mode",
        "simulate",
        "--warning-threshold",
        "0.8",
        "--critical-threshold",
        "0.2",
    ]
    orig_argv = sys.argv
    sys.argv = test_args
    try:
        exit_code = main()
        assert exit_code == 1
    finally:
        sys.argv = orig_argv
