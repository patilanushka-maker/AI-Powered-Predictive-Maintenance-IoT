"""Unit and integration tests for dashboard helper functions and Streamlit app (Phase 8)."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src import alert_system, dashboard_helpers, predict
from src.preprocessing import BASELINE_PREDICTORS


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Fixture providing a realistic sample simulation events DataFrame."""
    events = [
        {
            "event_id": "EVT-00001",
            "timestamp": "2026-01-01T00:00:00Z",
            "machine_id": "M-001",
            "scenario": "nominal",
            "Type": "L",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1550.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
            "model_name": "RandomForestClassifier",
            "predicted_class": 0,
            "predicted_label": "No failure label predicted",
            "positive_class_score": 0.05,
            "alert_level": alert_system.ALERT_NORMAL,
            "status": "success",
            "reason": "Nominal",
            "recommended_action": alert_system.ACTION_NORMAL,
            "warnings": [],
        },
        {
            "event_id": "EVT-00002",
            "timestamp": "2026-01-01T00:00:01Z",
            "machine_id": "M-002",
            "scenario": "degradation",
            "Type": "M",
            "Air temperature [K]": 300.2,
            "Process temperature [K]": 311.5,
            "Rotational speed [rpm]": 1420.0,
            "Torque [Nm]": 55.0,
            "Tool wear [min]": 180.0,
            "model_name": "RandomForestClassifier",
            "predicted_class": 1,
            "predicted_label": "Failure label predicted",
            "positive_class_score": 0.55,
            "alert_level": alert_system.ALERT_WARNING,
            "status": "success",
            "reason": "Warning risk score",
            "recommended_action": alert_system.ACTION_WARNING,
            "warnings": [],
        },
        {
            "event_id": "EVT-00003",
            "timestamp": "2026-01-01T00:00:02Z",
            "machine_id": "M-001",
            "scenario": "degradation",
            "Type": "L",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 315.0,
            "Rotational speed [rpm]": 1300.0,
            "Torque [Nm]": 68.0,
            "Tool wear [min]": 220.0,
            "model_name": "RandomForestClassifier",
            "predicted_class": 1,
            "predicted_label": "Failure label predicted",
            "positive_class_score": 0.88,
            "alert_level": alert_system.ALERT_CRITICAL,
            "status": "success",
            "reason": "Critical risk score",
            "recommended_action": alert_system.ACTION_CRITICAL,
            "warnings": [],
        },
        {
            "event_id": "EVT-00004",
            "timestamp": "2026-01-01T00:00:03Z",
            "machine_id": "M-002",
            "scenario": "fault_injection",
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": -50.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 15.0,
            "model_name": "RandomForestClassifier",
            "predicted_class": None,
            "predicted_label": None,
            "positive_class_score": None,
            "alert_level": alert_system.ALERT_INVALID,
            "status": "invalid",
            "reason": "Telemetry validation failed: negative speed",
            "recommended_action": alert_system.ACTION_INVALID,
            "warnings": ["Negative speed"],
        },
    ]
    return pd.DataFrame(events)


# 1. KPI Aggregation Tests
def test_compute_kpi_summary_valid(sample_events_df: pd.DataFrame):
    """Verify KPI aggregation computes correct counts across all alert categories."""
    kpi = dashboard_helpers.compute_kpi_summary(sample_events_df)
    assert kpi["total_events"] == 4
    assert kpi["normal_count"] == 1
    assert kpi["warning_count"] == 1
    assert kpi["critical_count"] == 1
    assert kpi["invalid_count"] == 1
    assert kpi["distinct_machines"] == 2


def test_compute_kpi_summary_empty():
    """Verify KPI aggregation handles empty and None inputs safely without errors."""
    kpi_empty = dashboard_helpers.compute_kpi_summary(pd.DataFrame())
    assert kpi_empty["total_events"] == 0
    assert kpi_empty["normal_count"] == 0
    assert kpi_empty["distinct_machines"] == 0

    kpi_none = dashboard_helpers.compute_kpi_summary(None)
    assert kpi_none["total_events"] == 0


def test_compute_kpi_summary_missing_columns():
    """Verify KPI aggregation functions when alert_level column is missing."""
    df_missing = pd.DataFrame([{"machine_id": "M-001", "other": 123}])
    kpi = dashboard_helpers.compute_kpi_summary(df_missing)
    assert kpi["total_events"] == 1
    assert kpi["normal_count"] == 0
    assert kpi["distinct_machines"] == 1


# 2. Event Filtering Tests
def test_filter_events_by_machine(sample_events_df: pd.DataFrame):
    """Verify filtering simulation events by selected machine IDs."""
    filtered_m1 = dashboard_helpers.filter_events(sample_events_df, machines=["M-001"])
    assert len(filtered_m1) == 2
    assert set(filtered_m1["machine_id"].unique()) == {"M-001"}

    filtered_m2 = dashboard_helpers.filter_events(sample_events_df, machines=["M-002"])
    assert len(filtered_m2) == 2
    assert set(filtered_m2["machine_id"].unique()) == {"M-002"}


def test_filter_events_by_severity(sample_events_df: pd.DataFrame):
    """Verify filtering events by single and multiple alert severity levels."""
    filtered_crit = dashboard_helpers.filter_events(
        sample_events_df, severities=[alert_system.ALERT_CRITICAL]
    )
    assert len(filtered_crit) == 1
    assert filtered_crit.iloc[0]["alert_level"] == alert_system.ALERT_CRITICAL

    filtered_multi = dashboard_helpers.filter_events(
        sample_events_df,
        severities=[alert_system.ALERT_WARNING, alert_system.ALERT_CRITICAL],
    )
    assert len(filtered_multi) == 2


def test_filter_events_empty_inputs():
    """Verify filter_events handles empty DataFrame and empty filter lists gracefully."""
    empty_res = dashboard_helpers.filter_events(pd.DataFrame())
    assert empty_res.empty

    no_filter = dashboard_helpers.filter_events(pd.DataFrame({"machine_id": ["M-001"]}))
    assert len(no_filter) == 1


# 3. Latest Telemetry Extraction Tests
def test_get_latest_machine_telemetry(sample_events_df: pd.DataFrame):
    """Verify extraction of the most recent telemetry record per machine."""
    latest = dashboard_helpers.get_latest_machine_telemetry(sample_events_df)
    assert len(latest) == 2
    assert set(latest["machine_id"]) == {"M-001", "M-002"}

    # M-001 latest record was EVT-00003 (at 00:00:02Z)
    row_m1 = latest[latest["machine_id"] == "M-001"].iloc[0]
    assert row_m1["event_id"] == "EVT-00003"
    assert row_m1["alert_level"] == alert_system.ALERT_CRITICAL

    # M-002 latest record was EVT-00004 (at 00:00:03Z)
    row_m2 = latest[latest["machine_id"] == "M-002"].iloc[0]
    assert row_m2["event_id"] == "EVT-00004"
    assert row_m2["alert_level"] == alert_system.ALERT_INVALID


def test_get_latest_machine_telemetry_empty():
    """Verify get_latest_machine_telemetry handles empty DataFrames safely."""
    res = dashboard_helpers.get_latest_machine_telemetry(pd.DataFrame())
    assert res.empty


# 4. Time-Series Preparation Tests
def test_prepare_time_series_data_success(sample_events_df: pd.DataFrame):
    """Verify time-series preparation sorts chronologically and filters by machine."""
    ts_df = dashboard_helpers.prepare_time_series_data(sample_events_df, machine_id="M-001")
    assert len(ts_df) == 2
    assert "_datetime" in ts_df.columns
    assert ts_df.iloc[0]["event_id"] == "EVT-00001"
    assert ts_df.iloc[1]["event_id"] == "EVT-00003"


def test_prepare_time_series_data_errors():
    """Verify error conditions in time-series preparation."""
    # Empty DataFrame
    with pytest.raises(ValueError, match="Cannot prepare time-series"):
        dashboard_helpers.prepare_time_series_data(pd.DataFrame())

    # Missing timestamp
    with pytest.raises(ValueError, match="does not contain a 'timestamp' column"):
        dashboard_helpers.prepare_time_series_data(pd.DataFrame([{"speed": 100}]))

    # Unknown machine ID
    with pytest.raises(ValueError, match="No event records found for machine 'M-999'"):
        dashboard_helpers.prepare_time_series_data(
            pd.DataFrame([{"timestamp": "2026-01-01T00:00:00Z", "machine_id": "M-001"}]),
            machine_id="M-999",
        )


# 5. CSV Upload & Validation Tests
def test_validate_and_parse_uploaded_csv_simulation_events():
    """Verify recognition of uploaded valid pre-generated simulation events CSV."""
    cols = ["event_id", "timestamp", "machine_id", "alert_level"] + BASELINE_PREDICTORS
    csv_text = f"{','.join(cols)}\nEVT-1,2026-01-01T00:00:00Z,M-001,NORMAL,L,300.0,310.0,1500.0,40.0,10.0\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "simulation_events"
    assert len(df) == 1
    assert errors == []


def test_validate_and_parse_uploaded_csv_event_missing_predictors():
    """Verify that event CSV missing baseline predictors cannot bypass validation."""
    csv_text = "event_id,timestamp,machine_id,alert_level\nEVT-1,2026-01-01T00:00:00Z,M-001,NORMAL\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "invalid"
    assert any("must include baseline sensor features" in err for err in errors)


def test_validate_and_parse_uploaded_csv_event_invalid_alert_level():
    """Verify that event CSV containing invalid alert levels is rejected."""
    cols = ["event_id", "timestamp", "machine_id", "alert_level"] + BASELINE_PREDICTORS
    csv_text = f"{','.join(cols)}\nEVT-1,2026-01-01T00:00:00Z,M-001,HACKED_ALERT,L,300.0,310.0,1500.0,40.0,10.0\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "invalid"
    assert any("Unrecognized alert level" in err for err in errors)


def test_validate_and_parse_uploaded_csv_duplicate_columns():
    """Verify rejection of uploaded CSV with duplicate column headers."""
    cols = ["Type", "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    csv_text = f"{','.join(cols)}\nL,L,300.0,310.0,1500.0,40.0,10.0\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "invalid"
    assert any("Duplicate column names detected" in err for err in errors)


def test_validate_and_parse_uploaded_csv_unexpected_columns():
    """Verify rejection when unexpected extra feature columns are passed in raw telemetry."""
    cols = BASELINE_PREDICTORS + ["unapproved_sensor_voltage", "extra_noise"]
    csv_text = f"{','.join(cols)}\nL,300.0,310.0,1500.0,40.0,15.0,12.5,0.01\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "invalid"
    assert any("Unexpected feature columns detected" in err for err in errors)


def test_validate_and_parse_uploaded_csv_raw_telemetry():
    """Verify recognition and validation of uploaded raw sensor telemetry CSV."""
    cols = ",".join(BASELINE_PREDICTORS)
    csv_text = f"{cols}\nL,300.0,310.0,1500.0,40.0,15.0\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "raw_telemetry"
    assert len(df) == 1
    assert errors == []


def test_validate_and_parse_uploaded_csv_forbidden_columns():
    """Verify rejection of CSV containing target labels or post-failure indicators."""
    cols = ",".join(BASELINE_PREDICTORS) + ",Machine failure,TWF"
    csv_text = f"{cols}\nL,300.0,310.0,1500.0,40.0,15.0,0,0\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "invalid"
    assert any("Target leakage violation" in err for err in errors)


def test_validate_and_parse_uploaded_csv_missing_predictors():
    """Verify rejection when baseline predictors are missing from raw telemetry input."""
    csv_text = "Type,Air temperature [K],Rotational speed [rpm]\nL,300.0,1500.0\n"
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(csv_text))
    assert schema_type == "invalid"
    assert any("Missing required baseline predictors" in err for err in errors)


def test_validate_and_parse_uploaded_csv_empty():
    """Verify handling of completely empty CSV file."""
    schema_type, df, errors = dashboard_helpers.validate_and_parse_uploaded_csv(io.StringIO(""))
    assert schema_type == "invalid"
    assert any("zero records" in err.lower() or "failed to parse" in err.lower() for err in errors)


# 6. Convert Raw Telemetry to Events Tests
def test_convert_raw_telemetry_to_events_mocked(monkeypatch: pytest.MonkeyPatch):
    """Verify conversion of raw telemetry into structured events using mocked predict."""
    df_raw = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1550.0,
                "Torque [Nm]": 42.0,
                "Tool wear [min]": 18.0,
            }
        ]
    )

    mock_pipeline = MagicMock()
    monkeypatch.setattr(
        predict,
        "load_inference_pipeline",
        lambda model=None: (mock_pipeline, "MockModel"),
    )
    monkeypatch.setattr(
        predict,
        "predict_single",
        lambda val_rec, model=None, threshold=0.5: {
            "model_name": "MockModel",
            "predicted_class": 0,
            "predicted_label": "No failure label predicted",
            "positive_class_score": 0.12,
            "status": "success",
            "warnings": [],
        },
    )

    events = dashboard_helpers.convert_raw_telemetry_to_events(df_raw)
    assert len(events) == 1
    evt = events[0]
    assert evt["status"] == "success"
    assert evt["alert_level"] == alert_system.ALERT_NORMAL
    assert evt["positive_class_score"] == 0.12
    assert evt["model_name"] == "MockModel"
    assert evt["Type"] == "L"


def test_convert_raw_telemetry_invalid_record():
    """Verify invalid raw telemetry record fails safely into INVALID alert."""
    df_invalid = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": -99.0,  # Negative speed violation
                "Torque [Nm]": 42.0,
                "Tool wear [min]": 18.0,
            }
        ]
    )

    events = dashboard_helpers.convert_raw_telemetry_to_events(
        df_invalid,
        model="random_forest",
    )
    assert len(events) == 1
    evt = events[0]
    assert evt["status"] == "invalid"
    assert evt["alert_level"] == alert_system.ALERT_INVALID
    assert evt["positive_class_score"] is None
    assert "Validation failed" in evt["reason"]


# 7. Time-Series Figure Creation Tests
def test_create_time_series_figure(sample_events_df: pd.DataFrame):
    """Verify Plotly figure generation for sensor and model score channels."""
    ts_df = dashboard_helpers.prepare_time_series_data(sample_events_df, machine_id="M-001")

    # Sensor channel figure
    fig_sensor = dashboard_helpers.create_time_series_figure(
        ts_df,
        machine_id="M-001",
        channel="Rotational speed [rpm]",
    )
    assert len(fig_sensor.data) >= 1
    assert "M-001" in fig_sensor.layout.title.text

    # Risk score channel figure (has threshold lines)
    fig_score = dashboard_helpers.create_time_series_figure(
        ts_df,
        machine_id="M-001",
        channel="positive_class_score",
        warning_thresh=0.3,
        critical_thresh=0.7,
    )
    assert len(fig_score.data) >= 1
    assert len(fig_score.layout.shapes) >= 2  # Horizontal threshold lines


# 8. Streamlit App Syntax and Import Smoke Test
def test_app_syntax_and_import():
    """Smoke test verifying app.py can be compiled and parsed without syntax error."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.is_file()

    with open(app_path, "r", encoding="utf-8") as fp:
        code_content = fp.read()

    # Verify standard compilation to bytecode
    compiled = compile(code_content, str(app_path), "exec")
    assert compiled is not None


# 9. Warning Demonstration Distribution Verification
def test_warning_demonstration_scenario_distribution():
    """Verify that the mixed scenario seed 3 produces >=1 NORMAL, exactly 1 WARNING, 0 CRITICAL, 0 INVALID."""
    from src import simulator

    events = simulator.run_simulation(
        num_records=4,
        num_machines=2,
        scenario="mixed",
        seed=3,
        model="random_forest",
    )
    df = pd.DataFrame(events)
    kpi = dashboard_helpers.compute_kpi_summary(df)

    assert kpi["total_events"] == 4
    assert kpi["normal_count"] == 3
    assert kpi["warning_count"] == 1
    assert kpi["critical_count"] == 0
    assert kpi["invalid_count"] == 0
    assert kpi["distinct_machines"] == 2

    # Verify warning score falls within interval [0.3, 0.7)
    warning_events = [e for e in events if e["alert_level"] == alert_system.ALERT_WARNING]
    assert len(warning_events) == 1
    assert 0.30 <= warning_events[0]["positive_class_score"] < 0.70


def test_sample_warning_csv_distribution():
    """Verify data/sample_warning_records.csv produces >=1 NORMAL, exactly 1 WARNING, 0 CRITICAL, 0 INVALID."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_warning_records.csv"
    assert sample_path.is_file()

    df_sample = pd.read_csv(sample_path)
    events = dashboard_helpers.convert_raw_telemetry_to_events(
        df=df_sample,
        model="random_forest",
        alert_config=alert_system.AlertConfig(warning_threshold=0.3, critical_threshold=0.7),
    )
    df_events = pd.DataFrame(events)
    kpi = dashboard_helpers.compute_kpi_summary(df_events)

    assert kpi["total_events"] == 4
    assert kpi["normal_count"] == 3
    assert kpi["warning_count"] == 1
    assert kpi["critical_count"] == 0
    assert kpi["invalid_count"] == 0

