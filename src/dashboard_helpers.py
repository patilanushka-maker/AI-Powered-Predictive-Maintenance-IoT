"""Dashboard Utility and Helper Functions for Streamlit Application (Phase 8).

This module provides focused, testable data transformation, aggregation,
validation, and visualization helper functions for the Streamlit monitoring interface.
It consumes simulation events and raw telemetry, delegates core business logic to
the existing src modules, and enforces fail-safe data handling.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

from src import alert_system, predict
from src.preprocessing import (
    BASELINE_PREDICTORS,
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
)

# Standard Event Schema Columns
EVENT_COLUMNS: list[str] = [
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
]

# Color map for alert levels across dashboard widgets and charts
ALERT_COLORS: dict[str, str] = {
    alert_system.ALERT_NORMAL: "#16a34a",    # Forest / Emerald Green
    alert_system.ALERT_WARNING: "#d97706",   # Industrial Amber
    alert_system.ALERT_CRITICAL: "#dc2626",  # Crimson Red
    alert_system.ALERT_INVALID: "#64748b",   # Slate / Neutral Gray
}


def compute_kpi_summary(df: pd.DataFrame | None) -> dict[str, int]:
    """Compute aggregate KPI metrics from an event DataFrame.

    Handles empty, None, or partially structured DataFrames safely without exceptions.

    Args:
        df: Pandas DataFrame of simulation events.

    Returns:
        dict[str, int]: Aggregate counts for total events, alert levels, and distinct machines.
    """
    default_summary: dict[str, int] = {
        "total_events": 0,
        "normal_count": 0,
        "warning_count": 0,
        "critical_count": 0,
        "invalid_count": 0,
        "distinct_machines": 0,
    }

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return default_summary

    total_events = len(df)
    distinct_machines = (
        int(df["machine_id"].nunique()) if "machine_id" in df.columns else 0
    )

    if "alert_level" not in df.columns:
        return {
            "total_events": total_events,
            "normal_count": 0,
            "warning_count": 0,
            "critical_count": 0,
            "invalid_count": 0,
            "distinct_machines": distinct_machines,
        }

    level_counts = df["alert_level"].value_counts().to_dict()

    return {
        "total_events": total_events,
        "normal_count": int(level_counts.get(alert_system.ALERT_NORMAL, 0)),
        "warning_count": int(level_counts.get(alert_system.ALERT_WARNING, 0)),
        "critical_count": int(level_counts.get(alert_system.ALERT_CRITICAL, 0)),
        "invalid_count": int(level_counts.get(alert_system.ALERT_INVALID, 0)),
        "distinct_machines": distinct_machines,
    }


def filter_events(
    df: pd.DataFrame | None,
    machines: list[str] | None = None,
    severities: list[str] | None = None,
) -> pd.DataFrame:
    """Filter simulation events by machine identifiers and alert severity levels.

    Args:
        df: Input DataFrame of simulation events.
        machines: Optional list of machine IDs to include.
        severities: Optional list of alert levels to include.

    Returns:
        pd.DataFrame: Filtered DataFrame preserving row order.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    filtered = df.copy()

    if machines and "machine_id" in filtered.columns:
        filtered = filtered[filtered["machine_id"].isin(machines)]

    if severities and "alert_level" in filtered.columns:
        filtered = filtered[filtered["alert_level"].isin(severities)]

    return filtered


def get_latest_machine_telemetry(df: pd.DataFrame | None) -> pd.DataFrame:
    """Extract the most recent telemetry event for each simulated machine.

    Args:
        df: Input DataFrame of simulation events.

    Returns:
        pd.DataFrame: One row per unique machine containing its latest reading.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    if "machine_id" not in df.columns:
        return df.copy()

    # Sort chronologically if timestamp exists to ensure latest is last
    working = df.copy()
    if "timestamp" in working.columns:
        try:
            working["_parsed_ts"] = pd.to_datetime(working["timestamp"], errors="coerce")
            working = working.sort_values(by="_parsed_ts", ascending=True)
            working = working.drop(columns=["_parsed_ts"])
        except Exception:
            pass

    # Group by machine_id and take last record
    latest = working.groupby("machine_id", as_index=False).last()
    return latest


def prepare_time_series_data(
    df: pd.DataFrame | None,
    machine_id: str | None = None,
) -> pd.DataFrame:
    """Sort and validate event telemetry chronologically for time-series charts.

    Args:
        df: Input DataFrame of simulation events.
        machine_id: Optional machine identifier to filter by.

    Returns:
        pd.DataFrame: Chronologically sorted DataFrame with a valid datetime index or column.

    Raises:
        ValueError: If DataFrame is empty, missing timestamp, or timestamps cannot be parsed.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Cannot prepare time-series chart from empty event data.")

    if "timestamp" not in df.columns:
        raise ValueError("Event dataset does not contain a 'timestamp' column.")

    working = df.copy()
    if machine_id and "machine_id" in working.columns:
        working = working[working["machine_id"] == machine_id]
        if working.empty:
            raise ValueError(f"No event records found for machine '{machine_id}'.")

    # Parse timestamps
    working["_datetime"] = pd.to_datetime(working["timestamp"], errors="coerce")
    valid_count = working["_datetime"].notna().sum()
    if valid_count == 0:
        raise ValueError("Timestamps in the active dataset are invalid or unparseable.")

    working = working.sort_values(by="_datetime", ascending=True)
    return working


def validate_and_parse_uploaded_csv(
    file_or_content: Any,
) -> tuple[str, pd.DataFrame, list[str]]:
    """Inspect and validate an uploaded CSV file.

    Differentiates between:
    1. Pre-generated Simulation Events CSV (has event_id, alert_level, etc.)
    2. Raw Sensor Telemetry CSV (has the 6 permitted baseline predictors)

    Strictly flags target leakage indicators and missing predictors.

    Args:
        file_or_content: Uploaded file object, path, or string buffer.

    Returns:
        tuple[str, pd.DataFrame, list[str]]:
            - Schema type: 'simulation_events', 'raw_telemetry', or 'invalid'
            - Parsed pandas DataFrame
            - List of validation error/warning messages
    """
    # Inspect raw header line to detect duplicate columns before pandas mangles them
    raw_header_line: str | None = None
    try:
        if isinstance(file_or_content, (str, Path)) and Path(file_or_content).is_file():
            with open(file_or_content, "r", encoding="utf-8") as f:
                raw_header_line = f.readline()
        elif hasattr(file_or_content, "seek") and hasattr(file_or_content, "readline"):
            pos = file_or_content.tell()
            line = file_or_content.readline()
            if isinstance(line, bytes):
                raw_header_line = line.decode("utf-8", errors="ignore")
            elif isinstance(line, str):
                raw_header_line = line
            file_or_content.seek(pos)
    except Exception:
        raw_header_line = None

    if raw_header_line and "," in raw_header_line:
        header_tokens = [col.strip() for col in raw_header_line.strip().split(",")]
        seen_tokens = set()
        duplicate_headers = []
        for token in header_tokens:
            if token in seen_tokens and token not in duplicate_headers:
                duplicate_headers.append(token)
            seen_tokens.add(token)
        if duplicate_headers:
            return "invalid", pd.DataFrame(), [f"Malformed CSV schema: Duplicate column names detected in uploaded file: {sorted(duplicate_headers)}."]

    try:
        if isinstance(file_or_content, (str, Path)):
            df = pd.read_csv(file_or_content)
        elif hasattr(file_or_content, "read"):
            df = pd.read_csv(file_or_content)
        else:
            return "invalid", pd.DataFrame(), [f"Unsupported input type: {type(file_or_content).__name__}"]
    except Exception as exc:
        return "invalid", pd.DataFrame(), [f"Failed to parse CSV file: {exc}"]

    if df.empty or len(df) == 0:
        return "invalid", pd.DataFrame(), ["Uploaded CSV file contains zero records."]

    # Fallback check for duplicate columns if pandas preserved any
    if df.columns.has_duplicates:
        duplicate_cols = sorted(list(set(df.columns[df.columns.duplicated()])))
        return "invalid", df, [f"Malformed CSV schema: Duplicate column names detected in uploaded file: {duplicate_cols}."]

    cols = list(df.columns)
    errors: list[str] = []

    # Check for forbidden target leakage columns in any uploaded CSV
    forbidden_present = [col for col in cols if col in EXCLUDED_COLUMNS or col == TARGET_COLUMN]
    if forbidden_present:
        errors.append(
            f"Target leakage violation: Forbidden columns detected in input: {forbidden_present}. "
            "Post-failure indicators and target labels must not be passed to the model."
        )
        return "invalid", df, errors

    # Check for Simulation Events schema
    if "event_id" in cols and "alert_level" in cols:
        # Validate that precomputed events cannot bypass schema requirements
        missing_event_predictors = [col for col in BASELINE_PREDICTORS if col not in cols]
        if missing_event_predictors:
            errors.append(
                f"Malformed event schema: Event records must include baseline sensor features, missing: {missing_event_predictors}."
            )
            return "invalid", df, errors

        if "timestamp" not in cols:
            errors.append("Malformed event schema: Missing required 'timestamp' column.")
            return "invalid", df, errors

        # Validate alert level values
        observed_levels = set(df["alert_level"].dropna().astype(str).unique())
        invalid_levels = observed_levels - alert_system.VALID_ALERT_LEVELS
        if invalid_levels:
            errors.append(
                f"Malformed event schema: Unrecognized alert level(s): {sorted(invalid_levels)}. "
                f"Must be subset of valid levels: {sorted(alert_system.VALID_ALERT_LEVELS)}."
            )
            return "invalid", df, errors

        return "simulation_events", df, []

    # Check for raw telemetry schema
    missing_predictors = [col for col in BASELINE_PREDICTORS if col not in cols]
    if missing_predictors:
        errors.append(
            f"Missing required baseline predictors: {missing_predictors}. "
            f"Required predictors: {BASELINE_PREDICTORS}"
        )
        return "invalid", df, errors

    # Ensure unexpected extraneous feature columns cannot silently enter inference
    allowed_raw_cols = set(BASELINE_PREDICTORS) | {"event_id", "timestamp", "machine_id", "scenario"}
    unexpected_cols = [col for col in cols if col not in allowed_raw_cols]
    if unexpected_cols:
        errors.append(
            f"Unexpected feature columns detected in raw telemetry: {unexpected_cols}. "
            f"Only approved predictors {BASELINE_PREDICTORS} and standard metadata are accepted."
        )
        return "invalid", df, errors

    return "raw_telemetry", df, []


def convert_raw_telemetry_to_events(
    df: pd.DataFrame,
    model: str | Path | Pipeline | None = None,
    alert_config: alert_system.AlertConfig | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute inference and alert evaluation on uploaded raw telemetry records.

    Converts raw telemetry rows into structured event records without mutating models.

    Args:
        df: Input DataFrame containing the 6 baseline predictors.
        model: Model key, artifact path, Pipeline, or None.
        alert_config: AlertConfig instance or threshold dictionary.

    Returns:
        list[dict[str, Any]]: Structured simulation events matching standard contract.
    """
    cfg = alert_system.resolve_alert_config(alert_config)
    pipeline, model_name = predict.load_inference_pipeline(model)

    records: list[dict[str, Any]] = df.to_dict(orient="records")
    events: list[dict[str, Any]] = []

    start_dt = datetime.now(timezone.utc)

    for i, rec in enumerate(records):
        # Extract metadata if already present, or generate sensible defaults
        evt_id = str(rec.get("event_id", f"UPLOAD-{i + 1:05d}"))
        ts = str(rec.get("timestamp", start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")))
        m_id = str(rec.get("machine_id", f"M-{(i % 3) + 1:03d}"))
        scenario = str(rec.get("scenario", "user_upload"))

        raw_telemetry = {col: rec.get(col) for col in BASELINE_PREDICTORS}

        # Validate record
        try:
            val_rec, val_warns = predict.validate_prediction_record(raw_telemetry)
            validation_error = None
        except ValueError as val_err:
            val_rec = None
            val_warns = []
            validation_error = str(val_err)

        if validation_error is not None:
            invalid_event = {
                "event_id": evt_id,
                "timestamp": ts,
                "machine_id": m_id,
                "scenario": scenario,
                "Type": rec.get("Type", "UNKNOWN"),
                "Air temperature [K]": rec.get("Air temperature [K]"),
                "Process temperature [K]": rec.get("Process temperature [K]"),
                "Rotational speed [rpm]": rec.get("Rotational speed [rpm]"),
                "Torque [Nm]": rec.get("Torque [Nm]"),
                "Tool wear [min]": rec.get("Tool wear [min]"),
                "model_name": model_name,
                "predicted_class": None,
                "predicted_label": None,
                "positive_class_score": None,
                "alert_level": alert_system.ALERT_INVALID,
                "status": "invalid",
                "reason": f"Validation failed: {validation_error}",
                "recommended_action": alert_system.ACTION_INVALID,
                "warnings": [validation_error],
            }
            events.append(invalid_event)
            continue

        # Predict
        try:
            pred_res = predict.predict_single(
                val_rec,
                model=pipeline,
                threshold=cfg.warning_threshold,
            )
        except Exception as exc:
            pred_res = {
                "status": "error",
                "error": str(exc),
                "model_name": model_name,
                "warnings": val_warns,
            }

        # Alert
        alert_res = alert_system.classify_alert(pred_res, config=cfg)

        is_ok = (pred_res.get("status") == "success") and (alert_res.get("status") == "success")
        status_val = "success" if is_ok else "invalid"

        event = {
            "event_id": evt_id,
            "timestamp": ts,
            "machine_id": m_id,
            "scenario": scenario,
            "Type": val_rec["Type"],
            "Air temperature [K]": val_rec["Air temperature [K]"],
            "Process temperature [K]": val_rec["Process temperature [K]"],
            "Rotational speed [rpm]": val_rec["Rotational speed [rpm]"],
            "Torque [Nm]": val_rec["Torque [Nm]"],
            "Tool wear [min]": val_rec["Tool wear [min]"],
            "model_name": alert_res.get("model_name") or model_name,
            "predicted_class": pred_res.get("predicted_class"),
            "predicted_label": pred_res.get("predicted_label"),
            "positive_class_score": alert_res.get("positive_class_score")
            if alert_res.get("positive_class_score") is not None
            else pred_res.get("positive_class_score"),
            "alert_level": alert_res.get("alert_level", alert_system.ALERT_INVALID),
            "status": status_val,
            "reason": alert_res.get("reason", ""),
            "recommended_action": alert_res.get("recommended_action"),
            "warnings": list(dict.fromkeys(val_warns + pred_res.get("warnings", []))),
        }
        events.append(event)

    return events


def create_time_series_figure(
    df: pd.DataFrame,
    machine_id: str,
    channel: str,
    warning_thresh: float = 0.3,
    critical_thresh: float = 0.7,
) -> go.Figure:
    """Generate a clean, responsive Plotly interactive time-series figure.

    Args:
        df: Sorted time-series event DataFrame with _datetime column.
        machine_id: Machine label for title and legend.
        channel: Telemetry sensor channel or model score.
        warning_thresh: Warning score threshold for reference lines.
        critical_thresh: Critical score threshold for reference lines.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    fig = go.Figure()

    x_vals = df["_datetime"]
    y_vals = df[channel]

    # Main telemetry trace
    is_score = (channel == "positive_class_score")
    line_color = "#2563eb" if not is_score else "#4f46e5"
    channel_display_name = "Positive-Class Score (Model Output)" if is_score else channel

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name=f"{machine_id} - {channel_display_name}",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=5, color=line_color, line=dict(width=1, color="#ffffff")),
            hovertemplate="<b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>",
        )
    )

    # If plotting risk score, add threshold horizontal reference lines
    if is_score:
        fig.add_hline(
            y=warning_thresh,
            line_dash="dash",
            line_color=ALERT_COLORS[alert_system.ALERT_WARNING],
            annotation_text=f"Warning Threshold ({warning_thresh:.2f})",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#b45309"),
        )
        fig.add_hline(
            y=critical_thresh,
            line_dash="dash",
            line_color=ALERT_COLORS[alert_system.ALERT_CRITICAL],
            annotation_text=f"Critical Threshold ({critical_thresh:.2f})",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#b91c1c"),
        )
        fig.update_yaxes(range=[-0.05, 1.05])

    fig.update_layout(
        title=dict(
            text=f"<b>{channel_display_name}</b> — Machine: <code>{machine_id}</code>",
            x=0.01,
            font=dict(size=15, color="#0f172a", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        ),
        xaxis=dict(
            title=dict(text="Simulation Timestamp (UTC)", font=dict(size=12, color="#475569")),
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
        ),
        yaxis=dict(
            title=dict(text=channel_display_name, font=dict(size=12, color="#475569")),
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=420,
        margin=dict(l=50, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569"),
        ),
    )

    return fig
