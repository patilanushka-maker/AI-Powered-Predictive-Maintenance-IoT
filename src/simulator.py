"""IoT Sensor Telemetry Simulator Module for Predictive Maintenance.

This module provides a reproducible, software-based IoT sensor telemetry simulator
designed to generate synthetic operational records reflecting industrial milling
equipment behavior. It feeds generated telemetry records through the existing
inference engine (src.predict) and rule-based maintenance alert system
(src.alert_system) to produce structured simulation event streams.

IMPORTANT DISCLAIMER:
This is a software simulation and demonstration module designed for testing and
evaluating machine learning inference and alerting workflows. It does not connect
to physical industrial equipment, nor does it establish physical equipment safety
or predict true unobserved failure times of real-world machines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import alert_system, predict
from src.preprocessing import BASELINE_PREDICTORS

# Simulation Constants & Defaults
DEFAULT_NUM_RECORDS: int = 10
DEFAULT_NUM_MACHINES: int = 1
DEFAULT_INTERVAL_SECONDS: float = 1.0
DEFAULT_SCENARIO: str = "nominal"
DEFAULT_SEED: int = 42
DEFAULT_START_TIME: str = "2026-01-01T00:00:00Z"

VALID_SCENARIOS: set[str] = {
    "nominal",
    "degradation",
    "mixed",
    "fault_injection",
}


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for IoT sensor telemetry simulation.

    Attributes:
        num_records: Total number of telemetry events to generate (must be >= 0).
        num_machines: Number of simulated machines (must be >= 1).
        interval_seconds: Simulation time step between machine readings in seconds (must be > 0).
        scenario: Operational scenario ('nominal', 'degradation', 'mixed', 'fault_injection').
        seed: Random seed for deterministic reproducibility (None for non-deterministic).
        start_time: ISO-8601 formatted UTC timestamp string for simulation start.
        fault_injection: Whether to explicitly inject an invalid sensor reading for error testing.
    """

    num_records: int = DEFAULT_NUM_RECORDS
    num_machines: int = DEFAULT_NUM_MACHINES
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    scenario: str = DEFAULT_SCENARIO
    seed: int | None = DEFAULT_SEED
    start_time: str = DEFAULT_START_TIME
    fault_injection: bool = False

    def __post_init__(self) -> None:
        """Validate simulation configuration parameters."""
        if isinstance(self.num_records, bool) or not isinstance(self.num_records, int):
            raise ValueError(
                f"num_records must be an integer, got {type(self.num_records).__name__}: {self.num_records}"
            )
        if self.num_records < 0:
            raise ValueError(
                f"num_records must be non-negative (>= 0), got: {self.num_records}"
            )

        if isinstance(self.num_machines, bool) or not isinstance(self.num_machines, int):
            raise ValueError(
                f"num_machines must be an integer, got {type(self.num_machines).__name__}: {self.num_machines}"
            )
        if self.num_machines <= 0:
            raise ValueError(
                f"num_machines must be a positive integer (>= 1), got: {self.num_machines}"
            )

        if isinstance(self.interval_seconds, bool) or not isinstance(
            self.interval_seconds, (int, float)
        ):
            raise ValueError(
                f"interval_seconds must be numeric, got {type(self.interval_seconds).__name__}: {self.interval_seconds}"
            )
        iv_val = float(self.interval_seconds)
        if not math.isfinite(iv_val) or iv_val <= 0.0:
            raise ValueError(
                f"interval_seconds must be a finite positive number (> 0), got: {self.interval_seconds}"
            )

        if self.scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Invalid simulation scenario '{self.scenario}'. "
                f"Must be one of: {sorted(VALID_SCENARIOS)}"
            )

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError(
                f"seed must be an integer or None, got {type(self.seed).__name__}: {self.seed}"
            )

        if not isinstance(self.start_time, str):
            raise ValueError(
                f"start_time must be an ISO-8601 string, got {type(self.start_time).__name__}."
            )
        try:
            _parse_iso_timestamp(self.start_time)
        except Exception as exc:
            raise ValueError(
                f"Invalid start_time format '{self.start_time}'. Must be valid ISO-8601 string: {exc}"
            ) from exc


def _parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    clean_ts = ts_str.strip()
    if clean_ts.endswith("Z"):
        clean_ts = clean_ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(clean_ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_iso_timestamp(dt: datetime) -> str:
    """Format datetime as standard ISO-8601 UTC string with sub-second support."""
    if dt.microsecond == 0:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Preserve millisecond/microsecond resolution when fractional seconds exist
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")



def generate_telemetry_records(
    num_records: int = DEFAULT_NUM_RECORDS,
    num_machines: int = DEFAULT_NUM_MACHINES,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    scenario: str = DEFAULT_SCENARIO,
    seed: int | None = DEFAULT_SEED,
    start_time: str = DEFAULT_START_TIME,
    fault_injection: bool = False,
    config: SimulationConfig | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate reproducible synthetic IoT sensor telemetry records.

    Generates multivariate telemetry for the 6 baseline model predictors:
    - Type ('L', 'M', 'H')
    - Air temperature [K]
    - Process temperature [K]
    - Rotational speed [rpm]
    - Torque [Nm]
    - Tool wear [min]

    And simulation metadata:
    - event_id
    - timestamp
    - machine_id
    - scenario

    Args:
        num_records: Total number of records to emit (default 10).
        num_machines: Number of distinct simulated machines (default 1).
        interval_seconds: Step interval between records for a machine in seconds (default 1.0).
        scenario: Operational scenario ('nominal', 'degradation', 'mixed', 'fault_injection').
        seed: Random seed for deterministic reproducibility (default 42).
        start_time: ISO-8601 string representing simulation start time.
        fault_injection: Whether to inject an invalid record for failure testing.
        config: Optional pre-constructed SimulationConfig instance.
        **kwargs: Optional alias arguments (e.g. records, machines, interval).

    Returns:
        list[dict[str, Any]]: List of generated telemetry dictionaries.

    Raises:
        ValueError: If configuration parameters are invalid.
    """
    if "records" in kwargs and kwargs["records"] is not None:
        num_records = kwargs["records"]
    if "machines" in kwargs and kwargs["machines"] is not None:
        num_machines = kwargs["machines"]
    if "interval" in kwargs and kwargs["interval"] is not None:
        interval_seconds = kwargs["interval"]

    cfg = (
        config
        if config is not None
        else SimulationConfig(
            num_records=num_records,
            num_machines=num_machines,
            interval_seconds=interval_seconds,
            scenario=scenario,
            seed=seed,
            start_time=start_time,
            fault_injection=fault_injection,
        )
    )

    if cfg.num_records == 0:
        return []

    rng = np.random.default_rng(cfg.seed)
    start_dt = _parse_iso_timestamp(cfg.start_time)

    # Initialize per-machine properties and baselines
    machine_types = ["L", "M", "H"]
    machines_meta: list[dict[str, Any]] = []
    for m in range(cfg.num_machines):
        m_type = machine_types[m % len(machine_types)]
        m_meta = {
            "machine_id": f"M-{m + 1:03d}",
            "type": m_type,
            "base_air_temp": 300.0 + float(rng.uniform(-0.5, 0.5)),
            "base_temp_diff": 10.0 + float(rng.uniform(-0.3, 0.3)),
            "base_speed": 1540.0 + float(rng.uniform(-20.0, 20.0)),
            "base_torque": 40.0 + float(rng.uniform(-2.0, 2.0)),
            "base_wear": 15.0 + float(rng.uniform(0.0, 15.0)),
        }
        machines_meta.append(m_meta)

    records: list[dict[str, Any]] = []
    machine_step_counters = [0] * cfg.num_machines

    # Determine index for optional fault injection
    fault_target_index = 1 if (cfg.num_records > 1) else 0

    for i in range(cfg.num_records):
        m_idx = i % cfg.num_machines
        step = machine_step_counters[m_idx]
        machine_step_counters[m_idx] += 1
        meta = machines_meta[m_idx]

        # Determine scenario branch for this machine and step
        is_degraded = False
        if cfg.scenario == "degradation":
            is_degraded = True
        elif cfg.scenario == "mixed":
            if cfg.num_machines > 1:
                is_degraded = (m_idx % 2 == 1)
            else:
                is_degraded = i >= (cfg.num_records // 2)

        if not is_degraded:
            cur_scenario = "nominal"
            air_temp = meta["base_air_temp"] + float(rng.normal(0, 0.3))
            temp_diff = meta["base_temp_diff"] + float(rng.normal(0, 0.2))
            proc_temp = max(air_temp + 8.5, air_temp + temp_diff)
            speed = meta["base_speed"] + float(rng.normal(0, 15.0))
            torque = meta["base_torque"] + float(rng.normal(0, 2.0))
            wear = meta["base_wear"] + step * float(rng.uniform(0.2, 0.5))
        else:
            cur_scenario = "degradation"
            # Progressive mechanical and thermal stress accumulation
            wear = meta["base_wear"] + 140.0 + step * float(rng.uniform(2.5, 4.5))
            torque = meta["base_torque"] + 16.0 + (step * 0.9) + float(rng.normal(0, 2.0))
            speed = meta["base_speed"] - (step * 5.0) + float(rng.normal(0, 20.0))
            air_temp = meta["base_air_temp"] + (step * 0.06) + float(rng.normal(0, 0.3))
            temp_diff = meta["base_temp_diff"] + 2.5 + (step * 0.15) + float(rng.normal(0, 0.2))
            proc_temp = air_temp + temp_diff

        # Bound values within realistic domain intervals
        air_temp = round(float(np.clip(air_temp, 294.0, 310.0)), 2)
        proc_temp = round(float(np.clip(proc_temp, air_temp + 8.0, 320.0)), 2)
        speed = round(float(np.clip(speed, 1000.0, 2800.0)), 1)
        torque = round(float(np.clip(torque, 10.0, 85.0)), 2)
        wear = round(float(np.clip(wear, 0.0, 300.0)), 1)

        # Fault injection handling
        inject_here = (cfg.fault_injection or cfg.scenario == "fault_injection") and (
            i == fault_target_index
        )
        if inject_here:
            speed = -50.0  # Physically invalid negative rotational speed
            cur_scenario = "fault_injection"
        elif cfg.scenario == "fault_injection" and not inject_here:
            cur_scenario = "nominal"

        # Calculate timestamp for machine step
        ts = start_dt + timedelta(seconds=step * cfg.interval_seconds)
        ts_str = _format_iso_timestamp(ts)

        record = {
            "event_id": f"EVT-{i + 1:05d}",
            "timestamp": ts_str,
            "machine_id": meta["machine_id"],
            "scenario": cur_scenario,
            "Type": meta["type"],
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }
        records.append(record)

    return records


def run_simulation(
    num_records: int = DEFAULT_NUM_RECORDS,
    num_machines: int = DEFAULT_NUM_MACHINES,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    scenario: str = DEFAULT_SCENARIO,
    seed: int | None = DEFAULT_SEED,
    model: str | Path | Pipeline | None = None,
    alert_config: alert_system.AlertConfig | dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    start_time: str = DEFAULT_START_TIME,
    fault_injection: bool = False,
    config: SimulationConfig | None = None,
    predict_fn: Callable[..., dict[str, Any]] | None = None,
    alert_fn: Callable[..., dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Execute end-to-end telemetry simulation with model inference and alerting.

    Orchestrates the complete Phase 7 pipeline:
    1. Generates synthetic telemetry records according to configuration.
    2. Validates raw sensor values against permitted baseline schema.
    3. Executes Phase 5 inference to estimate failure risk scores.
    4. Evaluates Phase 6 rule-based maintenance alert classifications.
    5. Combines telemetry metadata, model predictions, and alert ratings into
       structured simulation events.
    6. Optionally persists simulation events to structured JSON or CSV file.

    Zero-Retraining Guarantee:
    This function strictly executes pre-fitted pipelines or user-injected inference
    functions. It never invokes fit, fit_transform, or training operations, and never
    mutates saved model artifacts.

    Args:
        num_records: Total number of simulation events to generate.
        num_machines: Number of simulated machines.
        interval_seconds: Simulation time step in seconds.
        scenario: Operational scenario ('nominal', 'degradation', 'mixed', 'fault_injection').
        seed: Random seed for deterministic reproducibility.
        model: Model key, artifact path, Pipeline instance, or None (defaults to provisional reference model).
        alert_config: AlertConfig instance, threshold dictionary, or None.
        output_path: Optional path to save structured output (.json or .csv).
        start_time: ISO-8601 string for simulation start timestamp.
        fault_injection: Whether to inject an invalid telemetry reading for error testing.
        config: Optional pre-constructed SimulationConfig.
        predict_fn: Optional dependency-injected prediction callable for isolated unit testing.
        alert_fn: Optional dependency-injected alert callable for isolated unit testing.
        **kwargs: Optional aliases (e.g. records, machines, interval).

    Returns:
        list[dict[str, Any]]: List of structured simulation event dictionaries.

    Raises:
        ValueError: If simulation configuration or parameters are invalid.
        FileNotFoundError: If the specified model artifact cannot be located on disk.
    """
    if "records" in kwargs and kwargs["records"] is not None:
        num_records = kwargs["records"]
    if "machines" in kwargs and kwargs["machines"] is not None:
        num_machines = kwargs["machines"]
    if "interval" in kwargs and kwargs["interval"] is not None:
        interval_seconds = kwargs["interval"]

    cfg = (
        config
        if config is not None
        else SimulationConfig(
            num_records=num_records,
            num_machines=num_machines,
            interval_seconds=interval_seconds,
            scenario=scenario,
            seed=seed,
            start_time=start_time,
            fault_injection=fault_injection,
        )
    )

    # 1. Generate raw synthetic telemetry records
    raw_records = generate_telemetry_records(config=cfg)

    # Handle empty simulation
    if len(raw_records) == 0:
        if output_path is not None:
            _save_simulation_events([], output_path)
        return []

    # 2. Resolve inference pipeline and alert configuration
    pipeline: Pipeline | None = None
    default_model_name: str | None = None
    if predict_fn is None:
        pipeline, default_model_name = predict.load_inference_pipeline(model)

    resolved_alert_cfg = alert_system.resolve_alert_config(alert_config)

    # 3. Process each record through validation, inference, and alert classification
    events: list[dict[str, Any]] = []

    for rec in raw_records:
        raw_telemetry = {col: rec[col] for col in BASELINE_PREDICTORS}

        # Step A: Validate telemetry inputs
        try:
            val_rec, val_warns = predict.validate_prediction_record(raw_telemetry)
            validation_error = None
        except ValueError as err:
            val_rec = None
            val_warns = []
            validation_error = str(err)

        # Step B: If validation failed, record invalid event immediately
        if validation_error is not None:
            invalid_event = {
                "event_id": rec["event_id"],
                "timestamp": rec["timestamp"],
                "machine_id": rec["machine_id"],
                "scenario": rec["scenario"],
                "Type": rec["Type"],
                "Air temperature [K]": rec["Air temperature [K]"],
                "Process temperature [K]": rec["Process temperature [K]"],
                "Rotational speed [rpm]": rec["Rotational speed [rpm]"],
                "Torque [Nm]": rec["Torque [Nm]"],
                "Tool wear [min]": rec["Tool wear [min]"],
                "model_name": default_model_name,
                "predicted_class": None,
                "predicted_label": None,
                "positive_class_score": None,
                "alert_level": alert_system.ALERT_INVALID,
                "status": "invalid",
                "reason": f"Telemetry validation failed: {validation_error}",
                "recommended_action": alert_system.ACTION_INVALID,
                "warnings": [validation_error],
            }
            events.append(invalid_event)
            continue

        # Step C: Execute inference
        try:
            if predict_fn is not None:
                pred_res = predict_fn(val_rec)
            else:
                pred_res = predict.predict_single(
                    val_rec,
                    model=pipeline,
                    threshold=resolved_alert_cfg.warning_threshold,
                )
        except Exception as exc:
            pred_res = {
                "status": "error",
                "error": f"Inference execution failed: {exc}",
                "model_name": default_model_name,
                "warnings": val_warns,
            }

        # Step D: Execute alert classification
        if alert_fn is not None:
            alert_res = alert_fn(pred_res)
        else:
            alert_res = alert_system.classify_alert(pred_res, config=resolved_alert_cfg)

        # Step E: Combine into structured event dictionary
        is_successful = (pred_res.get("status") == "success") and (
            alert_res.get("status") == "success"
        )
        event_status = "success" if is_successful else "invalid"

        combined_warnings = list(
            dict.fromkeys(
                val_warns
                + pred_res.get("warnings", [])
                + alert_res.get("warnings", [])
            )
        )

        event = {
            "event_id": rec["event_id"],
            "timestamp": rec["timestamp"],
            "machine_id": rec["machine_id"],
            "scenario": rec["scenario"],
            "Type": rec["Type"],
            "Air temperature [K]": rec["Air temperature [K]"],
            "Process temperature [K]": rec["Process temperature [K]"],
            "Rotational speed [rpm]": rec["Rotational speed [rpm]"],
            "Torque [Nm]": rec["Torque [Nm]"],
            "Tool wear [min]": rec["Tool wear [min]"],
            "model_name": alert_res.get("model_name") or pred_res.get("model_name"),
            "predicted_class": pred_res.get("predicted_class"),
            "predicted_label": pred_res.get("predicted_label"),
            "positive_class_score": alert_res.get("positive_class_score")
            if alert_res.get("positive_class_score") is not None
            else pred_res.get("positive_class_score"),
            "alert_level": alert_res.get("alert_level", alert_system.ALERT_INVALID),
            "status": event_status,
            "reason": alert_res.get("reason", "Unknown alert classification reason."),
            "recommended_action": alert_res.get("recommended_action"),
            "warnings": combined_warnings,
        }
        events.append(event)

    # 4. Save events if requested
    if output_path is not None:
        _save_simulation_events(events, output_path)

    return events


def _save_simulation_events(
    events: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save structured simulation events to a JSON or CSV destination.

    Args:
        events: List of simulation event dictionaries.
        output_path: Target file path.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ext = out_p.suffix.lower()
    if ext == ".json":
        with open(out_p, "w", encoding="utf-8") as fp:
            json.dump(events, fp, indent=2)
    elif ext == ".csv":
        if len(events) == 0:
            cols = [
                "event_id",
                "timestamp",
                "machine_id",
                "scenario",
                *BASELINE_PREDICTORS,
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
            pd.DataFrame(columns=cols).to_csv(out_p, index=False)
        else:
            df = pd.DataFrame(events)
            if "warnings" in df.columns:
                df["warnings"] = df["warnings"].apply(
                    lambda w: "; ".join(w) if isinstance(w, list) else str(w)
                )
            df.to_csv(out_p, index=False)
    else:
        raise ValueError(
            f"Unsupported simulation output format '{ext}'. Must be .json or .csv."
        )
