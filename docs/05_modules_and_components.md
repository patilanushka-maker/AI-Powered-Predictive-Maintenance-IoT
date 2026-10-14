# 05. Modules & Components Specification

---

## 1. System Module Inventory

| Module Path | Primary Responsibility | Implemented Phase | Key Public Functions / Classes |
| :--- | :--- | :---: | :--- |
| [`src/data_loader.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/data_loader.py) | Raw telemetry ingestion, schema validation, quality checks | Phase 2 | `load_raw_data`, `validate_schema`, `analyze_data_quality`, `generate_validation_report` |
| [`src/preprocessing.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/preprocessing.py) | Leakage-safe feature separation, transformation, stratified split | Phase 3 | `separate_features_and_target`, `create_preprocessor`, `split_data`, `preprocess_data` |
| [`src/train_model.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/train_model.py) | Baseline pipeline training, configuration factory, serialization | Phase 4 | `get_baseline_model_configs`, `build_model_pipeline`, `train_all_baseline_models`, `save_pipeline`, `load_pipeline` |
| [`src/evaluate_model.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/evaluate_model.py) | Metric calculation, report generation, validation threshold analysis | Phase 4 | `compute_classification_metrics`, `evaluate_pipeline`, `evaluate_all_models`, `run_baseline_evaluation` |
| [`src/predict.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/predict.py) | Raw telemetry validation, batch/single inference, structured results | Phase 5 | `predict`, `predict_single`, `predict_batch`, `load_inference_pipeline`, `validate_prediction_record` |
| [`src/alert_system.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/alert_system.py) | Severity ranking, multi-level alerting, recommended mitigation | Phase 6 | `AlertConfig`, `classify_alert`, `classify_alert_batch`, `evaluate_alerts_from_telemetry` |
| [`src/simulator.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/simulator.py) | Synthetic IoT sensor stream simulator, progressive degradation, CLI runner | Phase 7 | `SimulationConfig`, `generate_telemetry_records`, `run_simulation` |
| [`src/dashboard_helpers.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/src/dashboard_helpers.py) | Dashboard metrics, KPI aggregation, CSV ingestion, Plotly charts | Phase 8 | `compute_kpi_summary`, `filter_events`, `get_latest_machine_telemetry`, `prepare_time_series_data`, `validate_and_parse_uploaded_csv`, `convert_raw_telemetry_to_events`, `create_time_series_figure` |
| [`app.py`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/app.py) | Interactive Streamlit fleet monitoring and alert triage interface | Phase 8 | Streamlit web application |

---

## 2. Inference Engine Specification (`src/predict.py`)

The inference engine executes predictions on raw, un-transformed sensor telemetry by leveraging serialized Phase 4 scikit-learn pipelines.

### 2.1 Input Data Contract

The engine accepts raw dictionaries, pandas Series, lists of dictionaries, or DataFrames containing strictly the **six permitted baseline predictors**:

| Field Name | Type | Physical Range / Valid Values | Description |
| :--- | :--- | :--- | :--- |
| `Type` | `str` | `{'L', 'M', 'H'}` | Machine variant quality grade |
| `Air temperature [K]` | `float` | Continuous, $> 0\text{ K}$ | Ambient operating temperature |
| `Process temperature [K]` | `float` | Continuous, $> 0\text{ K}$ | Internal processing temperature |
| `Rotational speed [rpm]` | `float` or `int` | Continuous, $\ge 0\text{ rpm}$ | Spindle rotation speed |
| `Torque [Nm]` | `float` | Continuous, $\ge 0\text{ Nm}$ | Spindle resistance torque |
| `Tool wear [min]` | `float` or `int` | Continuous, $\ge 0\text{ min}$ | Cumulative tool usage |

#### Strictly Forbidden Fields
Inputs containing any of the following fields are rejected immediately with a `ValueError`:
* Row identifiers: `UDI`, `Product ID`
* Post-hoc failure modes: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
* Target label: `Machine failure`

### 2.2 Input Validation Rules
1. **Schema Completeness**: All 6 permitted fields must be present.
2. **Type Checking**: Non-numeric values (strings, `None`, booleans) in numeric sensor fields are rejected.
3. **Finiteness**: Values of `NaN`, `+Infinity`, or `-Infinity` are rejected.
4. **Physical Boundaries**: Negative rotational speed, torque, tool wear, or non-positive Kelvin temperatures are rejected.
5. **Caller Immutability**: Input dictionaries and DataFrames are never mutated or modified.
6. **Domain Plausibility Warnings**: Inverted thermal gradients ($T_{\text{process}} < T_{\text{air}}$) or extreme operational stresses generate non-blocking advisory warnings.

### 2.3 Output Result Contract

Each valid prediction produces a structured, machine-readable dictionary:

```python
{
    "model_name": "Random Forest (Unweighted)",
    "predicted_class": 1,
    "predicted_label": "Failure label predicted",
    "positive_class_score": 0.840000,
    "threshold": 0.5,
    "status": "success",
    "warnings": []
}
```

* `predicted_class`: Integer `0` (nominal) or `1` (predicted failure).
* `predicted_label`: Explicit string mapping:
  * `0` $\rightarrow$ `"No failure label predicted"`
  * `1` $\rightarrow$ `"Failure label predicted"`
  *(Note: Avoids misleading deterministic language like "machine is safe" or "machine will fail".)*
* `positive_class_score`: Estimated score/probability of class 1 under the fitted model.
* `threshold`: Classification decision threshold applied (default `0.5`).
* `status`: `"success"` or `"error"`.
* `warnings`: List of non-fatal operational advisories.

### 2.4 Model Resolution and Pipeline Reuse
* **Default Reference Model**: `models/baseline_random_forest.joblib` (`random_forest`).
* **Available Model Keys**:
  * `random_forest`: Provisional reference model (strongest F1 = 0.6667, PR-AUC = 0.7892).
  * `random_forest_balanced`: High-precision variant (Precision = 0.9677, 1 false alarm).
  * `logistic_regression_balanced`: High-sensitivity triage filter (Recall = 0.8235).
  * `logistic_regression`: Linear unweighted baseline.
* **Pipeline Reuse**: Functions accept an already-instantiated `Pipeline` object directly, eliminating redundant disk I/O when processing streaming telemetry.

### 2.5 Batch Execution & Error Policies
* `error_policy="raise"` (default): Rejects the entire batch on the first validation error, reporting the exact row index and field violation.
* `error_policy="per_record"`: Isolates invalid records by returning `{"status": "error", "row_index": i, "error": ...}` while utilizing vectorized execution across all valid records.

---

## 3. Maintenance Alert System Specification (`src/alert_system.py`)

The alert system consumes Phase 5 prediction outputs and evaluates rule-based maintenance alert levels (`NORMAL`, `WARNING`, `CRITICAL`, `INVALID`). It generates actionable technical guidance without retraining models or modifying model files.

### 3.1 Alert Levels & Boundary Semantics

| Alert Level | Trigger Condition | Operational Meaning | Recommended Maintenance Action |
| :--- | :--- | :--- | :--- |
| **`NORMAL`** | $\text{score} < \text{warning\_threshold}$ | Nominal operating condition; low degradation risk | Continue standard operational monitoring schedule. No maintenance intervention required. |
| **`WARNING`** | $\text{warning\_threshold} \le \text{score} < \text{critical\_threshold}$ | Elevated component wear or thermal/mechanical stress | Schedule non-urgent technical inspection. Review thermal and mechanical stress indicators. |
| **`CRITICAL`** | $\text{score} \ge \text{critical\_threshold}$ | Elevated failure risk score; prioritized review | Prioritize maintenance review. Verify sensor readings and operating conditions, and follow the site's approved safety procedures before taking operational action. |
| **`INVALID`** | Missing score, non-numeric, $\text{NaN}$, $\pm\infty$, or inference error | Input data corruption or pipeline execution failure | Inspect data quality and fix sensor telemetry input formatting before evaluating alerts. |

*Exact Boundary Rules*:
* A score exactly equal to `warning_threshold` produces `WARNING`.
* A score exactly equal to `critical_threshold` produces `CRITICAL`.
* Invalid scores or failed predictions **never** silently default to `NORMAL`.

### 3.2 Configuration Contract (`AlertConfig`)
* `warning_threshold`: Finite numeric float in $(0.0, 1.0)$ exclusive (default: `0.3`).
* `critical_threshold`: Finite numeric float in $(0.0, 1.0)$ exclusive (default: `0.7`).
* Ordering constraint: `critical_threshold > warning_threshold` strictly enforced.
* *Notice*: These thresholds are illustrative demonstration defaults. Real-world industrial deployment requires operational risk calibration based on unplanned downtime costs versus inspection overhead.

### 3.3 Output Result Contract
```python
{
    "status": "success",
    "alert_level": "WARNING",
    "reason": "Failure risk score (0.5000) reached or exceeded warning threshold (0.3000) but is below critical threshold (0.7000). Elevated component degradation risk.",
    "positive_class_score": 0.500000,
    "thresholds_used": {
        "warning_threshold": 0.3,
        "critical_threshold": 0.7
    },
    "recommended_action": "Schedule non-urgent technical inspection. Review thermal and mechanical stress indicators.",
    "model_name": "Random Forest (Unweighted)",
    "record_index": 0,
    "warnings": []
}
```

### 3.4 Distinction Between Scores, Classes, and Alerts
1. **Model Score (`positive_class_score`)**: The statistical confidence output of the classifier in $[0.0, 1.0]$. Not a calibrated real-world physical failure probability.
2. **Binary Decision (`predicted_class`)**: A model-level partition at a fixed threshold (e.g. 0.5).
3. **Operational Alert (`alert_level`)**: Rule-based dispatch policy applying multi-tier boundaries (`NORMAL`, `WARNING`, `CRITICAL`) to guide physical human maintenance workflows.

---

## 4. IoT Telemetry Simulator Specification (`src/simulator.py`)

The simulator module generates synthetic, multivariate IoT sensor streams reflecting industrial milling equipment behavior. It feeds generated records through the inference and alerting pipeline and records structured event histories.

### 4.1 Synthetic Simulation Disclaimer
This module is a local software generator for testing and demonstration. It does not connect to physical equipment, does not capture live physical failure dynamics, and does not predict true failure times of actual physical machinery.

### 4.2 Configuration Parameters (`SimulationConfig`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_records` | `int` | `10` | Total number of telemetry events to emit ($\ge 0$). |
| `num_machines` | `int` | `1` | Number of distinct simulated machine streams ($\ge 1$). |
| `interval_seconds` | `float` | `1.0` | Simulation time step between successive readings per machine ($> 0$). |
| `scenario` | `str` | `"nominal"` | Operational scenario: `'nominal'`, `'degradation'`, `'mixed'`, `'fault_injection'`. |
| `seed` | `int` or `None` | `42` | Random generator seed for strict reproducibility. |
| `start_time` | `str` | `"2026-01-01T00:00:00Z"` | ISO-8601 UTC timestamp string for simulation timeline start. |
| `fault_injection` | `bool` | `False` | When True, injects an invalid sensor reading to test error isolation. |

### 4.3 Simulation Scenarios
1. **`nominal`**: Machines operate within standard baseline boundaries. Rotational speed ~1540 rpm, torque ~40 Nm, temperatures normal ($T_{\text{process}} \approx T_{\text{air}} + 10 \text{ K}$), tool wear accumulates gradually.
2. **`degradation`**: Machines experience progressive mechanical and thermal wear. Tool wear starts higher (140+ min) and accumulates rapidly; cutting friction drives torque upward; spindle speed fluctuates; process temperature rises due to degraded heat dissipation.
3. **`mixed`**: Multi-machine simulation where half the fleet operates nominally while the remaining machines experience degradation.
4. **`fault_injection`**: Explicit error test where a designated step emits physically invalid telemetry (e.g., negative rotational speed) to verify non-fatal error isolation and `INVALID` alert generation.

### 4.4 Structured Event Record Schema

Each event produced by `run_simulation` contains the following fields:

```python
{
    "event_id": "EVT-00001",
    "timestamp": "2026-01-01T00:00:00Z",
    "machine_id": "M-001",
    "scenario": "nominal",
    "Type": "L",
    "Air temperature [K]": 299.88,
    "Process temperature [K]": 309.87,
    "Rotational speed [rpm]": 1549.6,
    "Torque [Nm]": 40.76,
    "Tool wear [min]": 16.4,
    "model_name": "RandomForestClassifier",
    "predicted_class": 0,
    "predicted_label": "No failure label predicted",
    "positive_class_score": 0.0000,
    "alert_level": "NORMAL",
    "status": "success",
    "reason": "Failure risk score (0.0000) is below warning threshold (0.3000). Nominal operating condition.",
    "recommended_action": "Continue standard operational monitoring schedule. No maintenance intervention required.",
    "warnings": []
}
```

*Status vs. Alert Level*:
* `status`: Processing validity (`"success"` or `"invalid"`).
* `alert_level`: Equipment maintenance urgency (`"NORMAL"`, `"WARNING"`, `"CRITICAL"`, `"INVALID"`).
* If telemetry validation fails, `status="invalid"`, `alert_level="INVALID"`, `predicted_class=None`, and `positive_class_score=None`.

### 4.5 Zero-Retraining Guarantee
The simulator strictly consumes serialized model pipelines via `predict.load_inference_pipeline` in read-only mode. It never calls `fit`, `fit_transform`, or retraining methods, and never modifies disk artifacts.

---

## 5. Interactive Monitoring Dashboard Specification (`app.py` & `src/dashboard_helpers.py`)

The Streamlit web application provides an operator-facing monitoring console designed to track fleet telemetry, visualize degradation trajectories, and triage maintenance alerts.

### 5.1 Architecture & Separation of Concerns
- **Application Layer (`app.py`)**: Responsible strictly for layout orchestration, session state management, and user interaction.
- **Helper Utilities (`src/dashboard_helpers.py`)**: Encapsulates data aggregation, KPI computation, input CSV validation, and Plotly figure generation.
- **Core Domain Logic**: Directly delegated to existing modules (`src.simulator`, `src.predict`, `src.alert_system`) to prevent redundant inference or alerting logic.

### 5.2 Dashboard Capabilities
1. **Fleet KPI Summary Deck**:
   - `Total Events`: Total telemetry cycles loaded or simulated.
   - `Normal [OK]`: Count of cycles below the warning risk threshold.
   - `Warning [Inspect]`: Count of cycles requiring non-urgent inspection.
   - `Critical [Mitigate]`: Count of cycles requiring immediate prioritized review.
   - `Invalid [Data Error]`: Count of records with malformed sensor readings (isolated and never silently treated as normal).
   - `Distinct Machines`: Total number of active machines in the telemetry stream.
2. **Machine Telemetry & Fleet Overview**:
   - Displays real-time status cards for each machine in the fleet.
   - Provides a filterable table by Machine ID and Alert Severity.
3. **Sensor & Risk Trajectory Visualizations**:
   - Interactive Plotly time-series plots for physical sensor variables (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).
   - Plot of model failure risk score (`positive_class_score`) with horizontal dashed threshold lines for warning (0.3) and critical (0.7) boundaries.
4. **Maintenance Alert Console**:
   - Triage console providing operational reason explanations and actionable mitigation instructions per event.
5. **CSV Ingestion & Leakage Prevention**:
   - Ingests both pre-generated simulation events and raw telemetry files.
   - Validates the 6 baseline predictors and strictly rejects target labels (`Machine failure`) or post-failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).



