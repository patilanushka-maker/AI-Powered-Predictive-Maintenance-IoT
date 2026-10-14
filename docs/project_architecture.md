# Project Architecture: AI-Predictive-Maintenance-IoT

## High-Level System Architecture

This document describes the end-to-end architecture and operational flow of the **AI-Predictive-Maintenance-IoT** prototype.

---

## 1. Planned End-to-End Pipeline Flow

```
[ Planned Pipeline Data & Decision Flow ]

+-------------------------+
|   1. Dataset Source     |  UCI AI4I 2020 Predictive Maintenance Dataset
|  (Raw Telemetry / Replay|  or Real-Time Virtual Sensor Telemetry Stream
+-------------------------+
             |
             v
+-------------------------+
|  2. Ingestion & Quality |  src/data_loader.py
|   Validation Pipeline   |  - Schema checks, range boundaries, null checks
+-------------------------+
             |
             v
+-------------------------+
|   3. Preprocessing &    |  src/preprocessing.py & src/feature_engineering.py
|   Feature Engineering   |  - Cleaning, scaling, thermal gradients, mechanical power
+-------------------------+
             |
             v
+-------------------------+
|    4. Model Training    |  src/train_model.py
|    & Serialization      |  - Baseline & ensemble estimators, class weighting
+-------------------------+
             |
             +---------------------------------------+
             |                                       |
             v                                       v
+-------------------------+             +-------------------------+
|  5. Model Evaluation    |             |  6. Inference Engine    |  src/predict.py
|  & Benchmarking         |             |  (Batch & Streaming)    |  - Risk scoring
|  src/evaluate_model.py  |             +-------------------------+
+-------------------------+                          |
                                                     v
                                        +-------------------------+
                                        |   7. Alerting System    |  src/alert_system.py
                                        |   & Risk Rules Engine   |  - Normal/Warning/Critical
                                        +-------------------------+
                                                     |
                                                     v
                                        +-------------------------+
                                        |  8. Operator Dashboard  |  app.py
                                        |   (Streamlit Interface) |  - Live telemetry & alerts
                                        +-------------------------+
```

---

## 2. Component Implementation Status Matrix

The following matrix distinguishes currently initialized scaffold components from planned functional components:

| Component | Target File | Status | Implementation Responsibility |
| :--- | :--- | :--- | :--- |
| **Repository Scaffolding** | Root & Directory Layout | **Implemented (Phase 1)** | Directory layout, documentation, dependencies, gitignore |
| **Data Loader** | `src/data_loader.py` | *Planned (Phase 2)* | CSV parsing, format checks, schema enforcement |
| **Data Preprocessing** | `src/preprocessing.py` | *Planned (Phase 2)* | Missing data handling, robust scaling, categorical encoding |
| **Feature Engineering** | `src/feature_engineering.py` | *Planned (Phase 2)* | Thermal delta, rotational torque power, strain wear metrics |
| **Model Training** | `src/train_model.py` | *Planned (Phase 3)* | Pipeline orchestration, cross-validation, model serialization |
| **Model Evaluation** | `src/evaluate_model.py` | *Planned (Phase 3)* | Precision, recall, ROC-AUC, PR-AUC, confusion matrix logging |
| **Inference Engine** | `src/predict.py` | *Planned (Phase 4)* | Batch scoring, single-instance inference pipeline |
| **Alert System** | `src/alert_system.py` | *Planned (Phase 4)* | Multi-tier thresholding, warning triggers, incident payloads |
| **IoT Telemetry Simulator** | `src/simulator.py` | *Planned (Phase 4)* | Streaming event replay, jitter generation, synthetic drift |
| **Streamlit Dashboard** | `app.py` | *Planned (Phase 5)* | Web GUI, metric visualizers, equipment status cards |
| **Workflow CLI** | `main.py` | *Planned (Phase 5)* | CLI unified runner for training, evaluation, and simulation |

---

## 3. Communication & Data Contracts (Planned)
- **Input Data Contract**: Structured multivariate tabular format (CSV or JSON dictionary) with strict typing for sensor channels.
- **Inference Output Contract**:
  ```json
  {
    "device_id": "M_001",
    "timestamp": "2026-10-14T07:45:00Z",
    "failure_predicted": false,
    "confidence_score": 0.082,
    "failure_mode": "None",
    "alert_level": "NORMAL"
  }
  ```
