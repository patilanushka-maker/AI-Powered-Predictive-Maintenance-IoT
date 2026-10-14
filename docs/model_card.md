# Model Card: AI-Predictive-Maintenance-IoT Baseline Models

---

## 1. Model Details
- **Model Name**: Industrial Machine Predictive Maintenance Baseline Classifiers
- **Model Family**: Scikit-Learn Supervised Classification Pipelines
- **Version**: 0.2.0-baseline (Phase 4 Baseline Initialized)
- **Implemented Configurations**:
  1. `baseline_random_forest.joblib` (**Provisional Reference Model**: strongest F1 = 0.6667, Average Precision (AP) = 0.7892)
  2. `baseline_random_forest_balanced.joblib` (Random Forest, class_weight='balanced')
  3. `baseline_logistic_regression.joblib` (L2 Logistic Regression, unweighted)
  4. `baseline_logistic_regression_balanced.joblib` (L2 Logistic Regression, class_weight='balanced')
- **Framework**: Scikit-Learn 1.4.2 / Joblib 1.4.2 / Python 3.10
- **Serialization Format**: Joblib binary `.joblib` encapsulating full `preprocessor` (`ColumnTransformer`) and `classifier`.
- **Inference Module**: `src/predict.py` (Phase 5: validates 6 permitted raw predictors, vectorized batch inference, configurable threshold).

---

## 2. Intended Use
- **Primary Intended Use**:
  - Baseline early-warning classification of binary equipment failure (`Machine failure`) based on 6 multivariate physical sensor channels.
  - Research and educational demonstration of leak-free predictive maintenance workflows on industrial IoT telemetry.
- **Out-of-Scope & Prohibited Uses**:
  - Direct autonomous control of physical industrial machinery without human-in-the-loop verification.
  - Predicting exact failure timing, remaining useful life (RUL), or physical degradation curves (the model predicts cycle failure classification only).
  - High-consequence safety-critical deployment without hardware certification, empirical calibration, and real-world failure data.

---

## 3. Training & Evaluation Setup
- **Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset (`data/raw/ai4i2020.csv`).
- **Target**: `Machine failure` (binary: 0 = nominal, 1 = failure event).
- **Permitted Predictors**: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- **Excluded Features**: `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- **Partitioning**: Stratified 80/20 train/test split (`random_state=42`).
  - Training: 8,000 instances (7,729 nominal, 271 failures = 3.39%).
  - Testing: 2,000 instances (1,932 nominal, 68 failures = 3.40%).
- **Leakage Prevention**: All transformations (imputation, standard scaling, one-hot encoding) fitted strictly on `X_train`. Held-out test split untouched during model fitting and threshold selection.

---

## 4. Empirical Baseline Performance (Test Split, Threshold = 0.5)

| Metric | Logistic Regression (Unweighted) | Logistic Regression (Balanced) | Random Forest (Unweighted) | Random Forest (Balanced) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.6364 | 0.1418 | **0.9000** | **0.9677** |
| **Recall** | 0.1029 | **0.8235** | 0.5294 | 0.4412 |
| **F1-Score** | 0.1772 | 0.2419 | **0.6667** | 0.6061 |
| **ROC-AUC** | 0.8994 | 0.9070 | **0.9653** | 0.9635 |
| **Average Precision (AP)** | 0.4205 | 0.3818 | **0.7892** | 0.7642 |
| **True Positives (TP)** | 7 | **56** | 36 | 30 |
| **False Positives (FP)** | 4 | 339 | 4 | **1** |
| **False Negatives (FN)** | 61 | **12** | 32 | 38 |
| **True Negatives (TN)** | 1928 | 1593 | 1928 | **1931** |

---

## 5. Factors, Biases & Technical Limitations
- **Synthetic Benchmark**: The UCI AI4I 2020 dataset is a synthetic benchmark generator. Generalization to actual physical machinery or distinct machine architectures cannot be guaranteed.
- **Extreme Class Imbalance**: With 3.39% minority failure prevalence, unweighted linear models exhibit severe under-detection (10.3% recall). Conversely, balanced linear models produce excessive false alarms (339 FP). Non-linear tree ensembles achieve the best balance.
- **Threshold Policy vs. Physical Probability**: Output probabilities represent statistical score distributions, not calibrated physical failure chances. Threshold 0.5 is an initial baseline default.
- **Temporal & Spatial Dynamics**: The 80/20 stratified split assumes independent records. Real industrial streams have temporal correlation and drift that require time-series validation in live environments.

---

## 6. Model Artifact Locations
- `models/baseline_logistic_regression.joblib`
- `models/baseline_logistic_regression_balanced.joblib`
- `models/baseline_random_forest.joblib`
- `models/baseline_random_forest_balanced.joblib`
- `outputs/metrics/baseline_evaluation_report.json`
- `outputs/metrics/baseline_models_summary.csv`
- `outputs/metrics/confusion_matrices.json`

---

## 7. Operational Alert Dispatch Policy (`src/alert_system.py`)
- **Tri-Level Maintenance Alerting**: Model risk scores in $[0.0, 1.0]$ are evaluated against configurable operational thresholds:
  - $\text{score} < 0.3$: **`NORMAL`** (standard operational monitoring; no intervention required).
  - $0.3 \le \text{score} < 0.7$: **`WARNING`** (elevated degradation risk; schedule non-urgent technical inspection).
  - $\text{score} \ge 0.7$: **`CRITICAL`** (high estimated risk score; prioritized maintenance review and technical inspection recommended).
  - Invalid scores, NaNs, infinities, or failed inference: **`INVALID`** (data quality alert; never defaults to normal).
- **Illustrative Threshold Disclaimer**: Warning (0.3) and Critical (0.7) thresholds are demonstration defaults. Real-world deployment requires cost-sensitive threshold calibration matching plant-specific downtime risks and inspection budgets.

---

## 8. IoT Telemetry Simulation & Testing (`src/simulator.py`)
- **Software Simulation Scope**: Phase 7 implements a reproducible software simulator for generating multivariate synthetic sensor streams (`Type`, temperatures, speed, torque, wear) across nominal, progressive degradation, mixed, and fault-injection operational regimes.
- **Inference & Alert Dispatch Integration**: Simulated events are passed through the existing model inference and alert dispatching components to evaluate streaming behavior without modifying models or retraining.
- **Physical Boundary Disclaimer**: The simulator provides a controlled evaluation testbed. Generated signals do not represent live physical telemetry from certified industrial machinery, and the ML models classify patterns learned from the synthetic UCI AI4I benchmark rather than forecasting actual physical equipment lifespans.


