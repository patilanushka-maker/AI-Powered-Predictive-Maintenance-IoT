# Dataset Documentation: UCI AI4I 2020 Predictive Maintenance

---

## 1. Dataset Origin & Retrieval
- **Dataset Name**: AI4I 2020 Predictive Maintenance Dataset
- **Official Source**: [UCI Machine Learning Repository (ID: 601)](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B)
- **Direct Archive**: `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
- **Authors**: Stephan Matzka (School of Engineering - Technology and Life, HTW Berlin, 2020)
- **Nature of Data**: Synthetic dataset reflecting real-world milling machine degradation processes.
- **Local Storage Path**: `data/raw/ai4i2020.csv` *(Git-ignored)*
- **File Size**: 522,048 bytes
- **SHA-256 Checksum**: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
- **Validation Artifact**: [`outputs/metrics/data_validation_report.json`](../outputs/metrics/data_validation_report.json)

---

## 2. Verified Dataset Schema & Empirical Distributions

The downloaded file was verified during Phase 2 via [`src/data_loader.py`](../src/data_loader.py). The dataset contains exactly **10,000 rows** and **14 columns** with **0 missing values** and **0 duplicate rows**.

| Column Name | Inferred Dtype | Missing | Physical Meaning / Description | Empirical Range / Distribution |
| :--- | :--- | :--- | :--- | :--- |
| `UDI` | `int64` | 0 | Unique row identifier (1 to 10,000) | Min: 1, Max: 10,000 |
| `Product ID` | `object` | 0 | Product variant character + serial number | 10,000 unique values |
| `Type` | `object` | 0 | Product quality variant | L: 6,000 (60.0%), M: 2,997 (30.0%), H: 1,003 (10.0%) |
| `Air temperature [K]` | `float64` | 0 | Ambient air temperature | Mean: 300.00 K, Min: 295.3 K, Max: 304.5 K |
| `Process temperature [K]`| `float64` | 0 | Internal process operating temperature | Mean: 310.01 K, Min: 305.7 K, Max: 313.8 K |
| `Rotational speed [rpm]` | `int64` | 0 | Spindle rotational speed | Mean: 1,538.8 rpm, Min: 1,168 rpm, Max: 2,886 rpm |
| `Torque [Nm]` | `float64` | 0 | Spindle resistance torque | Mean: 39.99 Nm, Min: 3.8 Nm, Max: 76.6 Nm |
| `Tool wear [min]` | `int64` | 0 | Accumulated cutting tool wear in minutes | Mean: 107.95 min, Min: 0 min, Max: 253 min |
| `Machine failure` | `int64` | 0 | **Primary Binary Target** (0: Normal, 1: Failure) | `0`: 9,661 (96.61%), `1`: 339 (**3.39%**) |
| `TWF` | `int64` | 0 | Tool Wear Failure mode | 46 positive cases (0.46%) |
| `HDF` | `int64` | 0 | Heat Dissipation Failure mode | 115 positive cases (1.15%) |
| `PWF` | `int64` | 0 | Power Failure mode | 95 positive cases (0.95%) |
| `OSF` | `int64` | 0 | Overstrain Failure mode | 98 positive cases (0.98%) |
| `RNF` | `int64` | 0 | Random Failure mode | 19 positive cases (0.19%) |

---

## 3. Class Imbalance & Sensor Range Observations
- **Severe Class Imbalance**: With only 339 positive cases (3.39%) out of 10,000 cycles (~28.5 to 1 ratio), standard accuracy is a misleading metric. Evaluation in Phase 3 must prioritize **PR-AUC, Recall, Precision, and F1-Score**.
- **Physical Sensor Consistency**:
  - All temperatures are strictly positive Kelvin values (min air temp 295.3 K = 22.15 °C; min process temp 305.7 K = 32.55 °C).
  - Rotational speeds (1,168 to 2,886 rpm) and torques (3.8 to 76.6 Nm) remain within realistic industrial cutting bounds.
  - Thermal delta (`Process temperature - Air temperature`) is positive across all 10,000 records (~10 K differential).

---

## 4. Failure Mode Discrepancies & Dataset Nuances
The dataset contains two specific nuances uncovered during data-quality analysis:
1. **Unspecified Failures (`Machine failure == 1` & all modes `== 0`)**: Exactly **9 records** experienced failure without any of the five specific failure mode flags being set.
2. **Non-fatal Random Failures (`Machine failure == 0` & modes `> 0`)**: Exactly **18 records** have `RNF == 1` while `Machine failure == 0`. Per the dataset documentation, RNF events have a small probability of occurring without causing a complete machine shutdown.

---

## 5. Target Leakage Prevention & Baseline Feature Policy

To build a reliable predictive maintenance model, strict feature selection boundaries must be enforced:

### Primary Prediction Target
* **`Machine failure`** (Binary: 0 for healthy operating state, 1 for equipment failure).

### Columns Strictly Excluded from Baseline Predictors
1. **`UDI`**: An artificial integer index (1 to 10,000). It carries no physical information about machinery state and would induce artificial time/order bias.
2. **`Product ID`**: A categorical serial number with 10,000 unique values (`M14860`, `L47181`, etc.). Using this would result in extreme overfitting and spurious memorization.
3. **`TWF`, `HDF`, `PWF`, `OSF`, `RNF`**: These are **post-hoc failure mode diagnostics**. If included as input features, they cause catastrophic **target leakage** because a failure diagnosis is only known *after* a failure has occurred.

### Permitted Baseline Predictive Features
The valid operational telemetry signals permitted for baseline and future model training are:
* **`Type`**: Machine variant quality category (L, M, H).
* **`Air temperature [K]`**: Ambient thermal state.
* **`Process temperature [K]`**: Core operating temperature.
* **`Rotational speed [rpm]`**: Spindle mechanical speed.
* **`Torque [Nm]`**: Mechanical load and resistance.
* **`Tool wear [min]`**: Cumulative wear stress indicator.

---

## 6. Preprocessing & Partitioning Architecture (Phase 3)

### Train/Test Split Configuration
- **Split Ratio**: 80% Training (8,000 instances), 20% Testing (2,000 instances).
- **Random Seed**: `random_state=42` for strict mathematical reproducibility.
- **Stratification**: Stratified by `Machine failure` to ensure both subsets preserve identical class balance:
  - **Training Set**: 7,729 Class 0 (96.61%), 271 Class 1 (3.39%).
  - **Testing Set**: 1,932 Class 0 (96.60%), 68 Class 1 (3.40%).

### Transformation Pipelines
1. **Numerical Features** (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`):
   - Imputation: `SimpleImputer(strategy="median")`.
   - Normalization: `StandardScaler()` (scales features to mean = 0, variance = 1).
2. **Categorical Features** (`Type`):
   - Imputation: `SimpleImputer(strategy="most_frequent")`.
   - Encoding: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` producing 3 binary columns (`Type_H`, `Type_L`, `Type_M`). Handles unexpected categories at test/inference time without raising exceptions.
3. **Combined Output Dimensions**: 8 transformed features (5 numerical + 3 one-hot columns).

### Strict Leakage Prevention Safeguards
1. **Split-Before-Fit**: The dataset is split into `X_train`, `X_test`, `y_train`, `y_test` *before* instantiating and fitting any imputer, scaler, or encoder.
2. **Train-Only Fitting**: The `ColumnTransformer` is fitted **strictly on `X_train`**. Preprocessing statistics (means, standard deviations, medians, one-hot category vocabularies) are learned solely from the training distribution.
3. **Transform-Only Test**: `X_test` is transformed using the parameters previously fitted on `X_train`. No test data statistics ever leak into the transformation pipeline.
4. **Target Isolation**: The target `y` is never scaled, imputed, or passed into the feature transformer.

### Class Imbalance & Absence of Resampling
Resampling techniques (such as SMOTE, ADASYN, or Random Under-Sampling) have deliberately **not** been applied during preprocessing. Applying global resampling prior to cross-validation introduces artificial synthetic artifacts and invalidates test set evaluation. Any cost-sensitive learning or training-fold resampling will be evaluated during model training in Phase 4.

### Limitations of Random Splitting
The 80/20 stratified random split provides an initial baseline evaluation framework under an i.i.d. assumption. In physical industrial deployments, sensor telemetry often exhibits serial correlation and time-dependent degradation. An i.i.d. random split may yield slightly optimistic performance estimates compared to time-series forward-chaining splits or machine-held-out cross-validation. This limitation will be considered during model evaluation.

### Verification Commands
```bash
# Execute preprocessing validation runner
python main.py --mode preprocess

# Execute baseline model training and evaluation (Phase 4)
python main.py --mode train
python main.py --mode evaluate

# Run test suite
pytest tests/ -v
```

> **Status Notice**: Phase 4 baseline classification models have been trained and evaluated on the untouched test partition. See [docs/04_model_architecture_and_training.md](04_model_architecture_and_training.md) and [docs/06_validation_and_benchmarks.md](06_validation_and_benchmarks.md) for full benchmarks. Prediction and telemetry simulation will be implemented in **Phase 5**.

