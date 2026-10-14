# 06. Validation & Benchmarking Protocols

---

## 1. Phase 4 Baseline Evaluation Protocol

All models in Phase 4 were evaluated on the exact same held-out test split of 2,000 samples (1,932 nominal operating cycles and 68 machine failure events; 3.40% failure prevalence).

### Evaluation Boundaries
* **Default Decision Threshold**: `threshold = 0.5` applied consistently across all models for initial baseline comparison.
* **Positive Class**: Evaluated strictly on the minority failure class (`Machine failure = 1`).
* **Primary Success Metrics**: PR-AUC (Average Precision), F1-Score, Recall, Precision, and ROC-AUC. Accuracy is reported for completeness but not treated as a primary success metric due to extreme class imbalance (~28.5:1 ratio).
* **Test Set Isolation**: The test set was NOT used for hyperparameter tuning or threshold optimization.

---

## 2. Empirical Benchmark Comparison (Threshold = 0.5)

The table below presents the verified empirical performance measured on the untouched test partition (`N = 2,000`):

| Model Configuration | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Avg Prec) | Accuracy* | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Unweighted)** | **0.9000** | 0.5294 | **0.6667** | **0.9653** | **0.7892** | 0.9820 | 36 | 4 | 1928 | 32 |
| **Random Forest (Balanced)** | **0.9677** | 0.4412 | 0.6061 | 0.9635 | 0.7642 | 0.9805 | 30 | **1** | **1931** | 38 |
| **Logistic Regression (Balanced)**| 0.1418 | **0.8235** | 0.2419 | 0.9070 | 0.3818 | 0.8245 | **56** | 339 | 1593 | **12** |
| **Logistic Regression (Unweighted)**| 0.6364 | 0.1029 | 0.1772 | 0.8994 | 0.4205 | 0.9675 | 7 | 4 | 1928 | 61 |

*\*Accuracy is shown as a secondary metric; high raw accuracy in imbalanced contexts can mask poor minority-class recall.*

---

## 3. Confusion Matrix Breakdown

### 1. Random Forest (Unweighted)
* **TP: 36** | **FP: 4**
* **FN: 32** | **TN: 1928**
* *Observation*: Achieves the highest baseline F1-score (0.6667) and highest PR-AUC (0.7892). Successfully detects 36 of 68 failures with only 4 false alarms out of 1,932 healthy cycles.

### 2. Random Forest (Balanced)
* **TP: 30** | **FP: 1**
* **FN: 38** | **TN: 1931**
* *Observation*: Demonstrates exceptional precision (96.77%) with only a single false alarm across 2,000 cycles, but exhibits lower recall (44.12%) than the unweighted tree ensemble at the 0.5 threshold.

### 3. Logistic Regression (Balanced)
* **TP: 56** | **FP: 339**
* **FN: 12** | **TN: 1593**
* *Observation*: Captures 82.35% of all failure events (56 of 68), missing only 12 failures. However, the aggressive minority weighting causes 339 false alarms, degrading precision to 14.18%. In an industrial context, this configuration acts as a high-sensitivity triage filter.

### 4. Logistic Regression (Unweighted)
* **TP: 7** | **FP: 4**
* **FN: 61** | **TN: 1928**
* *Observation*: Severely impaired by the 28.5:1 class imbalance. Without weighting, the linear decision surface classifies almost all instances into the majority class, capturing only 7 of 68 failures (10.29% recall).

---

## 4. Key Engineering Insights

1. **Class Weighting Dynamics**:
   - For linear models (Logistic Regression), balancing weights dramatically elevates recall ($10.3\% \rightarrow 82.4\%$) at the expense of high false positive rates ($4 \rightarrow 339$).
   - For non-linear tree ensembles (Random Forest), both configurations achieve outstanding discriminatory power ($\text{ROC-AUC} \approx 0.964\text{--}0.965$ and $\text{PR-AUC} \approx 0.764\text{--}0.789$).
2. **Linear vs Non-Linear Boundaries**:
   - The failure phenomena in AI4I 2020 involve multi-variable interactions (e.g., torque vs. rotational speed power envelope, tool wear thresholds). Tree-based ensembles naturally partition this space far better than a simple linear hyperplane.
3. **No Resampling Baseline Established**:
   - Strong PR-AUC figures were attained purely via scikit-learn preprocessing and estimator tuning, without resorting to synthetic oversampling (SMOTE) or undersampling.

---

## 5. Decision Threshold Selection Policy

* The default threshold of `0.5` represents an arbitrary statistical cutoff, not a calibrated failure probability.
* **Separation of Policy**: Threshold tuning must be treated as an operational decision policy balancing the business cost of unplanned downtime (false negative) versus unnecessary maintenance inspection (false positive).
* **Training-Only Threshold Analysis**: Any threshold tuning performed in subsequent phases must utilize only training-set out-of-fold cross-validation (`cross_val_predict`), keeping the held-out test split strictly quarantined.

---

## 6. Artifact & Report Locations

* **JSON Benchmark Report**: [`outputs/metrics/baseline_evaluation_report.json`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/outputs/metrics/baseline_evaluation_report.json)
* **CSV Comparative Summary**: [`outputs/metrics/baseline_models_summary.csv`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/outputs/metrics/baseline_models_summary.csv)
* **Confusion Matrices**: [`outputs/metrics/confusion_matrices.json`](file:///d:/Anushka/IIT/AI/AI-Powered%20Predictive%20Maintenance%20for%20IoT%20Devices/outputs/metrics/confusion_matrices.json)
