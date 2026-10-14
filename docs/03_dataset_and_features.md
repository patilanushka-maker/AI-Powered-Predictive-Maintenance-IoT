# 03. Dataset & Feature Engineering

## Dataset Overview
Specifications for chosen industrial IoT benchmark datasets (e.g., NASA C-MAPSS turbofan degradation, AI4I 2020 Predictive Maintenance Dataset, or custom sensor streams).

### Attributes & Sensor Modalities
- Device identifiers & operating conditions
- Continuous telemetry signals (e.g., thermal, mechanical vibration, acoustic emissions, rotational speed, hydraulic pressure)
- Degradation ground-truth markers (Remaining Useful Life [RUL], failure modes/flags)

### Feature Engineering Strategies
- **Temporal Aggregations**: Rolling statistics (mean, variance, skewness, kurtosis, peak-to-peak).
- **Domain Indicators**: Crest factor, degradation trend slopes, cumulative energy.
- **Scaling & Normalization**: Robust scaling per sensor channel.

*(Detailed schema tables and feature definitions will be added upon dataset selection and processing.)*
