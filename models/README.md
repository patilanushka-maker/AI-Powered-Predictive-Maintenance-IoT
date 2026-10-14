# Models Directory

This directory stores serialized model artifacts, training pipeline objects, and model metadata generated during the training lifecycle.

---

## Artifact Lifecycle & Exclusion Policy
- **Git Exclusion**: Binary model files (`.joblib`, `.pkl`, `.onnx`) are deliberately ignored by `.gitignore`. Serialized artifacts must not be checked into version control.
- **Reproducibility**: All models are programmatically reproducible from source code and deterministic random seed configurations.
- **Phase Status**: In **Phase 1**, no models have been trained or serialized. Model development will occur during **Phase 3 (Model Training & Evaluation)**.

---

## Regeneration Protocol
Once training pipelines are implemented, model binaries can be regenerated locally with:
```bash
python main.py --mode train
```
Expected output artifacts will include:
- `models/failure_classifier.joblib`: Primary binary classification pipeline (imbalance-adjusted).
- `models/failure_mode_classifier.joblib`: Multi-class failure mode classification model.
- `models/pipeline_metadata.json`: Feature schemas, scaler parameters, and training timestamps.
