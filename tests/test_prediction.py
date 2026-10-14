"""Unit and Integration tests for inference and prediction engine (Phase 5)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import predict, train_model
from src.preprocessing import BASELINE_PREDICTORS


@pytest.fixture
def valid_single_record() -> dict[str, Any]:
    """A single clean, valid telemetry record with 6 permitted predictors."""
    return {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }


@pytest.fixture
def small_fitted_pipeline() -> Pipeline:
    """Fixture providing a minimal, fast-fitted pipeline on synthetic data for unit testing."""
    df_train = pd.DataFrame({
        "Type": ["L", "M", "H", "L", "M"],
        "Air temperature [K]": [298.0, 299.0, 300.0, 301.0, 302.0],
        "Process temperature [K]": [308.0, 309.0, 310.0, 311.0, 312.0],
        "Rotational speed [rpm]": [1500.0, 1520.0, 1540.0, 1560.0, 1580.0],
        "Torque [Nm]": [40.0, 41.0, 42.0, 43.0, 44.0],
        "Tool wear [min]": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    y_train = pd.Series([0, 0, 0, 1, 1])
    clf = LogisticRegression(max_iter=500, random_state=42)
    pipeline = train_model.train_single_pipeline(clf, df_train, y_train)
    return pipeline


# 1. Successful loading of saved model pipeline
def test_load_inference_pipeline_success(small_fitted_pipeline: Pipeline, tmp_path: Path):
    """Verify loading of a serialized model pipeline from disk."""
    save_path = tmp_path / "test_pipeline.joblib"
    train_model.save_pipeline(small_fitted_pipeline, save_path)

    loaded_pipe, model_name = predict.load_inference_pipeline(save_path)
    assert isinstance(loaded_pipe, Pipeline)
    assert "preprocessor" in loaded_pipe.named_steps
    assert "classifier" in loaded_pipe.named_steps
    assert model_name == "test_pipeline"


# 2. Missing model artifact error handling
def test_load_inference_pipeline_missing_file(tmp_path: Path):
    """Verify FileNotFoundError with clear error message when model file does not exist."""
    missing_path = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        predict.load_inference_pipeline(missing_path)


# 3. Missing required predictor fields
def test_validate_record_missing_field(valid_single_record: dict[str, Any]):
    """Verify ValueError when any required predictor column is missing."""
    incomplete_record = dict(valid_single_record)
    del incomplete_record["Torque [Nm]"]

    with pytest.raises(ValueError, match="Missing required predictor field"):
        predict.validate_prediction_record(incomplete_record)


# 4. Unexpected or forbidden input columns
def test_validate_record_forbidden_columns(valid_single_record: dict[str, Any]):
    """Verify rejection of forbidden columns (identifiers, failure modes, target)."""
    # Test post-hoc failure mode
    record_with_twf = dict(valid_single_record, TWF=1)
    with pytest.raises(ValueError, match="Forbidden column.*TWF"):
        predict.validate_prediction_record(record_with_twf)

    # Test row identifier
    record_with_udi = dict(valid_single_record, UDI=101)
    with pytest.raises(ValueError, match="Forbidden column.*UDI"):
        predict.validate_prediction_record(record_with_udi)

    # Test target column
    record_with_target = dict(valid_single_record, **{"Machine failure": 0})
    with pytest.raises(ValueError, match="Forbidden column.*Machine failure"):
        predict.validate_prediction_record(record_with_target)

    # Test unexpected random column
    record_with_extra = dict(valid_single_record, extra_sensor=42.0)
    with pytest.raises(ValueError, match="Unexpected field"):
        predict.validate_prediction_record(record_with_extra)


# 5. Invalid categorical value for 'Type'
def test_validate_record_invalid_type(valid_single_record: dict[str, Any]):
    """Verify ValueError when machine 'Type' is not 'L', 'M', or 'H'."""
    invalid_record = dict(valid_single_record, Type="SPECIAL_GRADE")
    with pytest.raises(ValueError, match="Invalid machine 'Type' value"):
        predict.validate_prediction_record(invalid_record)

    non_string_type = dict(valid_single_record, Type=123)
    with pytest.raises(ValueError, match="Invalid machine 'Type' data type"):
        predict.validate_prediction_record(non_string_type)


# 6. Non-numeric sensor value
def test_validate_record_non_numeric_values(valid_single_record: dict[str, Any]):
    """Verify rejection of non-numeric values, strings, and booleans."""
    record_str = dict(valid_single_record, **{"Rotational speed [rpm]": "fast"})
    with pytest.raises(ValueError, match="Non-numeric value for sensor"):
        predict.validate_prediction_record(record_str)

    record_bool = dict(valid_single_record, **{"Torque [Nm]": True})
    with pytest.raises(ValueError, match="Invalid boolean value"):
        predict.validate_prediction_record(record_bool)

    record_none = dict(valid_single_record, **{"Tool wear [min]": None})
    with pytest.raises(ValueError, match="is None or missing"):
        predict.validate_prediction_record(record_none)


# 7. NaN and infinite values
def test_validate_record_nan_and_inf(valid_single_record: dict[str, Any]):
    """Verify rejection of NaN and infinity."""
    record_nan = dict(valid_single_record, **{"Air temperature [K]": float("nan")})
    with pytest.raises(ValueError, match="Non-finite value.*NaN"):
        predict.validate_prediction_record(record_nan)

    record_inf = dict(valid_single_record, **{"Torque [Nm]": float("inf")})
    with pytest.raises(ValueError, match="Non-finite value.*Infinity"):
        predict.validate_prediction_record(record_inf)


# 8. Negative values for non-negative sensor fields
def test_validate_record_negative_sensor_values(valid_single_record: dict[str, Any]):
    """Verify rejection of physically invalid negative physical values."""
    record_neg_speed = dict(valid_single_record, **{"Rotational speed [rpm]": -100})
    with pytest.raises(ValueError, match="Rotational speed cannot be negative"):
        predict.validate_prediction_record(record_neg_speed)

    record_neg_torque = dict(valid_single_record, **{"Torque [Nm]": -5.0})
    with pytest.raises(ValueError, match="Torque cannot be negative"):
        predict.validate_prediction_record(record_neg_torque)

    record_neg_wear = dict(valid_single_record, **{"Tool wear [min]": -1})
    with pytest.raises(ValueError, match="Tool wear cannot be negative"):
        predict.validate_prediction_record(record_neg_wear)

    record_neg_temp = dict(valid_single_record, **{"Air temperature [K]": -50.0})
    with pytest.raises(ValueError, match="Air temperature in Kelvin must be strictly positive"):
        predict.validate_prediction_record(record_neg_temp)


# 9. Invalid threshold
def test_validate_threshold():
    """Verify threshold boundary validation (strictly in 0.0 to 1.0)."""
    assert predict.validate_threshold(0.5) == 0.5
    assert predict.validate_threshold(0.01) == 0.01
    assert predict.validate_threshold(0.99) == 0.99

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        predict.validate_threshold(0.0)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        predict.validate_threshold(1.0)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        predict.validate_threshold(-0.2)

    with pytest.raises(ValueError, match="Threshold must be a numeric float"):
        predict.validate_threshold("0.5")  # string rejected


# 10. Successful prediction for a valid single record
def test_predict_single_success(small_fitted_pipeline: Pipeline, valid_single_record: dict[str, Any]):
    """Verify that predict_single returns complete, valid schema."""
    result = predict.predict_single(
        record=valid_single_record,
        model=small_fitted_pipeline,
        threshold=0.5,
    )

    required_keys = {
        "model_name",
        "predicted_class",
        "predicted_label",
        "positive_class_score",
        "threshold",
        "status",
        "warnings",
    }
    assert required_keys.issubset(set(result.keys()))
    assert result["status"] == "success"
    assert result["predicted_class"] in {0, 1}
    assert result["predicted_label"] == predict.LABEL_MAPPING[result["predicted_class"]]
    assert result["threshold"] == 0.5


# 11. Positive-class score lies in valid range [0, 1]
def test_positive_class_score_range(small_fitted_pipeline: Pipeline, valid_single_record: dict[str, Any]):
    """Verify that positive_class_score is a probability strictly in [0.0, 1.0]."""
    result = predict.predict_single(
        record=valid_single_record,
        model=small_fitted_pipeline,
    )
    score = result["positive_class_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# 12. Predicted class is consistent with score and threshold
def test_predicted_class_threshold_consistency(
    small_fitted_pipeline: Pipeline, valid_single_record: dict[str, Any]
):
    """Verify predicted_class matches threshold comparison logic."""
    res_default = predict.predict_single(valid_single_record, model=small_fitted_pipeline, threshold=0.5)
    score = res_default["positive_class_score"]

    # If threshold > score, must predict 0
    high_threshold = min(0.99, max(score + 0.1, 0.9))
    res_high = predict.predict_single(valid_single_record, model=small_fitted_pipeline, threshold=high_threshold)
    if score < high_threshold:
        assert res_high["predicted_class"] == 0
        assert res_high["predicted_label"] == "No failure label predicted"

    # If threshold <= score, must predict 1
    low_threshold = max(0.01, min(score - 0.1, 0.1))
    res_low = predict.predict_single(valid_single_record, model=small_fitted_pipeline, threshold=low_threshold)
    if score >= low_threshold:
        assert res_low["predicted_class"] == 1
        assert res_low["predicted_label"] == "Failure label predicted"


# 13. Correct handling of positive-class probability column
def test_extract_positive_class_score_dynamic_ordering(small_fitted_pipeline: Pipeline):
    """Verify positive class score correctly matches class 1 label index."""
    df_test = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 20.0,
    }])
    scores = predict.extract_positive_class_score(small_fitted_pipeline, df_test)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


# 14. Batch predictions preserve row order
def test_predict_batch_order_preservation(small_fitted_pipeline: Pipeline):
    """Verify that batch predictions strictly maintain input order."""
    records = [
        {"Type": "L", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0, "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 10},
        {"Type": "M", "Air temperature [K]": 299.0, "Process temperature [K]": 309.0, "Rotational speed [rpm]": 1520, "Torque [Nm]": 41.0, "Tool wear [min]": 20},
        {"Type": "H", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1540, "Torque [Nm]": 42.0, "Tool wear [min]": 30},
    ]

    results = predict.predict_batch(records, model=small_fitted_pipeline)
    assert len(results) == 3
    assert [r["row_index"] for r in results] == [0, 1, 2]


# 15. Batch validation errors follow documented policy
def test_predict_batch_error_policies(small_fitted_pipeline: Pipeline):
    """Verify 'raise' policy raises ValueError and 'per_record' policy isolates errors."""
    valid_row = {"Type": "L", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0, "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 10}
    invalid_row = {"Type": "INVALID", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0, "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 10}

    # Policy 'raise': Must raise ValueError mentioning row 1
    with pytest.raises(ValueError, match="Batch validation failed at row 1"):
        predict.predict_batch([valid_row, invalid_row], model=small_fitted_pipeline, error_policy="raise")

    # Policy 'per_record': Must return results for both rows with row 1 marked as error
    results = predict.predict_batch([valid_row, invalid_row], model=small_fitted_pipeline, error_policy="per_record")
    assert len(results) == 2
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "error"
    assert "Invalid machine 'Type'" in results[1]["error"]


# 16. Input data remains unchanged (immutability)
def test_input_immutability(small_fitted_pipeline: Pipeline, valid_single_record: dict[str, Any]):
    """Verify that caller's input dictionary and DataFrame are not mutated."""
    orig_copy = dict(valid_single_record)
    predict.predict_single(valid_single_record, model=small_fitted_pipeline)
    assert valid_single_record == orig_copy

    # Test DataFrame input
    df_input = pd.DataFrame([orig_copy])
    df_copy = df_input.copy()
    predict.predict_batch(df_input, model=small_fitted_pipeline)
    pd.testing.assert_frame_equal(df_input, df_copy)


# 17. Saved and reloaded pipelines produce equivalent predictions
def test_saved_and_reloaded_pipeline_equivalence(
    small_fitted_pipeline: Pipeline, valid_single_record: dict[str, Any], tmp_path: Path
):
    """Verify that serializing and deserializing a pipeline produces identical scores."""
    save_path = tmp_path / "equivalence_model.joblib"
    train_model.save_pipeline(small_fitted_pipeline, save_path)

    loaded_pipe, _ = predict.load_inference_pipeline(save_path)

    res_orig = predict.predict_single(valid_single_record, model=small_fitted_pipeline)
    res_loaded = predict.predict_single(valid_single_record, model=loaded_pipe)

    assert res_orig["positive_class_score"] == res_loaded["positive_class_score"]
    assert res_orig["predicted_class"] == res_loaded["predicted_class"]


# 18. Prediction does not invoke training
def test_prediction_does_not_invoke_training(
    small_fitted_pipeline: Pipeline, valid_single_record: dict[str, Any]
):
    """Verify that neither fit nor partial_fit is ever called during prediction."""
    mock_fit = MagicMock()
    small_fitted_pipeline.fit = mock_fit
    small_fitted_pipeline.named_steps["classifier"].fit = mock_fit

    predict.predict_single(valid_single_record, model=small_fitted_pipeline)
    mock_fit.assert_not_called()


# 19. Integration check: Loaded Phase 4 model predicts from raw records
def test_real_saved_random_forest_inference(valid_single_record: dict[str, Any]):
    """Integration test verifying inference using the real Phase 4 reference Random Forest artifact."""
    rf_path = predict.DEFAULT_MODELS_DIR / predict.DEFAULT_REFERENCE_MODEL_FILENAME
    if not rf_path.is_file():
        pytest.skip(f"Phase 4 reference model not found at: {rf_path}")

    res = predict.predict(valid_single_record)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert res["model_name"] == "Random Forest (Unweighted)"
    assert res["predicted_class"] in {0, 1}
    assert 0.0 <= res["positive_class_score"] <= 1.0


# 20. Domain plausibility warnings generation
def test_domain_plausibility_warning(small_fitted_pipeline: Pipeline):
    """Verify domain plausibility warnings are emitted for inverted thermal gradient."""
    inverted_thermal_record = {
        "Type": "L",
        "Air temperature [K]": 305.0,
        "Process temperature [K]": 298.0,  # Process < Air!
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
    }
    result = predict.predict_single(inverted_thermal_record, model=small_fitted_pipeline)
    assert result["status"] == "success"
    assert len(result["warnings"]) > 0
    assert any("Inverted thermal gradient" in w for w in result["warnings"])


# 21. CLI prediction integration test
def test_cli_prediction_success(monkeypatch: pytest.MonkeyPatch):
    """Verify that CLI predict mode succeeds on valid sample CSV."""
    import main

    sample_csv = Path(__file__).resolve().parent.parent / "data" / "sample_inference_records.csv"
    if not sample_csv.is_file():
        pytest.skip("sample_inference_records.csv not found")

    test_args = ["main.py", "--mode", "predict", "--input", str(sample_csv)]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = main.main()
    assert exit_code == 0


# 22. CLI error handling on missing model and invalid data
def test_cli_prediction_error_handling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify that CLI predict mode exits with code 1 for missing models and invalid inputs."""
    import main

    # 1. Missing model
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "sample_inference_records.csv"
    test_args_missing_model = [
        "main.py",
        "--mode",
        "predict",
        "--input",
        str(sample_csv),
        "--model",
        "non_existent_model_binary",
    ]
    monkeypatch.setattr("sys.argv", test_args_missing_model)
    assert main.main() == 1

    # 2. Invalid data CSV
    invalid_csv = tmp_path / "invalid_records.csv"
    invalid_csv.write_text("Type,Air temperature [K]\nINVALID,300\n", encoding="utf-8")
    test_args_invalid_data = ["main.py", "--mode", "predict", "--input", str(invalid_csv)]
    monkeypatch.setattr("sys.argv", test_args_invalid_data)
    assert main.main() == 1

