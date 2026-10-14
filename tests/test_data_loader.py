"""Unit and Integration tests for data_loader module."""

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader

# Sample valid record matching UCI AI4I 2020 schema
SAMPLE_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


@pytest.fixture
def valid_fixture_csv(tmp_path: Path) -> Path:
    """Create a small, valid temporary CSV fixture with known rows."""
    file_path = tmp_path / "valid_sample.csv"
    data = [
        [1, "M14860", "M", 298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0],
        [2, "L47181", "L", 298.2, 308.7, 1408, 46.3, 3, 0, 0, 0, 0, 0, 0],
        [3, "H29424", "H", 298.5, 309.0, 1498, 49.4, 5, 1, 1, 0, 0, 0, 0],
    ]
    df = pd.DataFrame(data, columns=SAMPLE_COLUMNS)
    df.to_csv(file_path, index=False, encoding="utf-8")
    return file_path


def test_load_raw_data_success(valid_fixture_csv: Path) -> None:
    """Verify successful loading of small, valid temporary CSV fixture."""
    df = data_loader.load_raw_data(valid_fixture_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == SAMPLE_COLUMNS
    assert df.loc[0, "Product ID"] == "M14860"
    assert df.loc[2, "Machine failure"] == 1


def test_load_raw_data_missing_file(tmp_path: Path) -> None:
    """Verify handling when the file does not exist."""
    missing_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset file not found"):
        data_loader.load_raw_data(missing_path)


def test_load_raw_data_empty_file(tmp_path: Path) -> None:
    """Verify handling when the file is 0 bytes."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        data_loader.load_raw_data(empty_file)


def test_validate_schema_success(valid_fixture_csv: Path) -> None:
    """Verify schema validation on a valid fixture."""
    df = data_loader.load_raw_data(valid_fixture_csv)
    results = data_loader.validate_schema(df)
    assert results["is_valid"] is True
    assert results["row_count"] == 3
    assert results["col_count"] == 14
    assert len(results["missing_columns"]) == 0
    assert results["total_missing"] == 0
    assert results["duplicate_rows"] == 0
    assert results["target_present"] is True
    assert results["target_distribution"] == {"0": 2, "1": 1}


def test_validate_schema_missing_required_column(tmp_path: Path) -> None:
    """Verify detection and error when a required sensor column is missing."""
    file_path = tmp_path / "missing_col.csv"
    subset_cols = [c for c in SAMPLE_COLUMNS if c != "Torque [Nm]"]
    df = pd.DataFrame([[1, "M14860", "M", 298.1, 308.6, 1551, 0, 0, 0, 0, 0, 0, 0]], columns=subset_cols)
    df.to_csv(file_path, index=False)
    loaded_df = pd.read_csv(file_path)

    with pytest.raises(ValueError, match="Missing required columns.*Torque \\[Nm\\]"):
        data_loader.validate_schema(loaded_df)


def test_validate_schema_missing_target_column(tmp_path: Path) -> None:
    """Verify detection and error when the target column is missing."""
    file_path = tmp_path / "missing_target.csv"
    subset_cols = [c for c in SAMPLE_COLUMNS if c != "Machine failure"]
    df = pd.DataFrame([[1, "M14860", "M", 298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0]], columns=subset_cols)
    df.to_csv(file_path, index=False)
    loaded_df = pd.read_csv(file_path)

    with pytest.raises(ValueError, match="Expected target column 'Machine failure' is missing"):
        data_loader.validate_schema(loaded_df, target_column="Machine failure")


def test_missing_value_detection(tmp_path: Path) -> None:
    """Verify accurate detection and reporting of null/missing values."""
    file_path = tmp_path / "with_nulls.csv"
    data = [
        [1, "M14860", "M", None, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0],
        [2, "L47181", None, 298.2, None, 1408, 46.3, 3, 0, 0, 0, 0, 0, 0],
    ]
    df = pd.DataFrame(data, columns=SAMPLE_COLUMNS)
    df.to_csv(file_path, index=False)
    loaded_df = data_loader.load_raw_data(file_path)

    results = data_loader.validate_schema(loaded_df)
    assert results["total_missing"] == 3
    assert results["missing_values"]["Air temperature [K]"] == 1
    assert results["missing_values"]["Type"] == 1
    assert results["missing_values"]["Process temperature [K]"] == 1


def test_duplicate_row_detection(tmp_path: Path) -> None:
    """Verify detection of duplicate records."""
    file_path = tmp_path / "duplicates.csv"
    row = [1, "M14860", "M", 298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0]
    df = pd.DataFrame([row, row, row], columns=SAMPLE_COLUMNS)
    df.to_csv(file_path, index=False)
    loaded_df = data_loader.load_raw_data(file_path)

    results = data_loader.validate_schema(loaded_df)
    assert results["row_count"] == 3
    assert results["duplicate_rows"] == 2


def test_analyze_data_quality(valid_fixture_csv: Path) -> None:
    """Verify statistical profiling and range checks in data quality analysis."""
    df = data_loader.load_raw_data(valid_fixture_csv)
    quality = data_loader.analyze_data_quality(df)

    assert quality["shape"] == [3, 14]
    assert "Air temperature [K]" in quality["numerical_summary"]
    assert quality["numerical_summary"]["Air temperature [K]"]["min"] == 298.1
    assert quality["categorical_summary"]["Type"]["unique_count"] == 3
    assert quality["failure_modes_summary"]["TWF"]["positive_count"] == 1
    assert quality["failure_rate"] == round(1 / 3, 6)


def test_generate_validation_report(valid_fixture_csv: Path, tmp_path: Path) -> None:
    """Verify end-to-end report generation and JSON output serialization."""
    out_json = tmp_path / "sub_dir" / "test_report.json"
    report = data_loader.generate_validation_report(
        file_path=valid_fixture_csv,
        output_path=out_json,
    )

    assert out_json.is_file()
    assert report["row_count"] == 3
    assert report["column_count"] == 14
    assert report["is_valid"] is True
    assert "sha256_checksum" in report
    assert report["file_size_bytes"] > 0


def test_real_dataset_if_present() -> None:
    """Integration check on actual data/raw/ai4i2020.csv if present."""
    default_path = data_loader.get_raw_data_path()
    if not default_path.is_file():
        pytest.skip("Official dataset data/raw/ai4i2020.csv not present.")

    df = data_loader.load_raw_data(default_path)
    assert len(df) == 10000
    assert len(df.columns) == 14
    assert int(df.isnull().sum().sum()) == 0
    assert int(df.duplicated().sum()) == 0

    schema_info = data_loader.validate_schema(df)
    assert schema_info["is_valid"] is True
    assert schema_info["target_distribution"] == {"0": 9661, "1": 339}
    assert schema_info["failure_rate"] == 0.0339
