"""Data Ingestion, Schema Validation, and Data Quality Module.

This module provides reusable functions for locating, loading, schema-validating,
and profiling raw IoT machine sensor telemetry (UCI AI4I 2020 Predictive Maintenance).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Standard Project File Locations
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_DATA_FILENAME = "ai4i2020.csv"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / DEFAULT_DATA_FILENAME

# Expected Schema Definitions
EXPECTED_COLUMNS: list[str] = [
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

TARGET_COLUMN: str = "Machine failure"

IDENTIFIER_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
]

FAILURE_MODE_COLUMNS: list[str] = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

BASELINE_FEATURE_COLUMNS: list[str] = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

DATASET_SOURCE_URL: str = (
    "https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B"
)


def get_raw_data_path(custom_path: Path | str | None = None) -> Path:
    """Resolve the configured raw dataset path.

    Args:
        custom_path: Optional explicit path. If None, defaults to data/raw/ai4i2020.csv.

    Returns:
        Path object pointing to the dataset location.
    """
    if custom_path is not None:
        return Path(custom_path).resolve()
    return DEFAULT_DATA_PATH.resolve()


def load_raw_data(
    file_path: Path | str | None = None,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """Load the raw telemetry dataset into a Pandas DataFrame.

    Does not modify, clean, or impute values in the raw dataset.

    Args:
        file_path: Optional path to the CSV file. Defaults to data/raw/ai4i2020.csv.
        encoding: Text encoding (defaults to 'utf-8-sig' to safely handle BOM).

    Returns:
        pd.DataFrame containing the raw ingested data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is 0 bytes or contains no data rows.
    """
    path = get_raw_data_path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file not found at: {path}. "
            f"Please verify placement or download from {DATASET_SOURCE_URL}."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty (0 bytes): {path}")

    try:
        df = pd.read_csv(path, encoding=encoding)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at {path}: {exc}") from exc

    if df.empty or len(df) == 0:
        raise ValueError(f"Dataset contains no data rows: {path}")

    return df


def validate_schema(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
    target_column: str | None = TARGET_COLUMN,
) -> dict[str, Any]:
    """Validate DataFrame against expected column schema and target constraints.

    Args:
        df: Pandas DataFrame to validate.
        expected_columns: List of required columns (defaults to EXPECTED_COLUMNS).
        target_column: Expected target column name (defaults to 'Machine failure').

    Returns:
        Dictionary containing validation findings, counts, and warnings.

    Raises:
        ValueError: If any expected columns or the target column are missing.
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    actual_columns = list(df.columns)
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    unexpected_columns = [col for col in actual_columns if col not in expected_columns]

    errors: list[str] = []
    warnings: list[str] = []

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if target_column is not None and target_column not in actual_columns:
        errors.append(f"Expected target column '{target_column}' is missing.")

    if unexpected_columns:
        warnings.append(f"Unexpected extra columns detected: {unexpected_columns}")

    # If critical columns are missing, raise ValueError as required by spec
    if errors:
        raise ValueError("; ".join(errors))

    # Calculate missing values
    missing_counts = {str(col): int(count) for col, count in df.isnull().sum().items()}
    total_missing = int(sum(missing_counts.values()))

    # Calculate duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Target class distribution
    target_present = target_column in actual_columns if target_column else False
    target_dist: dict[str, int] | None = None
    target_props: dict[str, float] | None = None
    failure_rate: float | None = None

    if target_present and target_column is not None:
        value_counts = df[target_column].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in value_counts.items()}
        total_rows = len(df)
        target_props = {str(k): round(float(v) / total_rows, 6) for k, v in value_counts.items()}
        if 1 in value_counts:
            failure_rate = round(float(value_counts[1]) / total_rows, 6)

    return {
        "is_valid": len(errors) == 0,
        "row_count": int(len(df)),
        "col_count": int(len(df.columns)),
        "columns": actual_columns,
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "missing_values": missing_counts,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_count,
        "target_column": target_column,
        "target_present": target_present,
        "target_distribution": target_dist,
        "target_proportions": target_props,
        "failure_rate": failure_rate,
        "warnings": warnings,
        "errors": errors,
    }


def analyze_data_quality(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Perform data quality analysis, summary statistics, and sensor range checks.

    Args:
        df: Pandas DataFrame to analyze.
        target_column: Primary target column name.

    Returns:
        Structured dictionary containing statistical distributions and quality markers.
    """
    quality_warnings: list[str] = []

    # Numerical statistics
    numeric_df = df.select_dtypes(include=["number"])
    numerical_summary: dict[str, dict[str, float]] = {}
    for col in numeric_df.columns:
        desc = numeric_df[col].describe()
        numerical_summary[col] = {
            "count": float(desc["count"]),
            "mean": round(float(desc["mean"]), 4),
            "std": round(float(desc["std"]), 4),
            "min": round(float(desc["min"]), 4),
            "25%": round(float(desc["25%"]), 4),
            "50%": round(float(desc["50%"]), 4),
            "75%": round(float(desc["75%"]), 4),
            "max": round(float(desc["max"]), 4),
        }

    # Categorical summary
    categorical_df = df.select_dtypes(include=["object", "category"])
    categorical_summary: dict[str, dict[str, Any]] = {}
    for col in categorical_df.columns:
        nunique = int(categorical_df[col].nunique())
        val_counts = categorical_df[col].value_counts()
        if nunique > 20:
            top_dict = {str(k): int(v) for k, v in val_counts.head(5).items()}
            categorical_summary[col] = {
                "unique_count": nunique,
                "is_high_cardinality": True,
                "top_5_values": top_dict,
            }
        else:
            categorical_summary[col] = {
                "unique_count": nunique,
                "is_high_cardinality": False,
                "value_counts": {str(k): int(v) for k, v in val_counts.items()},
            }

    # Range & Sanity Checks
    if "Air temperature [K]" in df.columns:
        min_air = float(df["Air temperature [K]"].min())
        if min_air <= 0:
            quality_warnings.append(f"Air temperature <= 0 K observed: {min_air}")

    if "Process temperature [K]" in df.columns and "Air temperature [K]" in df.columns:
        temp_diff = df["Process temperature [K]"] - df["Air temperature [K]"]
        if (temp_diff < 0).any():
            count_negative = int((temp_diff < 0).sum())
            quality_warnings.append(
                f"Negative thermal delta (Process Temp < Air Temp) in {count_negative} records"
            )

    if "Rotational speed [rpm]" in df.columns:
        min_speed = float(df["Rotational speed [rpm]"].min())
        if min_speed <= 0:
            quality_warnings.append(f"Non-positive rotational speed observed: {min_speed}")

    if "Torque [Nm]" in df.columns:
        min_torque = float(df["Torque [Nm]"].min())
        if min_torque < 0:
            quality_warnings.append(f"Negative torque observed: {min_torque}")

    if "Tool wear [min]" in df.columns:
        min_wear = float(df["Tool wear [min]"].min())
        if min_wear < 0:
            quality_warnings.append(f"Negative tool wear observed: {min_wear}")

    # Failure Modes Breakdown
    failure_modes_summary: dict[str, dict[str, Any]] = {}
    present_modes = [col for col in FAILURE_MODE_COLUMNS if col in df.columns]
    for mode in present_modes:
        count = int(df[mode].sum())
        failure_modes_summary[mode] = {
            "positive_count": count,
            "rate": round(count / len(df), 6),
        }

    # Discrepancy checks (Machine failure vs failure modes)
    if target_column in df.columns and present_modes:
        any_mode = (df[present_modes].sum(axis=1) > 0).astype(int)
        failed_no_mode = int(((df[target_column] == 1) & (any_mode == 0)).sum())
        mode_no_failure = int(((df[target_column] == 0) & (any_mode == 1)).sum())

        if failed_no_mode > 0:
            quality_warnings.append(
                f"Observed {failed_no_mode} instances with Machine failure == 1 but all failure mode flags == 0"
            )
        if mode_no_failure > 0:
            quality_warnings.append(
                f"Observed {mode_no_failure} instances with failure mode flag active but Machine failure == 0 (primarily RNF)"
            )

    # Target class distribution
    target_dist: dict[str, int] = {}
    target_props: dict[str, float] = {}
    failure_rate = 0.0

    if target_column in df.columns:
        val_counts = df[target_column].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in val_counts.items()}
        target_props = {str(k): round(float(v) / len(df), 6) for k, v in val_counts.items()}
        failure_rate = target_props.get("1", 0.0)

    return {
        "shape": [int(len(df)), int(len(df.columns))],
        "columns": list(df.columns),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values_per_column": {
            str(col): int(val) for col, val in df.isnull().sum().items()
        },
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
        "target_column": target_column,
        "target_distribution": target_dist,
        "target_proportions": target_props,
        "failure_rate": failure_rate,
        "failure_modes_summary": failure_modes_summary,
        "quality_warnings": quality_warnings,
    }


def generate_validation_report(
    file_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete dataset ingestion, schema validation, and profiling pipeline.

    Args:
        file_path: Path to dataset file. Defaults to data/raw/ai4i2020.csv.
        output_path: Optional path to write output JSON report.

    Returns:
        Structured, JSON-serializable validation report dictionary.
    """
    resolved_path = get_raw_data_path(file_path)
    df = load_raw_data(resolved_path)
    schema_results = validate_schema(df)
    quality_results = analyze_data_quality(df)

    # Compute SHA-256 hash and file size
    sha256_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    file_size_bytes = resolved_path.stat().st_size

    # Compute relative path from project root if possible
    try:
        project_root = Path(__file__).resolve().parent.parent
        relative_path = str(resolved_path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        relative_path = str(resolved_path).replace("\\", "/")

    report: dict[str, Any] = {
        "dataset_source": DATASET_SOURCE_URL,
        "relative_file_path": relative_path,
        "file_name": resolved_path.name,
        "file_size_bytes": file_size_bytes,
        "sha256_checksum": sha256_hash,
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_valid": schema_results["is_valid"],
        "row_count": schema_results["row_count"],
        "column_count": schema_results["col_count"],
        "columns": schema_results["columns"],
        "column_dtypes": schema_results["dtypes"],
        "missing_values_per_column": schema_results["missing_values"],
        "total_missing_values": schema_results["total_missing"],
        "duplicate_row_count": schema_results["duplicate_rows"],
        "target_column": schema_results["target_column"],
        "target_distribution": schema_results["target_distribution"],
        "target_proportions": schema_results["target_proportions"],
        "failure_rate": schema_results["failure_rate"],
        "failure_modes_summary": quality_results["failure_modes_summary"],
        "numerical_summary": quality_results["numerical_summary"],
        "categorical_summary": quality_results["categorical_summary"],
        "validation_warnings": quality_results["quality_warnings"],
        "validation_errors": schema_results["errors"],
    }

    if output_path is not None:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

    return report
