"""Main CLI Entry-Point for AI-Predictive-Maintenance-IoT.

This module provides the central command-line interface for orchestrating
the predictive maintenance workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Predictive-Maintenance-IoT Pipeline Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "preprocess", "train", "evaluate", "predict", "alert", "simulate"],
        default="validate",
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input telemetry data for validation, training, evaluation, prediction, or alerting.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path or directory to save validation, evaluation, or alert report artifacts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        help="Model identifier key (default: 'random_forest') or path to serialized .joblib pipeline.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold strictly between 0.0 and 1.0 (default: 0.5).",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=0.3,
        help="Alert system warning score threshold strictly between 0.0 and 1.0 (default: 0.3).",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=0.7,
        help="Alert system critical score threshold strictly between 0.0 and 1.0 (default: 0.7).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Telemetry simulation stream interval in seconds.",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=10,
        help="Number of simulated telemetry records to emit (default: 10).",
    )
    parser.add_argument(
        "--machines",
        type=int,
        default=1,
        help="Number of simulated machines (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic telemetry generation (default: 42).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["nominal", "degradation", "mixed", "fault_injection"],
        default="nominal",
        help="Operational simulation scenario: nominal, degradation, mixed, or fault_injection (default: 'nominal').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    print("=" * 78)
    print(" AI-Predictive-Maintenance-IoT | Pipeline Runner")
    print("=" * 78)
    print(f"Selected Mode: {args.mode}")

    if args.mode == "validate":
        from src.data_loader import generate_validation_report

        print("\nExecuting Phase 2 Dataset Validation & Profiling Pipeline...")
        output_report_path = args.output or "outputs/metrics/data_validation_report.json"
        try:
            report = generate_validation_report(
                file_path=args.input,
                output_path=output_report_path,
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Validation failed: {err}", file=sys.stderr)
            return 1

        print(f"  [OK] Dataset validated: {report['relative_file_path']}")
        print(f"  [OK] Shape: {report['row_count']} rows x {report['column_count']} columns")
        print(f"  [OK] Total Missing Values: {report['total_missing_values']}")
        print(f"  [OK] Duplicate Rows: {report['duplicate_row_count']}")
        print(f"  [OK] Target Distribution: {report['target_distribution']}")
        print(f"  [OK] Machine Failure Rate: {report['failure_rate']:.4%}")
        if report["validation_warnings"]:
            print(f"  [!] Validation Warnings ({len(report['validation_warnings'])}):")
            for w in report["validation_warnings"]:
                print(f"      - {w}")
        print(f"\nReport written to: {output_report_path}")
        print("=" * 78)
        return 0

    if args.mode == "preprocess":
        from src.data_loader import load_raw_data
        from src.preprocessing import preprocess_data

        print("\nExecuting Phase 3 Data Cleaning & Preprocessing Pipeline...")
        try:
            df = load_raw_data(args.input)
            res = preprocess_data(df)
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Preprocessing failed: {err}", file=sys.stderr)
            return 1

        print(f"  [OK] Original dataset dimensions: {len(df)} rows x {len(df.columns)} columns")
        print(f"  [OK] Training predictors shape: {res['train_shape'][0]} rows x {res['train_shape'][1]} columns")
        print(f"  [OK] Test predictors shape: {res['test_shape'][0]} rows x {res['test_shape'][1]} columns")
        print(f"  [OK] Train class distribution: {res['train_class_counts']} (Failure rate: {res['train_class_counts'][1] / res['train_shape'][0]:.4%})")
        print(f"  [OK] Test class distribution: {res['test_class_counts']} (Failure rate: {res['test_class_counts'][1] / res['test_shape'][0]:.4%})")
        print(f"  [OK] Transformed train feature dimensions: {res['transformed_train_shape'][0]} rows x {res['transformed_train_shape'][1]} columns")
        print(f"  [OK] Transformed test feature dimensions: {res['transformed_test_shape'][0]} rows x {res['transformed_test_shape'][1]} columns")
        print(f"  [OK] Output feature schema: {res['feature_names_out']}")
        print("  [OK] Leakage Safeguard: Preprocessor transformer was fitted STRICTLY on training data only.")
        print("=" * 78)
        return 0

    if args.mode == "train":
        from src.train_model import train_all_baseline_models

        print("\nExecuting Phase 4 Baseline Model Training Pipeline...")
        try:
            results = train_all_baseline_models(data_path=args.input)
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Model training failed: {err}", file=sys.stderr)
            return 1

        train_counts = results["training_metadata"]["train_class_distribution"]
        test_counts = results["training_metadata"]["test_class_distribution"]
        train_n = results["training_metadata"]["train_sample_count"]
        test_n = results["training_metadata"]["test_sample_count"]

        print(f"  [OK] Training dataset: {results['training_metadata']['dataset_path']}")
        print(f"  [OK] Partition: Train={train_n} samples ({train_counts.get(1, 0)} failures, {train_counts.get(1, 0) / train_n:.2%}) | Test={test_n} samples ({test_counts.get(1, 0)} failures, {test_counts.get(1, 0) / test_n:.2%})")
        print(f"  [OK] Permitted Predictors: {results['training_metadata']['permitted_predictors']}")
        print(f"  [OK] Strictly Excluded Columns: {results['training_metadata']['excluded_columns']}")
        print(f"  [OK] Trained Model Configurations ({len(results['models'])}):")
        for key, path in results["model_paths"].items():
            model_info = results["training_metadata"]["models"][key]
            print(f"      * {model_info['model_name']}: saved to {path.name}")
        print("\nAll baseline pipelines trained and serialized to models/.")
        print("=" * 78)
        return 0

    if args.mode == "evaluate":
        from src.evaluate_model import run_baseline_evaluation

        print("\nExecuting Phase 4 Baseline Model Evaluation Pipeline...")
        try:
            eval_results = run_baseline_evaluation(
                data_path=args.input,
                output_dir=args.output,
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Model evaluation failed: {err}", file=sys.stderr)
            return 1

        summary_df = eval_results["summary_table"]
        print("\nBaseline Evaluation Summary (Test Set, Threshold = 0.5):")
        print("-" * 78)
        cols_to_print = [
            "model_key",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "tp",
            "fp",
            "fn",
        ]
        display_df = summary_df.rename(columns={
            "true_positives": "tp",
            "false_positives": "fp",
            "false_negatives": "fn",
        })[cols_to_print]
        print(display_df.to_string(index=False))
        print("-" * 78)

        print("\nGenerated Evaluation Reports:")
        for name, path in eval_results["report_paths"].items():
            print(f"  [OK] {name}: {path}")
        print("=" * 78)
        return 0

    if args.mode == "predict":
        from src.predict import predict_batch, validate_threshold

        print("\nExecuting Phase 5 Inference Engine...")

        if not args.input:
            print(
                "\n[ERROR] Missing input file for prediction. "
                "Please specify a CSV file containing raw telemetry predictors:\n"
                "  Example: python main.py --mode predict --input data/sample_inference_records.csv",
                file=sys.stderr,
            )
            return 1

        input_path = Path(args.input).resolve()
        if not input_path.is_file():
            print(f"\n[ERROR] Prediction input file not found at: {input_path}", file=sys.stderr)
            return 1

        try:
            val_threshold = validate_threshold(args.threshold)
        except ValueError as err:
            print(f"\n[ERROR] Invalid threshold: {err}", file=sys.stderr)
            return 1

        try:
            df_input = pd.read_csv(input_path)
        except Exception as exc:
            print(f"\n[ERROR] Failed to parse input CSV at {input_path}: {exc}", file=sys.stderr)
            return 1

        if df_input.empty or len(df_input) == 0:
            print(f"\n[ERROR] Input CSV file at {input_path} contains zero records.", file=sys.stderr)
            return 1

        print(f"  [OK] Loaded input records: {len(df_input)} rows from {input_path.name}")
        print(f"  [OK] Model requested: {args.model}")
        print(f"  [OK] Classification threshold: {val_threshold}")

        try:
            batch_results = predict_batch(
                records=df_input,
                model=args.model,
                threshold=val_threshold,
                error_policy="per_record",
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Inference failed: {err}", file=sys.stderr)
            return 1

        print("\nPrediction Results:")
        print("-" * 78)

        success_count = sum(1 for r in batch_results if r.get("status") == "success")
        failure_pred_count = sum(1 for r in batch_results if r.get("predicted_class") == 1)
        nominal_pred_count = sum(1 for r in batch_results if r.get("predicted_class") == 0)
        error_count = len(batch_results) - success_count

        for res in batch_results:
            row_i = res.get("row_index", 0)
            if res.get("status") == "success":
                score_str = f"{res['positive_class_score']:.4f}"
                warn_str = f" [!] Warnings: {res['warnings']}" if res.get("warnings") else ""
                print(
                    f"  Row {row_i:3d} | Class {res['predicted_class']} ({res['predicted_label']}) | "
                    f"Score: {score_str} | Threshold: {res['threshold']}{warn_str}"
                )
            else:
                print(f"  Row {row_i:3d} | [ERROR] {res.get('error')}")

        print("-" * 78)
        print(
            f"Summary: {len(batch_results)} records evaluated | "
            f"Successful: {success_count} ({nominal_pred_count} nominal, {failure_pred_count} failure predicted) | "
            f"Errors: {error_count}"
        )
        print("=" * 78)
        return 0 if error_count == 0 else 1

    if args.mode == "alert":
        from src.alert_system import AlertConfig, evaluate_alerts_from_telemetry

        print("\nExecuting Phase 6 Maintenance Alert System...")

        if not args.input:
            print(
                "\n[ERROR] Missing input file for alerting. "
                "Please specify a CSV file containing raw telemetry predictors:\n"
                "  Example: python main.py --mode alert --input data/sample_inference_records.csv",
                file=sys.stderr,
            )
            return 1

        input_path = Path(args.input).resolve()
        if not input_path.is_file():
            print(f"\n[ERROR] Alert input file not found at: {input_path}", file=sys.stderr)
            return 1

        try:
            alert_config = AlertConfig(
                warning_threshold=args.warning_threshold,
                critical_threshold=args.critical_threshold,
            )
        except ValueError as err:
            print(f"\n[ERROR] Invalid alert configuration: {err}", file=sys.stderr)
            return 1

        try:
            df_input = pd.read_csv(input_path)
        except Exception as exc:
            print(f"\n[ERROR] Failed to parse input CSV at {input_path}: {exc}", file=sys.stderr)
            return 1

        if df_input.empty or len(df_input) == 0:
            print(f"\n[ERROR] Input CSV file at {input_path} contains zero records.", file=sys.stderr)
            return 1

        print(f"  [OK] Loaded input records: {len(df_input)} rows from {input_path.name}")
        print(f"  [OK] Model requested: {args.model}")
        print(f"  [OK] Warning Threshold: {alert_config.warning_threshold}")
        print(f"  [OK] Critical Threshold: {alert_config.critical_threshold}")

        try:
            alert_results = evaluate_alerts_from_telemetry(
                records=df_input,
                model=args.model,
                config=alert_config,
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Alert evaluation failed: {err}", file=sys.stderr)
            return 1

        print("\nMaintenance Alerts:")
        print("-" * 78)

        normal_count = sum(1 for a in alert_results if a.get("alert_level") == "NORMAL")
        warning_count = sum(1 for a in alert_results if a.get("alert_level") == "WARNING")
        critical_count = sum(1 for a in alert_results if a.get("alert_level") == "CRITICAL")
        invalid_count = sum(1 for a in alert_results if a.get("alert_level") == "INVALID")

        for alert in alert_results:
            row_i = alert.get("record_index", 0)
            level = alert.get("alert_level", "UNKNOWN")
            score = alert.get("positive_class_score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            reason = alert.get("reason", "")
            action = alert.get("recommended_action", "")

            indicator = {
                "NORMAL": "[NORMAL]  ",
                "WARNING": "[WARNING] ",
                "CRITICAL": "[CRITICAL]",
                "INVALID": "[INVALID] ",
            }.get(level, "[UNKNOWN] ")

            print(f"  Row {row_i:3d} | {indicator} | Score: {score_str}")
            print(f"          Reason: {reason}")
            print(f"          Action: {action}")

        print("-" * 78)
        print(
            f"Alert Summary: {len(alert_results)} evaluated | "
            f"NORMAL: {normal_count} | WARNING: {warning_count} | "
            f"CRITICAL: {critical_count} | INVALID: {invalid_count}"
        )

        if args.output:
            out_p = Path(args.output).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as fp:
                json.dump(alert_results, fp, indent=2)
            print(f"\nAlert results written to: {out_p}")

        print("=" * 78)
        return 0 if invalid_count == 0 else 1

    if args.mode == "simulate":
        from src.alert_system import AlertConfig
        from src.simulator import run_simulation

        print("\nExecuting Phase 7 IoT Telemetry Simulation & Alert Pipeline...")

        try:
            alert_config = AlertConfig(
                warning_threshold=args.warning_threshold,
                critical_threshold=args.critical_threshold,
            )
        except ValueError as err:
            print(f"\n[ERROR] Invalid alert configuration: {err}", file=sys.stderr)
            return 1

        print(f"  [OK] Model requested: {args.model}")
        print(f"  [OK] Records requested: {args.records}")
        print(f"  [OK] Simulated machines: {args.machines}")
        print(f"  [OK] Scenario: {args.scenario}")
        print(f"  [OK] Interval: {args.interval}s")
        print(f"  [OK] Seed: {args.seed}")
        print(f"  [OK] Warning Threshold: {alert_config.warning_threshold}")
        print(f"  [OK] Critical Threshold: {alert_config.critical_threshold}")

        try:
            events = run_simulation(
                num_records=args.records,
                num_machines=args.machines,
                interval_seconds=args.interval,
                scenario=args.scenario,
                seed=args.seed,
                model=args.model,
                alert_config=alert_config,
                output_path=args.output,
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[ERROR] Simulation failed: {err}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"\n[ERROR] Unexpected error during simulation: {exc}", file=sys.stderr)
            return 1

        print("\nSimulated Telemetry Events:")
        print("-" * 78)

        normal_count = sum(1 for e in events if e.get("alert_level") == "NORMAL")
        warning_count = sum(1 for e in events if e.get("alert_level") == "WARNING")
        critical_count = sum(1 for e in events if e.get("alert_level") == "CRITICAL")
        invalid_count = sum(1 for e in events if e.get("alert_level") == "INVALID")

        for evt in events:
            evt_id = evt.get("event_id", "")
            ts = evt.get("timestamp", "")
            m_id = evt.get("machine_id", "")
            m_type = evt.get("Type", "")
            level = evt.get("alert_level", "UNKNOWN")
            score = evt.get("positive_class_score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            speed = evt.get("Rotational speed [rpm]", 0.0)
            torque = evt.get("Torque [Nm]", 0.0)
            wear = evt.get("Tool wear [min]", 0.0)

            indicator = {
                "NORMAL": "[NORMAL]  ",
                "WARNING": "[WARNING] ",
                "CRITICAL": "[CRITICAL]",
                "INVALID": "[INVALID] ",
            }.get(level, "[UNKNOWN] ")

            print(
                f"  {evt_id} | {ts} | {m_id} ({m_type}) | {indicator} | Score: {score_str} | "
                f"Speed: {speed:6.1f} rpm | Torque: {torque:5.2f} Nm | Wear: {wear:5.1f} min"
            )

        print("-" * 78)
        print(
            f"Simulation Summary: {len(events)} events emitted | "
            f"NORMAL: {normal_count} | WARNING: {warning_count} | "
            f"CRITICAL: {critical_count} | INVALID: {invalid_count}"
        )

        if args.output:
            print(f"\nSimulation events written to: {args.output}")

        print("=" * 78)
        return 0

    print(
        f"\n[NOTICE] Unrecognized mode '{args.mode}'."
    )
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
