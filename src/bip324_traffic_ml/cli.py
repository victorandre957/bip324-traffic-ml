"""Command-line interface for dataset preparation and model training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .consolidate import EVENT_LABEL_COLUMNS, prepare_run
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare and evaluate passive BIP324 models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Pair one analyzed capture with its lab labels.",
    )
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--analysis-dir", required=True, type=Path)
    prepare.add_argument("--lab-dir", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--scale", type=float, default=0.25)
    prepare.add_argument("--pairing-tolerance", type=float, default=1.0)

    prepare_many = subparsers.add_parser(
        "prepare-many",
        help="Prepare every run listed in a CSV manifest.",
    )
    prepare_many.add_argument("--manifest", required=True, type=Path)
    prepare_many.add_argument("--output-dir", required=True, type=Path)
    prepare_many.add_argument("--scale", type=float, default=0.25)
    prepare_many.add_argument("--pairing-tolerance", type=float, default=1.0)

    train = subparsers.add_parser(
        "train",
        help="Train XGBoost and compare it with a heuristic baseline.",
    )
    train.add_argument("datasets", nargs="+", type=Path)
    train.add_argument("--target", required=True)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--test-runs", default="")
    train.add_argument("--seed", type=int, default=42)

    train_all = subparsers.add_parser(
        "train-all",
        help="Train every event target with the same held-out runs.",
    )
    train_all.add_argument("datasets", nargs="+", type=Path)
    train_all.add_argument("--output-dir", required=True, type=Path)
    train_all.add_argument("--test-runs", default="")
    train_all.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Dispatch one CLI command and print its machine-readable summary."""
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_run(
            args.run_id,
            args.analysis_dir,
            args.lab_dir,
            args.output_dir,
            args.scale,
            args.pairing_tolerance,
        )
    elif args.command == "prepare-many":
        result = _prepare_many(args)
    elif args.command == "train":
        test_runs = [run.strip() for run in args.test_runs.split(",") if run.strip()]
        result = train_model(
            args.datasets,
            args.target,
            args.output_dir,
            test_runs or None,
            args.seed,
        )
    else:
        result = _train_all(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def _prepare_many(args: argparse.Namespace) -> dict[str, object]:
    with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    required_columns = {"run_id", "analysis_dir", "lab_dir"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(
            "manifest must contain run_id, analysis_dir, and lab_dir columns"
        )
    summaries = []
    for row in rows:
        run_id = row["run_id"].strip()
        summaries.append(
            prepare_run(
                run_id,
                Path(row["analysis_dir"]),
                Path(row["lab_dir"]),
                args.output_dir / run_id,
                args.scale,
                args.pairing_tolerance,
            )
        )
    return {"prepared_runs": len(summaries), "runs": summaries}


def _train_all(args: argparse.Namespace) -> dict[str, object]:
    test_runs = [run.strip() for run in args.test_runs.split(",") if run.strip()]
    results = {}
    for target in EVENT_LABEL_COLUMNS:
        results[target] = train_model(
            args.datasets,
            target,
            args.output_dir / target.removeprefix("label_"),
            test_runs or None,
            args.seed,
        )
    return {"targets": results}
