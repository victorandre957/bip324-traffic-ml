"""CLI wiring only. Each pipeline stage can also be called independently in Python."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _preparation_options(parser, batch=False):
    parser.add_argument("--lab-root" if batch else "--lab-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    if batch:
        parser.add_argument(
            "--features-root",
            "--analysis-root",
            dest="features_root",
            type=Path,
            default=Path("artifacts/features"),
        )
        parser.add_argument("--seed-prefix")
    else:
        parser.add_argument("--features-dir", "--analysis-dir", dest="features_dir", type=Path)
        parser.add_argument("--run-id")
    parser.add_argument(
        "--reuse-features",
        "--reuse-analysis",
        dest="reuse_features",
        action="store_true",
        help="Label existing extracted CSVs, including legacy analysis exports.",
    )
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--pairing-tolerance", type=float, default=1.0)


def _training_options(parser):
    parser.add_argument(
        "datasets", nargs="+", type=Path, help="CSV files, glob patterns or a dataset root."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--test-runs", required=True, help="Comma-separated complete held-out run IDs."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-incomplete-classes", action="store_true")
    parser.add_argument("--model", choices=("xgboost", "temporal-cnn"), default="xgboost")
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_parser():
    parser = argparse.ArgumentParser(
        description="PCAP -> passive observations -> labels -> training/evaluation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        "extract", help="Extract full-PCAP observations without logs or labels."
    )
    extract.add_argument("--pcap", required=True, type=Path)
    extract.add_argument("--run-id", required=True)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--scale", type=float, default=0.25)
    label = commands.add_parser(
        "label",
        aliases=["prepare"],
        help="Label extracted observations using accepted lab metadata.",
    )
    _preparation_options(label)
    lab = commands.add_parser(
        "prepare-lab", help="Extract and label one accepted capture; preserve completed datasets."
    )
    _preparation_options(lab)
    accepted = commands.add_parser(
        "prepare-accepted", help="Incrementally prepare only new accepted captures."
    )
    _preparation_options(accepted, batch=True)
    train = commands.add_parser(
        "train",
        aliases=["train-activity"],
        help="Train one multiclass activity model and evaluate on held-out runs.",
    )
    _training_options(train)
    explain = commands.add_parser(
        "explain",
        help="Explain held-out predictions with XGBoost TreeSHAP or grouped input permutation.",
    )
    explain.add_argument("datasets", nargs="+", type=Path)
    explain.add_argument("--model-dir", required=True, type=Path)
    explain.add_argument("--output-dir", required=True, type=Path)
    explain.add_argument(
        "--runs", help="Comma-separated held-out run IDs; defaults to the model's test runs."
    )
    explain.add_argument("--max-samples", type=int, default=1024)
    explain.add_argument(
        "--error-samples",
        type=int,
        default=5,
        help="Additional diagnostic examples per error type/activity/noise family.",
    )
    explain.add_argument("--batch-size", type=int, default=128)
    explain.add_argument("--threads", type=int, default=4)
    explain.add_argument("--seed", type=int, default=42)
    explain.add_argument("--method", choices=("shap", "permutation"), default="shap")
    explain.add_argument("--repeats", type=int, default=3)
    explain.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    explain.add_argument(
        "--no-plots",
        action="store_true",
        help="Export numerical reports only, without optional SHAP/Matplotlib dependencies.",
    )
    predict = commands.add_parser("predict", help="Predict from unlabeled passive observations.")
    predict.add_argument("--observations", required=True, type=Path)
    predict.add_argument("--model-dir", required=True, type=Path)
    predict.add_argument("--output-dir", required=True, type=Path)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    predict.add_argument("--batch-size", type=int, default=1024)
    return parser


def dispatch(args):
    if args.command == "explain":
        from .explanation.xgboost import ExplainConfig, explain_xgboost

        runs = (
            [value.strip() for value in args.runs.split(",") if value.strip()]
            if args.runs
            else None
        )
        if args.method == "permutation":
            from .explanation.permutation import PermutationConfig, explain_permutation

            config = PermutationConfig(
                max_samples=args.max_samples,
                repeats=args.repeats,
                batch_size=args.batch_size,
                threads=args.threads,
                seed=args.seed,
                device=args.device,
                plots=not args.no_plots,
            )
            return explain_permutation(args.datasets, args.model_dir, args.output_dir, runs, config)
        config = ExplainConfig(
            max_samples=args.max_samples,
            error_samples=args.error_samples,
            batch_size=args.batch_size,
            threads=args.threads,
            seed=args.seed,
            plots=not args.no_plots,
        )
        return explain_xgboost(args.datasets, args.model_dir, args.output_dir, runs, config)
    if args.command == "extract":
        from .capture.extract import extract_pcap
        from .preparation import validate_run_id

        return extract_pcap(args.pcap, args.output_dir, validate_run_id(args.run_id), args.scale)
    if args.command == "prepare-accepted":
        from .preparation import prepare_accepted

        return prepare_accepted(
            args.lab_root,
            args.output_dir,
            args.features_root,
            args.seed_prefix,
            args.scale,
            args.pairing_tolerance,
            args.reuse_features,
        )
    if args.command in {"label", "prepare", "prepare-lab"}:
        from .preparation import accepted_lab_run, prepare_from_features, prepare_lab

        run_id = args.run_id or str(accepted_lab_run(args.lab_dir).get("seed") or args.lab_dir.name)
        features = args.features_dir or args.output_dir.with_name(
            args.output_dir.name + "-features"
        )
        if args.command in {"label", "prepare"}:
            if args.features_dir is None:
                raise ValueError("label requires --features-dir (or legacy --analysis-dir)")
            return prepare_from_features(
                run_id, features, args.lab_dir, args.output_dir, args.scale, args.pairing_tolerance
            )
        return prepare_lab(
            args.lab_dir,
            args.output_dir,
            features,
            run_id,
            args.scale,
            args.pairing_tolerance,
            args.reuse_features,
        )
    if args.command == "predict":
        from .inference import predict

        return predict(
            args.observations, args.model_dir, args.output_dir, args.device, args.batch_size
        )
    test_runs = [value.strip() for value in args.test_runs.split(",") if value.strip()]
    if args.model == "temporal-cnn":
        from .training.temporal import CNNConfig, train_cnn

        config = CNNConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate if args.learning_rate is not None else 1e-3,
            threads=args.threads,
            device=args.device,
        )
        return train_cnn(
            args.datasets, args.output_dir, test_runs, config, args.allow_incomplete_classes
        )
    from .training.xgboost import XGBoostConfig, train_xgboost

    config = XGBoostConfig(
        seed=args.seed,
        trees=args.trees,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate if args.learning_rate is not None else 0.05,
        threads=args.threads,
    )
    return train_xgboost(
        args.datasets, args.output_dir, test_runs, config, args.allow_incomplete_classes
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("[interrupted] completed datasets preserved", file=sys.stderr)
        raise SystemExit(130) from None
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
