"""Shared provenance and split reporting."""

from dataclasses import asdict
from pathlib import Path

from ..artifacts import environment, file_identity, write_json
from ..features import FEATURE_COLUMNS, FEATURE_SCHEMA
from .data import dataset_paths


def training_context(frame, split, model, config):
    train = frame.iloc[split.train]
    return {
        "task": "single_model_full_pcap_activity_classification",
        "model": model,
        "training_runs": split.training_runs,
        "test_runs": split.test_runs,
        "training_samples": len(train),
        "training_flows": len(train[["run_id", "flow_id"]].drop_duplicates()),
        "training_class_counts": train["label_activity"].value_counts().to_dict(),
        "ambiguous_windows_excluded": split.ambiguous,
        "missing_training_classes": split.missing_training_classes,
        "missing_test_classes": split.missing_test_classes,
        "coverage_complete": not (split.missing_training_classes or split.missing_test_classes),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_schema": FEATURE_SCHEMA,
        "analysis_mode": "retrospective",
        "configuration": asdict(config),
        "feature_policy": "34 allowlisted passive PCAP measurements; no labels, ports or IPs.",
        "split_policy": "Complete lab runs held out before fitting; test data never selects epochs or hyperparameters.",
    }


def save_manifest(output: Path, inputs, context, classes):
    write_json(output / "feature_columns.json", list(FEATURE_COLUMNS))
    write_json(output / "class_labels.json", classes)
    write_json(
        output / "training_manifest.json",
        {
            **context,
            "environment": environment(),
            "datasets": [file_identity(path, checksum=True) for path in dataset_paths(inputs)],
            "window_seconds": 0.25,
        },
    )
