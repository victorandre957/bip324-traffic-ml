"""Dataset discovery, legacy-label normalization and leakage-resistant splits."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import FEATURE_SCHEMA, feature_frame

ACTIVITY_CLASSES = (
    "other",
    "bitcoin_v2_other",
    "handshake",
    "block",
    "compact_block",
    "inventory",
    "request",
    "transaction",
)


def dataset_paths(inputs: list[Path]) -> list[Path]:
    """Accept individual CSVs, shell patterns, or a root containing all prepared runs."""
    paths = []
    for value in inputs:
        value = Path(value)
        if value.is_dir():
            matches = sorted(
                p
                for p in value.rglob("event_samples.csv")
                if not any(part.startswith(".") for part in p.relative_to(value).parts)
            )
        elif value.is_file():
            matches = [value]
        else:
            matches = [Path(p) for p in sorted(glob.glob(str(value)))]
        if not matches:
            raise ValueError(f"no datasets found: {value}")
        paths.extend(p.resolve() for p in matches)
    if not paths:
        raise ValueError("at least one dataset is required")
    if len(set(paths)) != len(paths):
        raise ValueError("the same dataset was supplied more than once")
    return sorted(paths)


def load_datasets(paths: list[Path]) -> pd.DataFrame:
    frames, runs_seen, sources_seen = [], set(), set()
    for path in dataset_paths(paths):
        frame = pd.read_csv(path)
        required = {"run_id", "flow_id", "bucket_index", "label_activity"}
        if not required.issubset(frame) or frame.empty:
            raise ValueError(f"missing required columns or empty dataset: {path}")
        if frame[list(required - {"label_activity"})].isna().any().any():
            raise ValueError(f"missing observation identifiers: {path}")
        runs = set(frame["run_id"].astype(str))
        if runs_seen & runs:
            raise ValueError(f"duplicate run IDs across datasets: {sorted(runs_seen & runs)}")
        runs_seen.update(runs)
        manifest = path.parent / "preparation_summary.json"
        if manifest.is_file():
            info = json.loads(manifest.read_text())
            if info.get("feature_schema", FEATURE_SCHEMA) != FEATURE_SCHEMA:
                raise ValueError(f"unsupported feature schema in {manifest}")
            if info.get("scale_seconds", 0.25) != 0.25:
                raise ValueError("training contract requires 250 ms windows; do not mix scales")
            source = info.get("lab_dir")
            if source and source in sources_seen:
                raise ValueError(f"same capture appears twice under different run IDs: {source}")
            if source:
                sources_seen.add(source)
        frame["label_activity"] = frame["label_activity"].replace(
            {
                "bitcoin_v1": "other",
                "transaction_sequence": "transaction",
            }
        )
        invalid = set(frame["label_activity"].dropna()) - set(ACTIVITY_CLASSES) - {"ambiguous"}
        if invalid or frame["label_activity"].isna().any():
            raise ValueError(f"invalid/missing activity labels in {path}: {sorted(invalid)}")
        feature_frame(frame)
        duplicate_key = ["run_id", "flow_id", "bucket_index"]
        if "phase" in frame:
            duplicate_key.append("phase")
        if frame.duplicated(duplicate_key).any():
            raise ValueError(f"duplicate observation rows in {path}")
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["run_id", "flow_id", "bucket_index"], kind="stable")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    training_runs: list[str]
    test_runs: list[str]
    missing_training_classes: list[str]
    missing_test_classes: list[str]
    ambiguous: int


def split_runs(frame: pd.DataFrame, test_runs: list[str], allow_incomplete: bool = False) -> Split:
    """Hold out complete captures before preprocessing or estimating class weights."""
    runs = sorted(frame["run_id"].astype(str).unique())
    if len(runs) < 3:
        raise ValueError("at least three independent runs are required")
    if not test_runs:
        raise ValueError("explicit test runs are required; keep the same untouched holdout")
    unknown = set(test_runs) - set(runs)
    if unknown:
        raise ValueError(f"unknown test runs: {sorted(unknown)}")
    valid = frame["label_activity"].isin(ACTIVITY_CLASSES).to_numpy()
    test_mask = frame["run_id"].astype(str).isin(test_runs).to_numpy()
    train, test = np.flatnonzero(valid & ~test_mask), np.flatnonzero(valid & test_mask)
    if not len(train) or not len(test):
        raise ValueError("both training and test partitions need samples")
    train_classes = set(frame.iloc[train]["label_activity"])
    test_classes = set(frame.iloc[test]["label_activity"])
    if len(train_classes) < 2:
        raise ValueError("training partition needs at least two classes")
    missing_train = sorted(set(ACTIVITY_CLASSES) - train_classes)
    missing_test = sorted(set(ACTIVITY_CLASSES) - test_classes)
    if (missing_train or missing_test) and not allow_incomplete:
        raise ValueError(
            f"missing activity classes: training={missing_train}, test={missing_test}; "
            "collect missing events or explicitly use --allow-incomplete-classes"
        )
    return Split(
        train,
        test,
        sorted(set(runs) - set(test_runs)),
        sorted(set(test_runs)),
        missing_train,
        missing_test,
        int((~valid).sum()),
    )
