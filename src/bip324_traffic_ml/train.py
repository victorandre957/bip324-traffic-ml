"""Train passive classifiers with complete lab runs held out for testing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NON_FEATURE_COLUMNS = {
    "run_id",
    "dataset",
    "flow_id",
    "key",
    "notes",
    "cleartext_hint",
    "label_source",
    "protocol",
    "endpoint_a_ip",
    "endpoint_a_port",
    "endpoint_b_ip",
    "endpoint_b_port",
    "start_time",
    "end_time",
    "timestamp",
    "bucket_index",
    "bitcoin_v2_candidate",
}

BASELINE_FOR_TARGET = {
    "label_bip324": "baseline_handshake",
}

RETROSPECTIVE_FEATURES = {
    "duration_seconds",
    "packet_count",
    "payload_packet_count",
}


def _load_datasets(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("at least one dataset is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric passive features after removing identifiers and labels."""
    excluded = {
        column
        for column in frame.columns
        if column in NON_FEATURE_COLUMNS
        or column.startswith("bip324_")
        or column.startswith("label_")
        or column.startswith("baseline_")
    }
    features = frame.drop(columns=sorted(excluded), errors="ignore")
    features = pd.get_dummies(features, columns=["phase"] if "phase" in features else [])
    boolean_columns = features.select_dtypes(include="bool").columns
    features[boolean_columns] = features[boolean_columns].astype(int)
    numeric = features.select_dtypes(include="number").fillna(0)
    if numeric.empty:
        raise ValueError("dataset has no passive numeric features")
    return numeric


def _binary_metrics(y_true, y_pred, probabilities=None) -> dict[str, float | int | None]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    metrics: dict[str, float | int | None] = {
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": None,
        "roc_auc": None,
    }
    if probabilities is not None and len(set(y_true)) == 2:
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def train_model(
    dataset_paths: list[Path],
    target: str,
    output_dir: Path,
    test_runs: list[str] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Train one classifier and evaluate it on complete held-out runs."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exception:
        raise RuntimeError(
            "xgboost is required; install the project with: pip install -e ."
        ) from exception

    frame = _load_datasets(dataset_paths)
    if target not in frame:
        raise ValueError(f"target column not found: {target}")
    runs = sorted(frame["run_id"].astype(str).unique())
    if len(runs) < 3:
        raise ValueError("at least three independent run_id values are required")
    selected_test_runs = test_runs or [runs[-1]]
    unknown_runs = sorted(set(selected_test_runs) - set(runs))
    if unknown_runs:
        raise ValueError(f"unknown test runs: {', '.join(unknown_runs)}")

    test_mask = frame["run_id"].astype(str).isin(selected_test_runs)
    train_rows = frame.loc[~test_mask].copy()
    test_rows = frame.loc[test_mask].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError("both training and test partitions must contain rows")
    if train_rows[target].nunique() < 2:
        raise ValueError(f"training partition needs positive and negative samples for {target}")

    all_features = _feature_frame(frame)
    train_features = all_features.loc[train_rows.index]
    test_features = all_features.loc[test_rows.index]
    y_train = train_rows[target].astype(int)
    y_test = test_rows[target].astype(int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_features, y_train)
    probabilities = model.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "model.json")
    prediction_output = test_rows[["run_id", "flow_id", target]].copy()
    if "timestamp" in test_rows:
        prediction_output["timestamp"] = test_rows["timestamp"]
    prediction_output["model_probability"] = probabilities
    prediction_output["model_prediction"] = predictions
    baseline_column = BASELINE_FOR_TARGET.get(
        target,
        target.replace("label_", "baseline_", 1),
    )
    if baseline_column in test_rows:
        prediction_output[baseline_column] = test_rows[baseline_column].astype(int)
    prediction_output.to_csv(output_dir / "test_predictions.csv", index=False)

    importance = pd.DataFrame(
        {"feature": train_features.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    metrics: dict[str, object] = {
        "target": target,
        "threshold": 0.5,
        "training_runs": sorted(set(runs) - set(selected_test_runs)),
        "test_runs": selected_test_runs,
        "training_samples": len(train_rows),
        "test_samples": len(test_rows),
        "feature_count": len(train_features.columns),
        "analysis_mode": "retrospective",
        "retrospective_features": sorted(
            column
            for column in train_features.columns
            if column in RETROSPECTIVE_FEATURES
            or column.startswith("bytes_total_lead_")
        ),
        "scale_pos_weight": scale_pos_weight,
        "model": _binary_metrics(y_test, predictions, probabilities),
        "baseline": None,
        "split_policy": (
            "Entire lab runs are held out; flows and windows from a test run "
            "never enter training."
        ),
    }
    if baseline_column in test_rows:
        metrics["baseline"] = _binary_metrics(y_test, test_rows[baseline_column].astype(int))
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(train_features.columns), indent=2),
        encoding="utf-8",
    )
    return metrics
