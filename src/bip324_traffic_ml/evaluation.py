"""Common metrics for all architectures; ground truth is never an inference input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .artifacts import write_json


def detection_metrics(truth, predicted) -> dict:
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    tp = int((truth & predicted).sum())
    tn = int((~truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "positive_samples": tp + fn,
        "negative_samples": tn + fp,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "accuracy": (tp + tn) / len(truth) if len(truth) else None,
    }


def aggregate_flows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Majority vote after mapping activities to v2/other; ties retain legacy v2 vote."""
    rows = []
    for (run_id, flow_id), group in predictions.groupby(["run_id", "flow_id"], sort=True):
        positive = group["model_prediction"].ne("other")
        row = {
            "run_id": run_id,
            "flow_id": flow_id,
            "window_count": len(group),
            "predicted_connection": "bitcoin_v2" if positive.mean() >= 0.5 else "other",
            "predicted_bitcoin_window_fraction": float(positive.mean()),
            "predicted_activity_counts": json.dumps(
                group["model_prediction"].value_counts().to_dict(), sort_keys=True
            ),
        }
        if "label_activity" in group:
            actual = group["label_activity"].ne("other").unique()
            if len(actual) != 1:
                raise ValueError(f"inconsistent connection labels for {run_id}/{flow_id}")
            row["actual_connection"] = "bitcoin_v2" if actual[0] else "other"
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    test: pd.DataFrame,
    predicted,
    output_dir: Path,
    classes: list[str],
    context: dict,
    confidence=None,
) -> dict:
    truth = test["label_activity"].to_numpy(dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    labels = sorted(set(classes) | set(truth))
    columns = [
        c
        for c in (
            "run_id",
            "flow_id",
            "bucket_index",
            "timestamp",
            "label_activity",
            "noise_family",
            "traffic_class",
        )
        if c in test
    ]
    predictions = test[columns].copy()
    predictions["model_prediction"] = predicted
    if confidence is not None:
        predictions["model_confidence"] = confidence
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    confusion = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=labels), index=labels, columns=labels
    )
    confusion.index.name = "actual"
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    flows = aggregate_flows(predictions)
    flows.to_csv(output_dir / "flow_predictions.csv", index=False)
    pd.crosstab(flows["actual_connection"], flows["predicted_connection"]).to_csv(
        output_dir / "flow_confusion_matrix.csv"
    )
    noise_report = pd.DataFrame(
        columns=["noise_family", "samples", "bitcoin_false_positives", "false_positive_rate"]
    )
    if "noise_family" in predictions:
        noise = predictions[predictions["label_activity"].eq("other")].copy()
        noise["noise_family"] = noise["noise_family"].fillna("unknown")
        noise["bitcoin_false_positive"] = noise["model_prediction"].ne("other")
        if not noise.empty:
            noise_report = (
                noise.groupby("noise_family", dropna=False)
                .agg(
                    samples=("bitcoin_false_positive", "size"),
                    bitcoin_false_positives=("bitcoin_false_positive", "sum"),
                )
                .reset_index()
            )
            noise_report["false_positive_rate"] = (
                noise_report["bitcoin_false_positives"] / noise_report["samples"]
            )
    noise_report.to_csv(output_dir / "confusion_by_noise_family.csv", index=False)
    metrics = {
        **context,
        "classes_trained": classes,
        "classes_in_test": sorted(set(truth)),
        "test_samples": len(test),
        "test_flows": len(flows),
        "test_class_counts": test["label_activity"].value_counts().to_dict(),
        "accuracy": float(accuracy_score(truth, predicted)),
        "error_rate": float(np.mean(truth != predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "bitcoin_detection": detection_metrics(truth != "other", predicted != "other"),
        "flow_detection": {
            "flows": len(flows),
            **detection_metrics(
                flows["actual_connection"].eq("bitcoin_v2"),
                flows["predicted_connection"].eq("bitcoin_v2"),
            ),
        },
        "noise_confusion": noise_report.to_dict(orient="records"),
        "unseen_test_classes": sorted(set(truth) - set(classes)),
        "flow_vote_tie_policy": "bitcoin_v2 (legacy-compatible)",
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
