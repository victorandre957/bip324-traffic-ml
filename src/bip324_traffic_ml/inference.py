"""Predict activity from passive observations only; no lab artifacts required."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import new_directory, write_json
from .evaluation import aggregate_flows
from .features import FEATURE_COLUMNS, FEATURE_SCHEMA, feature_frame


def predict(
    observations: Path, model_dir: Path, output: Path, device: str = "cpu", batch_size: int = 1024
) -> dict:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    frame = pd.read_csv(observations)
    if frame.empty or not {"run_id", "flow_id", "bucket_index"}.issubset(frame):
        raise ValueError("observations require run_id, flow_id, bucket_index and passive features")
    frame = frame.sort_values(["run_id", "flow_id", "bucket_index"], kind="stable").reset_index(
        drop=True
    )
    features = feature_frame(frame)
    columns = json.loads((model_dir / "feature_columns.json").read_text())
    if columns != list(FEATURE_COLUMNS):
        raise ValueError("saved model has a different feature schema")
    extraction = observations.parent / "extraction_summary.json"
    if extraction.is_file() and json.loads(extraction.read_text())["scale_seconds"] != 0.25:
        raise ValueError("model requires 250 ms observations")
    with new_directory(output) as target:
        if (model_dir / "model.json").is_file():
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(model_dir / "model.json")
            classes = json.loads((model_dir / "class_labels.json").read_text())
            probabilities = model.predict_proba(features)
        elif (model_dir / "model.pt").is_file():
            import torch
            from .training.temporal import (
                TemporalCNN,
                neighbor_indices,
                predict_probabilities,
                select_device,
            )

            try:
                checkpoint = torch.load(
                    model_dir / "model.pt", map_location="cpu", weights_only=True
                )
            except pickle.UnpicklingError as error:
                raise ValueError(
                    "checkpoint uses legacy pickle objects; preserved but not loaded by safe inference. "
                    "See docs/migration-v0.2.md."
                ) from error
            if checkpoint.get("checkpoint_version") != 2:
                raise ValueError(
                    "legacy CNN checkpoint: use its original experiment environment; new training writes safe version-2 checkpoints"
                )
            classes = checkpoint["classes"]
            model = TemporalCNN(len(columns), len(classes), checkpoint["channels"])
            model.load_state_dict(checkpoint["model_state"])
            selected_device = select_device(device)
            model.to(selected_device)
            values = features.to_numpy(dtype=np.float32)
            values = np.sign(values) * np.log1p(np.abs(values))
            normalized = np.clip(
                (values - np.array(checkpoint["normalization_mean"], dtype=np.float32))
                / np.array(checkpoint["normalization_scale"], dtype=np.float32),
                -8,
                8,
            )
            normalized = np.vstack([normalized, np.zeros((1, len(columns)), dtype=np.float32)])
            neighbors = neighbor_indices(frame, checkpoint["context_radius_buckets"])
            probabilities = predict_probabilities(
                model, normalized, neighbors, np.arange(len(frame)), batch_size, selected_device
            )
        else:
            raise ValueError(f"no model.json or model.pt found in {model_dir}")
        result = frame[
            [c for c in ("run_id", "flow_id", "bucket_index", "timestamp") if c in frame]
        ].copy()
        result["model_prediction"] = np.asarray(classes)[probabilities.argmax(axis=1)]
        result["model_confidence"] = probabilities.max(axis=1)
        result.to_csv(target / "window_predictions.csv", index=False)
        flows = aggregate_flows(result)
        flows.to_csv(target / "flow_predictions.csv", index=False)
        summary = {
            "windows": len(result),
            "flows": len(flows),
            "feature_schema": FEATURE_SCHEMA,
            "model_dir": str(model_dir.resolve()),
            "label_sources": [],
            "predicted_classes": result["model_prediction"].value_counts().to_dict(),
        }
        write_json(target / "prediction_summary.json", summary)
    return summary
