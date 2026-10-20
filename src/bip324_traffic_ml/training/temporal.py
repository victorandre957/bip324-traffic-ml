"""Temporal CNN backend; context is assembled per mini-batch, never materialized globally."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..artifacts import new_directory
from ..evaluation import evaluate
from ..features import feature_frame
from .common import save_manifest, training_context
from .data import load_datasets, split_runs


class TemporalCNN(nn.Module):
    """Classify the center window using a short bidirectional flow context."""

    def __init__(self, feature_count: int, class_count: int, channels: int = 96):
        super().__init__()
        self.input = nn.Conv1d(feature_count, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels, channels, kernel_size=3, padding=dilation, dilation=dilation
                    ),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(channels, channels, kernel_size=1),
                )
                for dilation in (1, 2, 1)
            ]
        )
        self.normalizers = nn.ModuleList(nn.GroupNorm(8, channels) for _ in self.blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, class_count),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.input(windows.transpose(1, 2))
        for block, normalizer in zip(self.blocks, self.normalizers):
            x = normalizer(x + block(x))
        return self.head(x[:, :, x.shape[-1] // 2])


def neighbor_indices(frame: pd.DataFrame, radius: int) -> np.ndarray:
    """Never cross a flow or a missing 250 ms bucket when constructing context."""
    group = frame.groupby(["run_id", "flow_id"], sort=False).ngroup().to_numpy()
    bucket = pd.to_numeric(frame["bucket_index"], errors="raise").to_numpy()
    row = np.arange(len(frame), dtype=np.int32)
    missing = len(frame)
    result = np.full((len(frame), 2 * radius + 1), missing, dtype=np.int32)
    for column, offset in enumerate(range(-radius, radius + 1)):
        candidate = row.astype(np.int64) + offset
        inside = (candidate >= 0) & (candidate < len(frame))
        safe = np.clip(candidate, 0, len(frame) - 1)
        valid = inside & (group[safe] == group) & (bucket[safe] - bucket == offset)
        result[valid, column] = candidate[valid]
    return result


def _normalise(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signed_log = np.sign(features) * np.log1p(np.abs(features))
    means = signed_log[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = signed_log[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    scales[scales < 1e-6] = 1.0
    normalized = np.clip((signed_log - means) / scales, -8, 8).astype(np.float32)
    normalized = np.vstack([normalized, np.zeros((1, normalized.shape[1]), dtype=np.float32)])
    return normalized, means, scales


@dataclass(frozen=True)
class CNNConfig:
    seed: int = 42
    epochs: int = 32
    batch_size: int = 1024
    learning_rate: float = 1e-3
    threads: int = 8
    device: str = "auto"
    radius: int = 4
    channels: int = 96

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.threads, self.channels) < 1 or self.radius < 0:
            raise ValueError(
                "epochs, batch size, threads and channels must be positive; radius nonnegative"
            )
        if self.channels % 8 or not 0 < self.learning_rate <= 1:
            raise ValueError("channels must be divisible by 8; learning rate must be in (0, 1]")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")


def select_device(name):
    device = torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu" if name == "auto" else name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return device


def predict_probabilities(model, normalized, neighbors, indices, batch_size, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            x = torch.from_numpy(normalized[neighbors[batch]]).to(device)
            predictions.append(model(x).softmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def train_cnn(
    inputs: list[Path],
    output: Path,
    test_runs: list[str],
    config: CNNConfig = CNNConfig(),
    allow_incomplete: bool = False,
) -> dict:
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(config.device)
    with new_directory(output) as target:
        frame = load_datasets(inputs)
        split = split_runs(frame, test_runs, allow_incomplete)
        features = feature_frame(frame)
        normalized, means, scales = _normalise(features.to_numpy(dtype=np.float32), split.train)
        neighbors = neighbor_indices(frame, config.radius)
        classes = sorted(frame.iloc[split.train]["label_activity"].unique())
        labels = pd.Categorical(frame["label_activity"], categories=classes).codes.astype(np.int64)
        counts = np.bincount(labels[split.train], minlength=len(classes))
        weights = np.minimum(len(split.train) / (len(classes) * counts), 20).astype(np.float32)
        model = TemporalCNN(features.shape[1], len(classes), config.channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-3
        )
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
        rng = np.random.default_rng(config.seed)
        history = []
        print(
            f"[train] TemporalCNN device={device}; {len(split.train):,} training windows",
            flush=True,
        )
        for epoch in range(1, config.epochs + 1):
            started = time.monotonic()
            model.train()
            shuffled = rng.permutation(split.train)
            total_loss, total_rows = 0.0, 0
            for start in range(0, len(shuffled), config.batch_size):
                batch = shuffled[start : start + config.batch_size]
                x = torch.from_numpy(normalized[neighbors[batch]]).to(device)
                y = torch.from_numpy(labels[batch]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
                total_rows += len(batch)
            history.append(total_loss / total_rows)
            print(
                f"[train] epoch={epoch}/{config.epochs} loss={history[-1]:.4f} seconds={time.monotonic()-started:.1f}",
                flush=True,
            )
        probabilities = predict_probabilities(
            model, normalized, neighbors, split.test, config.batch_size, device
        )
        torch.save(
            {
                "model_state": model.cpu().state_dict(),
                "feature_columns": list(features.columns),
                "classes": classes,
                "normalization_mean": means.tolist(),
                "normalization_scale": scales.tolist(),
                "channels": config.channels,
                "context_radius_buckets": config.radius,
                "checkpoint_version": 2,
            },
            target / "model.pt",
        )
        context = {
            **training_context(frame, split, "TemporalCNN", config),
            "device": str(device),
            "training_loss_by_epoch": history,
            "context_radius_buckets": config.radius,
        }
        save_manifest(target, inputs, context, classes)
        return evaluate(
            frame.iloc[split.test],
            np.asarray(classes)[probabilities.argmax(axis=1)],
            target,
            classes,
            context,
            probabilities.max(axis=1),
        )
