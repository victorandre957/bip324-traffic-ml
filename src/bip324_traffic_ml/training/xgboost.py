"""One multiclass XGBoost classifier; the test partition is evaluation only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import new_directory
from ..evaluation import evaluate
from ..features import feature_frame
from .common import save_manifest, training_context
from .data import load_datasets, split_runs


@dataclass(frozen=True)
class XGBoostConfig:
    seed: int = 42
    trees: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    threads: int = 8

    def __post_init__(self):
        if min(self.trees, self.max_depth, self.threads) < 1 or not 0 < self.learning_rate <= 1:
            raise ValueError(
                "trees, depth and threads must be positive; learning rate must be in (0, 1]"
            )


def train_xgboost(
    inputs: list[Path],
    output: Path,
    test_runs: list[str],
    config: XGBoostConfig = XGBoostConfig(),
    allow_incomplete: bool = False,
) -> dict:
    from xgboost import XGBClassifier

    with new_directory(output) as target:
        frame = load_datasets(inputs)
        split = split_runs(frame, test_runs, allow_incomplete)
        features = feature_frame(frame)
        train = frame.iloc[split.train]
        classes = sorted(train["label_activity"].unique())
        y = pd.Categorical(train["label_activity"], categories=classes).codes
        counts = np.bincount(y, minlength=len(classes))
        weights = np.minimum(len(y) / (len(classes) * counts[y]), 20)
        model = XGBClassifier(
            n_estimators=config.trees,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=len(classes),
            eval_metric="mlogloss",
            random_state=config.seed,
            n_jobs=config.threads,
        )
        print(
            f"[train] XGBoost: {len(split.train):,} training windows; {len(split.test):,} held-out windows",
            flush=True,
        )
        model.fit(features.iloc[split.train], y, sample_weight=weights)
        probabilities = model.predict_proba(features.iloc[split.test])
        predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
        model.save_model(target / "model.json")
        pd.DataFrame(
            {"feature": features.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            target / "feature_importance.csv", index=False
        )
        context = training_context(frame, split, "XGBoost", config)
        save_manifest(target, inputs, context, classes)
        return evaluate(
            frame.iloc[split.test], predicted, target, classes, context, probabilities.max(axis=1)
        )
