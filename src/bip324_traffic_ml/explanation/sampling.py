"""Bounded diagnostic sampling with explicit population weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_groups(frame: pd.DataFrame, predicted: np.ndarray) -> pd.Series:
    """Group errors by activity/noise family without using this metadata as features."""
    truth = frame["label_activity"].astype(str)
    prediction = pd.Series(predicted, index=frame.index)
    result = pd.Series("correct", index=frame.index)
    noise = frame.get("noise_family", pd.Series("unknown", index=frame.index)).fillna("unknown")
    false_positive = truth.eq("other") & prediction.ne("other")
    false_negative = truth.ne("other") & prediction.eq("other")
    confused = truth.ne("other") & prediction.ne("other") & truth.ne(prediction)
    result.loc[false_positive] = "false_positive:" + noise.loc[false_positive].astype(str)
    result.loc[false_negative] = "false_negative:" + truth.loc[false_negative]
    result.loc[confused] = "activity_confusion:" + truth.loc[confused]
    return result


def select_samples(
    frame: pd.DataFrame, groups: pd.Series, max_samples: int, errors_per_group: int, seed: int
) -> pd.DataFrame:
    """Balance true classes and undo oversampling for population-wide summaries.

    Extra error examples receive zero population weight, so inspecting errors
    cannot silently change the global importance estimate.
    """
    counts = frame["label_activity"].value_counts().sort_index()
    if max_samples < len(counts) or errors_per_group < 0:
        raise ValueError(
            "max-samples must cover every observed class; error-samples must be nonnegative"
        )
    rng = np.random.default_rng(seed)
    base, weights = [], {}
    quota, remainder = divmod(max_samples, len(counts))
    for position, (label, count) in enumerate(counts.items()):
        indices = frame.index[frame["label_activity"].eq(label)].to_numpy()
        selected = rng.choice(
            indices, min(len(indices), quota + int(position < remainder)), replace=False
        )
        base.extend(selected.tolist())
        for index in selected:
            weights[int(index)] = int(count) / (len(frame) * len(selected))
    additional = set()
    for name in sorted(groups.unique()):
        if name == "correct":
            continue
        candidates = frame.index[groups.eq(name)].to_numpy()
        additional.update(
            rng.choice(candidates, min(errors_per_group, len(candidates)), replace=False).tolist()
        )
    selected = sorted(set(base) | additional)
    result = pd.DataFrame({"population_index": selected})
    result["population_sample"] = result["population_index"].isin(base)
    result["population_weight"] = result["population_index"].map(weights).fillna(0.0)
    result["diagnostic_sample"] = result["population_index"].isin(additional)
    result["error_group"] = groups.loc[selected].to_numpy()
    return result
