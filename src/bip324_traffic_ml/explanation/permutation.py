"""Grouped input permutation for either architecture, with frozen model parameters.

CNN columns are permuted together across the entire nine-window input context.
This measures dependence under an input perturbation; it is not TreeSHAP or a
causal effect, and the mixed inputs may be outside the observed distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..artifacts import environment, file_identity, new_directory, write_json
from ..features import FEATURE_COLUMNS, feature_frame
from ..training.data import ACTIVITY_CLASSES, dataset_paths, load_datasets
from .sampling import select_samples
from .xgboost import FEATURE_GROUPS


@dataclass(frozen=True)
class PermutationConfig:
    max_samples: int = 2048
    repeats: int = 3
    batch_size: int = 1024
    threads: int = 4
    seed: int = 42
    device: str = "cpu"
    plots: bool = True

    def __post_init__(self):
        if min(self.max_samples, self.repeats, self.batch_size, self.threads) < 1:
            raise ValueError("sample size, repeats, batch size and threads must be positive")


def _model_inputs(model_dir, frame, rows, config):
    columns = json.loads((model_dir / "feature_columns.json").read_text())
    classes = json.loads((model_dir / "class_labels.json").read_text())
    if columns != list(FEATURE_COLUMNS):
        raise ValueError("model feature schema differs from passive-windows-v1")
    features = feature_frame(frame).to_numpy(dtype=np.float32)
    if (model_dir / "model.json").is_file():
        from xgboost import Booster, DMatrix

        model = Booster(model_file=model_dir / "model.json")
        model.set_param({"nthread": config.threads, "device": "cpu"})

        def predict(contexts):
            probabilities = model.predict(
                DMatrix(contexts[:, 0, :], feature_names=columns, nthread=config.threads)
            )
            return np.asarray(classes)[probabilities.argmax(axis=1)]

        return features[rows, None, :], predict, "XGBoost", "cpu"
    if not (model_dir / "model.pt").is_file():
        raise ValueError("no supported model found")
    import torch
    from ..training.temporal import TemporalCNN, neighbor_indices, select_device

    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != 2:
        raise ValueError("permutation requires a version-2 CNN checkpoint")
    torch.set_num_threads(config.threads)
    device = select_device(config.device)
    model = TemporalCNN(len(columns), len(classes), checkpoint["channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transformed = np.sign(features) * np.log1p(np.abs(features))
    transformed = np.clip(
        (transformed - np.asarray(checkpoint["normalization_mean"], dtype=np.float32))
        / np.asarray(checkpoint["normalization_scale"], dtype=np.float32),
        -8,
        8,
    )
    transformed = np.vstack([transformed, np.zeros((1, len(columns)), dtype=np.float32)])
    neighbors = neighbor_indices(frame, checkpoint["context_radius_buckets"])
    contexts = transformed[neighbors[rows]]

    def predict(contexts):
        guesses = []
        with torch.inference_mode():
            for start in range(0, len(contexts), config.batch_size):
                x = torch.from_numpy(contexts[start : start + config.batch_size]).to(device)
                guesses.append(model(x).argmax(dim=1).cpu().numpy())
        return np.asarray(classes)[np.concatenate(guesses)]

    return contexts, predict, "TemporalCNN", str(device)


def explain_permutation(
    inputs,
    model_dir: Path,
    output_dir: Path,
    runs=None,
    config: PermutationConfig = PermutationConfig(),
):
    metadata_file = model_dir / "training_manifest.json"
    if not metadata_file.is_file():
        metadata_file = model_dir / "metrics.json"
    metadata = json.loads(metadata_file.read_text())
    selected_runs = sorted(set(runs or metadata.get("test_runs", [])))
    if not selected_runs or set(selected_runs) & set(metadata.get("training_runs", [])):
        raise ValueError(
            "explicit held-out runs are required; training runs cannot be explained here"
        )
    sources = dataset_paths(inputs)
    identities = [file_identity(path) for path in sources]
    model_file = model_dir / ("model.json" if (model_dir / "model.json").is_file() else "model.pt")
    model_identity = file_identity(model_file, checksum=True)
    with new_directory(output_dir) as target:
        frame = load_datasets(sources)
        unknown = set(selected_runs) - set(frame.run_id.astype(str))
        if unknown:
            raise ValueError(f"unknown held-out runs: {sorted(unknown)}")
        # Keep ambiguous windows as CNN context, while excluding them as targets.
        frame = frame[frame.run_id.astype(str).isin(selected_runs)].reset_index(drop=True)
        population = frame[frame.label_activity.isin(ACTIVITY_CLASSES)].copy()
        population["context_row"] = population.index
        population = population.reset_index(drop=True)
        if population.empty:
            raise ValueError("no unambiguous held-out windows")
        groups = pd.Series("correct", index=population.index)
        selection = select_samples(population, groups, config.max_samples, 0, config.seed)
        sample = population.iloc[selection.population_index].reset_index(drop=True)
        weights = selection.population_weight.to_numpy()
        contexts, predict, architecture, device = _model_inputs(
            model_dir, frame, sample.context_row.to_numpy(), config
        )
        classes = sorted(
            set(json.loads((model_dir / "class_labels.json").read_text()))
            | set(sample.label_activity)
        )
        truth = sample.label_activity.to_numpy()

        def score(predicted):
            return {
                "macro_f1": float(
                    f1_score(
                        truth,
                        predicted,
                        labels=classes,
                        average="macro",
                        sample_weight=weights,
                        zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(truth, predicted, sample_weight=weights)),
            }

        base = score(predict(contexts))
        permutations = [
            np.random.default_rng(config.seed + i).permutation(len(sample))
            for i in range(config.repeats)
        ]
        results = []
        for name, members in FEATURE_GROUPS.items():
            columns = [FEATURE_COLUMNS.index(column) for column in members]
            for repeat, permutation in enumerate(permutations):
                perturbed = contexts.copy()
                perturbed[:, :, columns] = contexts[permutation][:, :, columns]
                measured = score(predict(perturbed))
                results.append(
                    {
                        "group": name,
                        "repeat": repeat + 1,
                        "features": len(columns),
                        "macro_f1_drop": base["macro_f1"] - measured["macro_f1"],
                        "accuracy_drop": base["accuracy"] - measured["accuracy"],
                    }
                )
            print(f"[permutation] {architecture}: {name} done", flush=True)
        repeats = pd.DataFrame(results)
        importance = (
            repeats.groupby("group")
            .agg(
                features=("features", "first"),
                macro_f1_drop_mean=("macro_f1_drop", "mean"),
                macro_f1_drop_std=("macro_f1_drop", "std"),
                accuracy_drop_mean=("accuracy_drop", "mean"),
                accuracy_drop_std=("accuracy_drop", "std"),
            )
            .fillna(0)
            .reset_index()
            .sort_values("macro_f1_drop_mean", ascending=False)
        )
        repeats.to_csv(target / "permutation_repeats.csv", index=False)
        importance.to_csv(target / "permutation_importance.csv", index=False)
        sample[["run_id", "flow_id", "bucket_index", "label_activity"]].assign(
            population_weight=weights
        ).to_csv(target / "sample.csv", index=False)
        summary = {
            "model": architecture,
            "method": "grouped input permutation",
            "device": device,
            "runs": selected_runs,
            "population_windows": len(population),
            "sample_windows": len(sample),
            "sample_baseline": base,
            "configuration": asdict(config),
            "weighting": "true-class prevalence correction for class-balanced sampling",
            "context_policy": "each feature group is permuted across samples jointly at every context offset",
            "interpretation": "sampled score drops under perturbation, not causal effects or SHAP values; correlated unpermuted inputs may substitute",
            "importance": importance.to_dict(orient="records"),
            "environment": environment(),
            "model_source": model_identity,
            "dataset_sources": identities,
        }
        if identities != [
            file_identity(path) for path in sources
        ] or model_identity != file_identity(model_file, checksum=True):
            raise ValueError("inputs changed during explanation")
        if config.plots:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rows = importance.iloc[::-1]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(rows.group, rows.macro_f1_drop_mean * 100, xerr=rows.macro_f1_drop_std * 100)
            ax.set_xlabel("Macro F1 drop (percentage points; mean ± SD across permutations)")
            ax.set_title(
                f"{architecture}: grouped input permutation ({len(sample):,} sampled windows)"
            )
            fig.tight_layout()
            fig.savefig(target / "permutation_importance.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        write_json(target / "permutation_summary.json", summary)
    return summary
