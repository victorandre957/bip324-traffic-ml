"""Exact TreeSHAP in raw multiclass margin space, with no refitting.

XGBoost's native pred_contribs implements TreeSHAP. It avoids depending on SHAP's
model parser while using shap.Explanation and SHAP plots for presentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import environment, file_identity, new_directory, write_json
from ..evaluation import detection_metrics
from ..features import FEATURE_COLUMNS, FEATURE_SCHEMA, feature_frame
from ..training.data import ACTIVITY_CLASSES, dataset_paths, load_datasets
from .sampling import error_groups, select_samples

FEATURE_GROUPS = {
    "current_window": (
        "observed_bytes_a_to_b",
        "observed_bytes_b_to_a",
        "packet_count_a_to_b",
        "packet_count_b_to_a",
        "bytes_total",
        "packets_total",
        "byte_direction_ratio",
    ),
    "burst_timing": tuple(
        name for name in FEATURE_COLUMNS if name.startswith(("burst_", "previous_bucket_gap_"))
    ),
    "neighboring_windows": tuple(
        name
        for name in FEATURE_COLUMNS
        if name.startswith(
            ("bytes_total_lag_", "bytes_total_lead_", "bytes_previous_", "packets_previous_")
        )
    ),
    "connection_age": (
        "seconds_from_flow_start",
        "phase_handshake",
        "phase_post_handshake_initial",
        "phase_steady_state",
    ),
    "opening_exchange": (
        "entropy_first_payloads",
        "first_payload_a_to_b",
        "first_payload_b_to_a",
        "initial_flight_a_to_b",
        "initial_flight_b_to_a",
    ),
    "whole_flow": ("duration_seconds", "packet_count", "payload_packet_count"),
    "transport": ("observed_is_tcp",),
}


@dataclass(frozen=True)
class ExplainConfig:
    max_samples: int = 1024
    error_samples: int = 5
    batch_size: int = 128
    threads: int = 4
    seed: int = 42
    plots: bool = True

    def __post_init__(self):
        if min(self.max_samples, self.batch_size, self.threads) < 1 or self.error_samples < 0:
            raise ValueError(
                "max-samples, batch-size and threads must be positive; error-samples nonnegative"
            )


def tree_shap(booster, features: pd.DataFrame, batch_size: int = 128):
    """Return sample x feature x class values, per-sample bias, and raw margins."""
    from xgboost import DMatrix

    values, bases, margins = [], [], []
    for start in range(0, len(features), batch_size):
        matrix = DMatrix(features.iloc[start : start + batch_size], nthread=1)
        contributions = booster.predict(
            matrix, pred_contribs=True, approx_contribs=False, strict_shape=True
        )
        raw = booster.predict(matrix, output_margin=True, strict_shape=True)
        if contributions.ndim != 3:
            raise ValueError("expected multiclass TreeSHAP contributions")
        # Native layout is samples x classes x (features + bias).
        phi = contributions[:, :, :-1].transpose(0, 2, 1)
        bias = contributions[:, :, -1]
        if not np.allclose(phi.sum(axis=1) + bias, raw, rtol=2e-5, atol=1e-4):
            raise ValueError("TreeSHAP failed the raw-margin additivity check")
        values.append(phi)
        bases.append(bias)
        margins.append(raw)
        print(
            f"[explain] TreeSHAP {min(start + batch_size, len(features))}/{len(features)} windows",
            flush=True,
        )
    return np.concatenate(values), np.concatenate(bases), np.concatenate(margins)


def importance_tables(values: np.ndarray, samples: pd.DataFrame, classes: list[str]) -> tuple:
    """Separate population-weighted score importance from true-class diagnostics."""
    weights = samples["population_weight"].to_numpy()
    magnitude = np.average(np.abs(values), axis=0, weights=weights)
    overall = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": magnitude.mean(axis=1)})
    overall = overall.sort_values("mean_abs_shap", ascending=False)
    per_output, per_truth = [], []
    for code, label in enumerate(classes):
        for feature, mean in zip(FEATURE_COLUMNS, magnitude[:, code]):
            per_output.append(
                {"output_class": label, "feature": feature, "mean_abs_shap": float(mean)}
            )
        mask = (
            samples["label_activity"].eq(label).to_numpy() & samples["population_sample"].to_numpy()
        )
        if mask.any():
            own = np.abs(values[mask, :, code]).mean(axis=0)
            for feature, mean in zip(FEATURE_COLUMNS, own):
                per_truth.append(
                    {
                        "true_class": label,
                        "feature": feature,
                        "mean_abs_shap": float(mean),
                        "sample_windows": int(mask.sum()),
                    }
                )
    grouped = []
    for name, members in FEATURE_GROUPS.items():
        positions = [FEATURE_COLUMNS.index(member) for member in members]
        contribution = values[:, positions, :].sum(axis=1)
        grouped.append(
            {
                "group": name,
                "feature_count": len(members),
                "mean_abs_group_shap": float(
                    np.average(np.abs(contribution), axis=0, weights=weights).mean()
                ),
            }
        )
    return (
        overall,
        pd.DataFrame(per_output).sort_values(
            ["output_class", "mean_abs_shap"], ascending=[True, False]
        ),
        pd.DataFrame(
            per_truth, columns=["true_class", "feature", "mean_abs_shap", "sample_windows"]
        ).sort_values(["true_class", "mean_abs_shap"], ascending=[True, False]),
        pd.DataFrame(grouped).sort_values("mean_abs_group_shap", ascending=False),
    )


def _diagnostic_tables(values, samples, classes):
    """Explain a mistaken class relative to the actual class (a raw score contrast)."""
    rows = []
    indices = []
    for name, group in samples[samples["error_group"].ne("correct")].groupby(
        "error_group", sort=True
    ):
        selected = group.index[
            group["diagnostic_sample"] & group["label_activity"].isin(classes)
        ].to_numpy()
        if not len(selected):
            continue
        contrasts = np.stack(
            [
                values[index, :, classes.index(samples.loc[index, "model_prediction"])]
                - values[index, :, classes.index(samples.loc[index, "label_activity"])]
                for index in selected
            ]
        )
        for feature, signed, absolute in zip(
            FEATURE_COLUMNS, contrasts.mean(axis=0), np.abs(contrasts).mean(axis=0)
        ):
            rows.append(
                {
                    "error_group": name,
                    "feature": feature,
                    "mean_shap_predicted_minus_actual": float(signed),
                    "mean_abs_contrast_shap": float(absolute),
                    "sample_windows": len(selected),
                }
            )
        indices.append(int(selected[0]))
    return (
        pd.DataFrame(
            rows,
            columns=[
                "error_group",
                "feature",
                "mean_shap_predicted_minus_actual",
                "mean_abs_contrast_shap",
                "sample_windows",
            ],
        ),
        indices,
    )


def explain_xgboost(
    inputs: list[Path],
    model_dir: Path,
    output_dir: Path,
    runs: list[str] | None = None,
    config: ExplainConfig = ExplainConfig(),
) -> dict:
    from xgboost import Booster, DMatrix

    if config.plots:
        try:
            import shap  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise ValueError(
                "plots require the explain extra in the existing environment: pip install -e './bip324-traffic-ml[explain]'; or use --no-plots"
            ) from error
    model_file = model_dir / "model.json"
    if not model_file.is_file():
        raise ValueError(
            "explain currently supports multiclass XGBoost (model.json); CNN needs a temporal explainer"
        )
    metadata_path = model_dir / "training_manifest.json"
    if not metadata_path.is_file():
        metadata_path = model_dir / "metrics.json"
    metadata = json.loads(metadata_path.read_text())
    selected_runs = sorted(set(runs or metadata.get("test_runs", [])))
    if not selected_runs:
        raise ValueError("no held-out runs recorded; provide --runs explicitly")
    if set(selected_runs) & set(metadata.get("training_runs", [])):
        raise ValueError("explain expects held-out runs; requested runs include training data")
    columns = json.loads((model_dir / "feature_columns.json").read_text())
    classes = json.loads((model_dir / "class_labels.json").read_text())
    if columns != list(FEATURE_COLUMNS) or len(classes) < 2 or "other" not in classes:
        raise ValueError(
            "model must be a multiclass activity classifier with the current 34-feature schema"
        )
    if len(set(classes)) != len(classes) or not set(classes).issubset(ACTIVITY_CLASSES):
        raise ValueError("invalid saved activity class names")
    sources = dataset_paths(inputs)
    source_identities = [file_identity(path) for path in sources]
    model_identity = file_identity(model_file, checksum=True)
    with new_directory(output_dir) as target:
        frame = load_datasets(sources)
        unknown = set(selected_runs) - set(frame["run_id"].astype(str))
        if unknown:
            raise ValueError(f"explanation runs not found in datasets: {sorted(unknown)}")
        frame = frame[
            frame["run_id"].astype(str).isin(selected_runs)
            & frame["label_activity"].isin(ACTIVITY_CLASSES)
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError("no unambiguous windows to explain")
        features = feature_frame(frame)
        booster = Booster(params={"nthread": config.threads, "device": "cpu"})
        booster.load_model(model_file)
        booster.set_param({"nthread": config.threads, "device": "cpu"})
        settings = json.loads(booster.save_config())["learner"]["learner_model_param"]
        if int(settings["num_class"]) != len(classes) or booster.feature_names != columns:
            raise ValueError("saved booster does not match feature/class sidecars")
        probabilities = booster.predict(
            DMatrix(features, nthread=config.threads), strict_shape=True
        )
        predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
        groups = error_groups(frame, predicted)
        selection = select_samples(
            frame, groups, config.max_samples, config.error_samples, config.seed
        )
        indices = selection["population_index"].to_numpy()
        sample = frame.iloc[indices][
            [
                c
                for c in [
                    "run_id",
                    "flow_id",
                    "bucket_index",
                    "timestamp",
                    "label_activity",
                    "noise_family",
                ]
                if c in frame
            ]
        ].reset_index(drop=True)
        sample = pd.concat([sample, selection], axis=1)
        sample["model_prediction"] = predicted[indices]
        sample["model_confidence"] = probabilities[indices].max(axis=1)
        sample_features = features.iloc[indices].reset_index(drop=True)
        print(
            f"[explain] {len(frame):,} held-out windows; explaining {len(indices):,}; no fitting",
            flush=True,
        )
        values, bases, raw = tree_shap(booster, sample_features, config.batch_size)
        overall, per_output, per_truth, grouped = importance_tables(values, sample, classes)
        diagnostic, examples = _diagnostic_tables(values, sample, classes)
        overall.to_csv(target / "global_importance.csv", index=False)
        per_output.to_csv(target / "importance_by_output_class.csv", index=False)
        per_truth.to_csv(target / "importance_within_true_class.csv", index=False)
        grouped.to_csv(target / "group_importance.csv", index=False)
        diagnostic.to_csv(target / "error_contrast_importance.csv", index=False)
        sample.to_csv(target / "explained_samples.csv", index=False)
        sample_features.to_csv(target / "explained_features.csv", index=False)
        np.savez_compressed(
            target / "shap_values.npz",
            values=values,
            base_values=bases,
            margins=raw,
            features=sample_features.to_numpy(),
            feature_names=np.asarray(columns),
            class_names=np.asarray(classes),
        )
        if config.plots:
            from .plots import render_plots

            render_plots(
                target,
                sample,
                sample_features,
                values,
                bases,
                classes,
                overall,
                grouped,
                examples,
                config.seed,
            )
        summary = {
            "model": "XGBoost",
            "method": "exact native TreeSHAP",
            "feature_schema": FEATURE_SCHEMA,
            "output_space": "raw multiclass margin, not probability percentage points",
            "feature_perturbation": "tree_path_dependent (training path cover stored in the booster)",
            "runs": selected_runs,
            "population_windows": len(frame),
            "population_class_counts": frame["label_activity"].value_counts().to_dict(),
            "unseen_true_classes": sorted(set(frame["label_activity"]) - set(classes)),
            "explained_windows": len(sample),
            "population_sample_windows": int(sample["population_sample"].sum()),
            "sample_class_counts": sample[sample["population_sample"]]["label_activity"]
            .value_counts()
            .to_dict(),
            "sampling": "uniform within true classes, capped balanced quotas, no replacement",
            "global_weighting": "N_class / (N_population * n_sampled_class); diagnostic-only rows weight zero",
            "global_definition": "population-weighted mean(abs(SHAP)), then equal mean across output scores",
            "error_population_counts": groups[groups.ne("correct")].value_counts().to_dict(),
            "bitcoin_detection_full_population": detection_metrics(
                frame["label_activity"].ne("other"), predicted != "other"
            ),
            "max_additivity_error": float(np.max(np.abs(values.sum(axis=1) + bases - raw))),
            "top_features": overall.head(10).to_dict(orient="records"),
            "configuration": asdict(config),
            "model_source": model_identity,
            "dataset_sources": source_identities,
            "metadata_source": str(metadata_path.resolve()),
            "environment": {**environment(), **_plot_versions(config.plots)},
            "interpretation": "Model attribution, not causation or proof of mainnet fingerprinting; correlated features can share credit.",
        }
        if [file_identity(path) for path in sources] != source_identities or file_identity(
            model_file, checksum=True
        ) != model_identity:
            raise ValueError("source data/model changed while calculating explanations")
        write_json(target / "explanation_summary.json", summary)
        _write_report(target, summary, per_truth)
    return summary


def _plot_versions(enabled):
    if not enabled:
        return {}
    from importlib.metadata import version

    return {name: version(name) for name in ("shap", "matplotlib")}


def _write_report(target, summary, per_truth):
    lines = [
        "# XGBoost SHAP explanation",
        "",
        f"Explained {summary['explained_windows']:,} windows from a held-out population of {summary['population_windows']:,}.",
        "",
        "No model was fitted. Attributions sum to raw class scores, not probability changes.",
        "",
        "## Global feature ranking",
        "",
        "Population-weighted mean absolute attribution, averaged equally across output class scores.",
        "",
        "| Feature | Mean absolute SHAP |",
        "| --- | ---: |",
    ]
    lines += [
        f"| `{row['feature']}` | {row['mean_abs_shap']:.5f} |" for row in summary["top_features"]
    ]
    lines += [
        "",
        "## Within each true activity",
        "",
        "Top three inputs to that activity's own score, among sampled windows of that activity.",
        "",
    ]
    for label, group in per_truth.groupby("true_class", sort=True):
        lines.append(
            f"- **{label}:** "
            + ", ".join(
                f"`{r.feature}` ({r.mean_abs_shap:.4f})" for r in group.head(3).itertuples()
            )
        )
    lines += [
        "",
        "## Errors",
        "",
        "Counts below describe the complete supplied held-out population. Error explanations are a bounded diagnostic sample.",
        "",
    ]
    lines += [
        f"- {name}: {count}" for name, count in sorted(summary["error_population_counts"].items())
    ]
    lines += [
        "",
        "`error_contrast_importance.csv` explains predicted score minus actual score; a positive value supports the mistaken class over the actual one.",
        "",
        "## Reading the plots",
        "",
        "- `global_importance.png`: the weighted global ranking above.",
        "- `group_importance.png`: signed attributions summed within related feature groups before taking magnitudes.",
        "- `beeswarm_CLASS.png`: direction of influence on that class score; colors are feature values. These plots show the unweighted class-balanced sample.",
        "- `waterfall_*.png`: one example of each sampled error type/noise family, showing predicted-minus-actual score attribution.",
        "",
        "## Limits",
        "",
        "SHAP explains this fitted model. It does not establish causality or detection on mainnet.",
        "Correlated byte/timing measurements can share credit. Grouped sums do not remove correlation or compute a new coalition game.",
        "Future-window and completed-flow features are retrospective and cannot justify an online detector.",
        "Use validation data to choose features; repeated inspection of the holdout makes it exploratory.",
        "The sample is not an independent-window confidence interval: windows within a flow are related.",
        "",
        "Method: [XGBoost native SHAP contributions](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict); "
        "plots: [SHAP](https://shap.readthedocs.io/en/latest/tabular_examples.html).",
        "",
    ]
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
