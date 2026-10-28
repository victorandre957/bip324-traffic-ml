"""Standalone SHAP figures; plotting dependencies are optional for numeric reports."""

from __future__ import annotations

import numpy as np


def render_plots(
    target, samples, features, values, bases, classes, overall, grouped, examples, seed
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    def bar(table, label_column, value_column, filename, title):
        rows = table.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(rows[label_column], rows[value_column], color="#237f9b")
        ax.set_xlabel("Mean absolute SHAP contribution (raw margin)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(target / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    bar(
        overall,
        "feature",
        "mean_abs_shap",
        "global_importance.png",
        "Population-weighted feature importance (mean over output scores)",
    )
    bar(
        grouped,
        "group",
        "mean_abs_group_shap",
        "group_importance.png",
        "Population-weighted grouped contributions",
    )
    population = samples["population_sample"].to_numpy()
    for code, label in enumerate(classes):
        explanation = shap.Explanation(
            values=values[population, :, code],
            base_values=bases[population, code],
            data=features.to_numpy()[population],
            feature_names=list(features.columns),
        )
        # Fix randomness used by SHAP's vertical jitter; do not disturb the caller.
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            shap.plots.beeswarm(explanation, max_display=15, show=False)
            plt.title(f"{label} score — class-balanced sample (unweighted)")
            plt.gcf().savefig(target / f"beeswarm_{label}.png", dpi=160, bbox_inches="tight")
            plt.close(plt.gcf())
        finally:
            np.random.set_state(state)
    # Explain score contrasts, preserving additivity for why one class beat another.
    for index in examples:
        predicted = samples.loc[index, "model_prediction"]
        actual = samples.loc[index, "label_activity"]
        p, a = classes.index(predicted), classes.index(actual)
        explanation = shap.Explanation(
            values=values[index, :, p] - values[index, :, a],
            base_values=float(bases[index, p] - bases[index, a]),
            data=features.iloc[index].to_numpy(),
            feature_names=list(features.columns),
        )
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title(f"{predicted} minus {actual}: {samples.loc[index, 'error_group']}")
        plt.gcf().savefig(target / f"waterfall_{index:04d}.png", dpi=160, bbox_inches="tight")
        plt.close(plt.gcf())
