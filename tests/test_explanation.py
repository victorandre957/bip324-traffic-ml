"""Check sampling weights, exact attributions and read-only post-training reports."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix, XGBClassifier

from bip324_traffic_ml.explanation.sampling import error_groups, select_samples
from bip324_traffic_ml.explanation.xgboost import ExplainConfig, explain_xgboost, tree_shap
from bip324_traffic_ml.features import FEATURE_COLUMNS, feature_frame
from helpers import activity_dataset


class ExplanationTest(unittest.TestCase):
    def test_balanced_sampling_recovers_population_and_errors_have_zero_extra_weight(self):
        frame = pd.DataFrame(
            {"label_activity": ["other"] * 90 + ["block"] * 10, "noise_family": ["https"] * 100}
        )
        predicted = np.asarray(["block"] + ["other"] * 89 + ["block"] * 10)
        groups = error_groups(frame, predicted)
        selected = select_samples(frame, groups, max_samples=10, errors_per_group=1, seed=42)
        self.assertAlmostEqual(selected.population_weight.sum(), 1)
        labels = frame.loc[selected.population_index, "label_activity"].to_numpy()
        self.assertAlmostEqual(selected.loc[labels == "other", "population_weight"].sum(), 0.9)
        self.assertAlmostEqual(selected.loc[labels == "block", "population_weight"].sum(), 0.1)
        self.assertIn(0, selected.population_index.to_list())
        self.assertTrue(selected.loc[~selected.population_sample, "population_weight"].eq(0).all())
        pd.testing.assert_frame_equal(selected, select_samples(frame, groups, 10, 1, 42))

    def test_tree_shap_matches_margins_and_unused_feature_is_zero(self):
        x = pd.DataFrame({"signal": np.tile([0, 1, 2], 40), "constant": 1.0})
        y = x.signal.to_numpy()
        model = XGBClassifier(
            n_estimators=8, max_depth=2, n_jobs=1, objective="multi:softprob", num_class=3
        ).fit(x, y)
        values, bases, raw = tree_shap(model.get_booster(), x.iloc[:7], batch_size=3)
        self.assertEqual(values.shape, (7, 2, 3))
        np.testing.assert_allclose(values.sum(axis=1) + bases, raw, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(values[:, 1, :], 0)
        self.assertGreater(np.abs(values[:, 0, :]).mean(), 0)
        np.testing.assert_allclose(
            raw, model.get_booster().predict(DMatrix(x.iloc[:7]), output_margin=True)
        )

    def test_report_uses_only_recorded_holdout_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = [root / f"run-{i}.csv" for i in range(3)]
            for i, path in enumerate(paths):
                frame = activity_dataset(path, f"run-{i}")
            model = XGBClassifier(n_estimators=3, n_jobs=1, objective="multi:softprob", num_class=8)
            classes = sorted(frame.label_activity.unique())
            model.fit(
                feature_frame(frame), pd.Categorical(frame.label_activity, categories=classes).codes
            )
            model_dir = root / "model"
            model_dir.mkdir()
            model.save_model(model_dir / "model.json")
            (model_dir / "feature_columns.json").write_text(json.dumps(list(FEATURE_COLUMNS)))
            (model_dir / "class_labels.json").write_text(json.dumps(classes))
            (model_dir / "metrics.json").write_text(
                json.dumps({"training_runs": ["run-0", "run-1"], "test_runs": ["run-2"]})
            )
            before = {p: p.read_bytes() for p in [*paths, model_dir / "model.json"]}
            config = ExplainConfig(max_samples=16, error_samples=1, plots=False, threads=1)
            result = explain_xgboost(paths, model_dir, root / "report", config=config)
            self.assertEqual(result["runs"], ["run-2"])
            self.assertEqual(result["population_windows"], 48)
            samples = pd.read_csv(root / "report/explained_samples.csv")
            self.assertEqual(set(samples.run_id), {"run-2"})
            self.assertAlmostEqual(samples.population_weight.sum(), 1)
            with np.load(root / "report/shap_values.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["values"].shape[1:], (34, 8))
                np.testing.assert_allclose(
                    saved["values"].sum(axis=1) + saved["base_values"], saved["margins"], atol=1e-4
                )
            self.assertTrue((root / "report/report.md").is_file())
            self.assertTrue((root / "report/error_contrast_importance.csv").is_file())
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                explain_xgboost(paths, model_dir, root / "report", config=config)
            with self.assertRaisesRegex(ValueError, "training data"):
                explain_xgboost(
                    paths, model_dir, root / "train-report", runs=["run-0"], config=config
                )
            self.assertFalse((root / "train-report").exists())

    def test_rejects_cnn_instead_of_claiming_tree_explanations_apply(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "CNN needs a temporal explainer"):
                explain_xgboost([], Path(d), Path(d) / "out", config=ExplainConfig(plots=False))

    def test_error_contrasts_use_predicted_minus_actual_score(self):
        from bip324_traffic_ml.explanation.xgboost import _diagnostic_tables, FEATURE_GROUPS

        samples = pd.DataFrame(
            {
                "label_activity": ["other", "handshake"],
                "model_prediction": ["handshake", "other"],
                "diagnostic_sample": [True, True],
                "error_group": ["false_positive:https", "false_negative:handshake"],
            }
        )
        values = np.zeros((2, 34, 2))
        values[0, :, 0], values[0, :, 1] = 1, 3
        values[1, :, 0], values[1, :, 1] = 2, 5
        rows, examples = _diagnostic_tables(values, samples, ["other", "handshake"])
        self.assertTrue(
            rows.loc[
                rows.error_group.eq("false_positive:https"), "mean_shap_predicted_minus_actual"
            ]
            .eq(2)
            .all()
        )
        self.assertTrue(
            rows.loc[
                rows.error_group.eq("false_negative:handshake"), "mean_shap_predicted_minus_actual"
            ]
            .eq(-3)
            .all()
        )
        self.assertEqual(set(examples), {0, 1})
        members = [column for columns in FEATURE_GROUPS.values() for column in columns]
        self.assertEqual(set(members), set(FEATURE_COLUMNS))
        self.assertEqual(len(members), len(FEATURE_COLUMNS))

    def test_grouped_permutation_has_fixed_model_and_holds_out_runs(self):
        from bip324_traffic_ml.explanation.permutation import PermutationConfig, explain_permutation

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = [root / f"run-{i}.csv" for i in range(3)]
            for i, path in enumerate(paths):
                frame = activity_dataset(path, f"run-{i}")
            classes = sorted(frame.label_activity.unique())
            model = XGBClassifier(n_estimators=4, n_jobs=1, objective="multi:softprob", num_class=8)
            model.fit(
                feature_frame(frame), pd.Categorical(frame.label_activity, categories=classes).codes
            )
            saved = root / "model"
            saved.mkdir()
            model.save_model(saved / "model.json")
            (saved / "feature_columns.json").write_text(json.dumps(list(FEATURE_COLUMNS)))
            (saved / "class_labels.json").write_text(json.dumps(classes))
            (saved / "metrics.json").write_text(
                json.dumps({"training_runs": ["run-0", "run-1"], "test_runs": ["run-2"]})
            )
            original = (saved / "model.json").read_bytes()
            result = explain_permutation(
                paths,
                saved,
                root / "importance",
                config=PermutationConfig(max_samples=16, repeats=2, plots=False, threads=1),
            )
            self.assertEqual(result["runs"], ["run-2"])
            self.assertEqual(result["sample_windows"], 16)
            self.assertEqual(len(result["importance"]), 7)
            rows = pd.read_csv(root / "importance/sample.csv")
            self.assertEqual(set(rows.run_id), {"run-2"})
            self.assertEqual((saved / "model.json").read_bytes(), original)
