import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bip324_traffic_ml.features import FEATURE_COLUMNS, feature_frame
from bip324_traffic_ml.inference import predict
from bip324_traffic_ml.training.data import load_datasets, split_runs
from bip324_traffic_ml.training.xgboost import XGBoostConfig, train_xgboost
from bip324_traffic_ml.evaluation import detection_metrics, aggregate_flows
from helpers import activity_dataset


class TrainingTest(unittest.TestCase):
    def test_feature_allowlist_rejects_new_numeric_label_leakage(self):
        frame = pd.DataFrame([{c: 1 for c in FEATURE_COLUMNS}])
        for column in (
            "node_number",
            "event_timestamp",
            "label_new",
            "peer_id",
            "endpoint_a_port",
            "noise_family_code",
        ):
            frame[column] = 999
        self.assertEqual(list(feature_frame(frame)), list(FEATURE_COLUMNS))
        with self.assertRaisesRegex(ValueError, "missing passive features"):
            feature_frame(frame.drop(columns=[FEATURE_COLUMNS[0]]))
        frame[FEATURE_COLUMNS[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            feature_frame(frame)

    def test_phase_encoding_is_stable_when_categories_absent(self):
        frame = pd.DataFrame([{c: 1 for c in FEATURE_COLUMNS if not c.startswith("phase_")}])
        frame["phase"] = "steady_state"
        features = feature_frame(frame)
        self.assertEqual(features.phase_handshake.iloc[0], 0)
        self.assertEqual(features.phase_steady_state.iloc[0], 1)
        self.assertEqual(len(features.columns), 34)

    def test_all_datasets_used_and_duplicate_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(3):
                directory = root / str(i)
                directory.mkdir()
                activity_dataset(directory / "event_samples.csv", f"run-{i}")
            frame = load_datasets([root])
            split = split_runs(frame, ["run-2"])
            self.assertEqual(set(frame.iloc[split.train].run_id), {"run-0", "run-1"})
            self.assertEqual(set(frame.iloc[split.test].run_id), {"run-2"})
            with self.assertRaisesRegex(ValueError, "more than once"):
                load_datasets([root, root / "0/event_samples.csv"])

    def test_xgboost_roundtrip_prediction_without_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = [root / f"run-{i}.csv" for i in range(3)]
            for i, path in enumerate(paths):
                frame = activity_dataset(path, f"run-{i}")
            result = train_xgboost(
                paths, root / "model", ["run-2"], XGBoostConfig(trees=4, threads=1)
            )
            self.assertTrue(result["coverage_complete"])
            self.assertEqual(result["feature_count"], 34)
            self.assertEqual(result["test_flows"], 8)
            self.assertIn("false_positive_rate", result["flow_detection"])
            frame.drop(columns=["label_activity", "noise_family"]).to_csv(
                root / "observations.csv", index=False
            )
            predict(root / "observations.csv", root / "model", root / "predictions")
            scored = pd.read_csv(root / "model/test_predictions.csv")
            predicted = pd.read_csv(root / "predictions/window_predictions.csv")
            self.assertEqual(scored.model_prediction.tolist(), predicted.model_prediction.tolist())
            before = (root / "model/model.json").read_bytes()
            with self.assertRaises(FileExistsError):
                train_xgboost(paths, root / "model", ["run-2"], XGBoostConfig(trees=1, threads=1))
            self.assertEqual((root / "model/model.json").read_bytes(), before)

    def test_predict_everything_bitcoin_has_full_noise_false_positive_rate(self):
        metrics = detection_metrics([False, False, True], [True, True, True])
        self.assertEqual(metrics["false_positive_rate"], 1)
        self.assertEqual(metrics["false_negative_rate"], 0)
        rows = pd.DataFrame(
            {
                "run_id": ["a"] * 3,
                "flow_id": ["x"] * 3,
                "model_prediction": ["other", "other", "handshake"],
            }
        )
        self.assertEqual(aggregate_flows(rows).predicted_connection.iloc[0], "other")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch missing")
    def test_cnn_roundtrip_and_train_only_normalization(self):
        from bip324_traffic_ml.training.temporal import CNNConfig, _normalise, train_cnn

        normalized, means, _ = _normalise(
            np.array([[0.0], [1.0], [1e9]], dtype=np.float32), np.array([0, 1])
        )
        self.assertAlmostEqual(means[0], np.log(2) / 2, places=6)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = [root / f"run-{i}.csv" for i in range(3)]
            for i, path in enumerate(paths):
                frame = activity_dataset(path, f"run-{i}")
            result = train_cnn(
                paths,
                root / "model",
                ["run-2"],
                CNNConfig(epochs=1, batch_size=16, threads=1, device="cpu", channels=8),
            )
            self.assertEqual(result["test_flows"], 8)
            frame.drop(columns=["label_activity", "noise_family"]).to_csv(
                root / "obs.csv", index=False
            )
            predict(root / "obs.csv", root / "model", root / "predictions", batch_size=16)
            truth = pd.read_csv(root / "model/test_predictions.csv")
            inferred = pd.read_csv(root / "predictions/window_predictions.csv")
            self.assertEqual(truth.model_prediction.tolist(), inferred.model_prediction.tolist())

    def test_hidden_partial_preparation_is_not_a_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            partial = root / ".seed-pending"
            partial.mkdir()
            activity_dataset(partial / "event_samples.csv", "unfinished")
            complete = root / "seed"
            complete.mkdir()
            activity_dataset(complete / "event_samples.csv", "complete")
            frame = load_datasets([root])
            self.assertEqual(set(frame.run_id), {"complete"})
