"""Check temporal context construction without requiring a full training run."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class TemporalCNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "experiments" / "temporal_cnn" / "train.py"
        spec = importlib.util.spec_from_file_location("temporal_cnn_experiment", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_context_does_not_cross_flow_or_missing_bucket(self):
        frame = pd.DataFrame(
            {
                "run_id": ["a"] * 4 + ["b"],
                "flow_id": ["flow"] * 5,
                "bucket_index": [0, 1, 3, 4, 0],
            }
        )
        neighbors = self.module.neighbor_indices(frame, radius=1)
        missing = len(frame)
        self.assertEqual(
            neighbors.tolist(),
            [
                [missing, 0, 1],
                [0, 1, missing],
                [missing, 2, 3],
                [2, 3, missing],
                [missing, 4, missing],
            ],
        )

    def test_network_returns_one_class_logit_per_window(self):
        import torch

        network = self.module.TemporalCNN(feature_count=34, class_count=8)
        prediction = network(torch.zeros(3, 9, 34))
        self.assertEqual(tuple(prediction.shape), (3, 8))
        self.assertTrue(torch.isfinite(prediction).all().item())


if __name__ == "__main__":
    unittest.main()
