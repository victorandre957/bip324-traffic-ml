import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bip324_traffic_ml.preparation import prepare_lab, prepare_accepted, run_id_for_lab
from helpers import tiny_pcap


def lab_fixture(root):
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "seed": "seed-1",
                "status": "finished",
                "bitcoin_node_p2p_addresses": {
                    "tank-0001": {"pod_ip": "10.0.0.1"},
                    "tank-0002": {"pod_ip": "10.0.0.2"},
                },
            }
        )
    )
    (root / "capture-quality.json").write_text('{"accepted": true}')
    (root / "connection-events.jsonl").write_text(
        json.dumps(
            {
                "action": "connected",
                "timestamp": 1000,
                "session_id": "s1",
                "source": "tank-0001",
                "peer_id": 1,
                "local_address": "10.0.0.1:40000",
                "target_p2p_address": "10.0.0.2:18444",
                "transport_protocol": "v2",
            }
        )
        + "\n"
    )
    (root / "tank-0001-debug.log").write_text(
        "1970-01-01T00:16:40.000Z [net] Added connection to 10.0.0.2:18444 peer=1\n"
        "1970-01-01T00:16:40.100Z [net] start sending v2 handshake to peer=1\n"
        "1970-01-01T00:16:40.400Z [net] sending inv (37 bytes) peer=1\n"
    )
    tiny_pcap(root / "isp-capture.pcap")


class PreparationTest(unittest.TestCase):
    def test_extract_label_then_skip_even_without_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lab = root / "run-one"
            lab_fixture(lab)
            result = prepare_lab(lab, root / "data/seed-1", root / "features/seed-1")
            self.assertEqual(result["activity_classes"]["other"], 1)
            self.assertEqual(result["activity_classes"]["handshake"], 1)
            self.assertEqual(result["activity_classes"]["inventory"], 1)
            output = root / "data/seed-1/event_samples.csv"
            before = output.read_bytes()
            (lab / "isp-capture.pcap").unlink()
            (lab / "tank-0001-debug.log").unlink()
            with patch(
                "bip324_traffic_ml.preparation.extract_pcap",
                side_effect=AssertionError("must skip"),
            ):
                summary = prepare_accepted(root, root / "data", root / "features")
            self.assertEqual(summary["skipped_runs"], 1)
            self.assertEqual(output.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "preserved"):
                prepare_lab(lab, root / "data/seed-1", root / "features/seed-1", scale=1)
            self.assertEqual(output.read_bytes(), before)

    def test_reused_seed_gets_new_id(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = root / "data/seed-1"
            original.mkdir(parents=True)
            (original / "preparation_summary.json").write_text(
                json.dumps({"run_id": "seed-1", "lab_dir": str(root / "run-old")})
            )
            new = root / "run-new"
            self.assertEqual(run_id_for_lab("seed-1", new, root / "data", {}), "seed-1-run-new")
            self.assertEqual(
                run_id_for_lab("seed-1", root / "run-old", root / "data", {}), "seed-1"
            )

    def test_failed_capture_is_not_used(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lab_fixture(root / "run-failed")
            (root / "run-failed/capture-quality.json").write_text('{"accepted": false}')
            with self.assertRaisesRegex(ValueError, "no accepted"):
                prepare_accepted(root, root / "data", root / "features")
            self.assertFalse((root / "data").exists())

    def test_empty_logs_do_not_label_all_as_background(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lab_fixture(root / "run-one")
            (root / "run-one/tank-0001-debug.log").write_text("unparseable\n")
            with self.assertRaisesRegex(ValueError, "no Core log events"):
                prepare_lab(root / "run-one", root / "data", root / "features")
            self.assertFalse((root / "data").exists())
