import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bip324_traffic_ml.consolidate import (
    pair_connection_sessions,
    pair_log_events,
    parse_address,
    parse_flow_key,
)
from bip324_traffic_ml.train import _feature_frame


class ConsolidationTest(unittest.TestCase):
    def test_parses_flow_and_ipv6_addresses(self) -> None:
        flow = parse_flow_key("['TCP', ['10.0.0.1', 41000], ['10.0.0.2', 18444]]")
        self.assertEqual(flow.protocol, "TCP")
        self.assertEqual(flow.endpoint_b.port, 18444)
        self.assertEqual(parse_address("[2001:db8::1]:8333").ip, "2001:db8::1")

    def test_pairs_same_endpoints_with_correct_time_interval(self) -> None:
        flows = pd.DataFrame(
            [
                {
                    "flow_id": "old",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "endpoint_a_ip": "10.0.0.1",
                    "endpoint_a_port": 41000,
                    "endpoint_b_ip": "10.0.0.2",
                    "endpoint_b_port": 18444,
                },
                {
                    "flow_id": "new",
                    "start_time": 30.0,
                    "end_time": 40.0,
                    "endpoint_a_ip": "10.0.0.1",
                    "endpoint_a_port": 42000,
                    "endpoint_b_ip": "10.0.0.2",
                    "endpoint_b_port": 18444,
                },
            ]
        )
        logs = pd.DataFrame(
            [
                {
                    "timestamp": 35.0,
                    "node": "tank-0001",
                    "peer_addr": "10.0.0.2:18444",
                    "event_type": "tx_or_relay_message",
                    "message_type": "tx",
                    "message_size": 300,
                }
            ]
        )

        paired = pair_log_events(logs, flows, {"tank-0001": "10.0.0.1"}, 1.0)

        self.assertEqual(paired.iloc[0]["flow_id"], "new")
        self.assertEqual(paired.iloc[0]["pairing_status"], "matched")

    def test_detector_decisions_and_labels_are_not_model_features(self) -> None:
        frame = pd.DataFrame(
            {
                "run_id": ["run-a"],
                "flow_id": ["flow-a"],
                "bytes_total": [100],
                "bitcoin_v2_candidate": [1],
                "bip324_entropy_is_high": [1],
                "label_bip324": [1],
                "baseline_handshake": [1],
                "traffic_class": ["bitcoin_v2"],
                "noise_family": ["not_noise"],
            }
        )

        features = _feature_frame(frame)

        self.assertEqual(list(features.columns), ["bytes_total"])

    def test_pairs_transient_session_by_exact_four_tuple(self) -> None:
        flows = pd.DataFrame(
            [
                {
                    "flow_id": "flow-v1",
                    "protocol": "TCP",
                    "endpoint_a_ip": "10.0.0.1",
                    "endpoint_a_port": 42000,
                    "endpoint_b_ip": "10.0.0.2",
                    "endpoint_b_port": 18444,
                    "start_time": 99.8,
                    "end_time": 120.0,
                }
            ]
        )
        event = {
            "action": "connected",
            "timestamp": 100.0,
            "session_id": "session-1",
            "source": "tank-0001",
            "peer_id": 7,
            "local_address": "10.0.0.1:42000",
            "target_p2p_address": "10.0.0.2:18444",
            "requested_transport": "v1",
            "transport_protocol": "v1",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            lab_dir = Path(temporary_directory)
            (lab_dir / "connection-events.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )
            paired = pair_connection_sessions(lab_dir, flows)

        self.assertEqual(paired.iloc[0]["flow_id"], "flow-v1")
        self.assertEqual(paired.iloc[0]["transport_protocol"], "v1")


if __name__ == "__main__":
    unittest.main()
