import tempfile
import unittest
from pathlib import Path
import pandas as pd
from bip324_traffic_ml.consolidate import (
    EVENT_LABEL_COLUMNS,
    activity_labels,
    build_flow_samples,
    build_event_samples,
)


class ActivityTest(unittest.TestCase):
    def test_single_label_covers_noise_v1_events_and_ambiguous_windows(self):
        rows = [
            ("other", []),
            ("bitcoin_v1", []),
            ("bitcoin_v2", []),
            ("bitcoin_v2", ["label_handshake"]),
            ("bitcoin_v2", ["label_block", "label_compact_block"]),
            ("bitcoin_v2", ["label_transaction", "label_transaction_sequence"]),
            ("bitcoin_v2", ["label_block", "label_request"]),
            ("bitcoin_v2", ["label_handshake", "label_block"]),
        ]
        frame = pd.DataFrame(
            {
                "label_transport": [transport for transport, _ in rows],
                **{
                    name: [int(name in active) for _, active in rows]
                    for name in EVENT_LABEL_COLUMNS
                },
            }
        )
        self.assertEqual(
            activity_labels(frame).tolist(),
            [
                "other",
                "other",
                "bitcoin_v2_other",
                "handshake",
                "compact_block",
                "transaction",
                "block",
                "ambiguous",
            ],
        )

    def test_udp_from_full_pcap_is_other_and_tcp_bitcoin_is_positive(self):
        flows = pd.DataFrame(
            [
                {
                    "flow_id": "udp",
                    "protocol": "UDP",
                    "endpoint_a_ip": "10.0.0.1",
                    "endpoint_b_ip": "10.0.0.2",
                    "bitcoin_v2_candidate": False,
                },
                {
                    "flow_id": "tcp",
                    "protocol": "TCP",
                    "endpoint_a_ip": "10.0.0.1",
                    "endpoint_b_ip": "10.0.0.2",
                    "bitcoin_v2_candidate": True,
                },
            ]
        )
        samples = build_flow_samples(
            "run",
            flows,
            {"a": "10.0.0.1", "b": "10.0.0.2"},
            pd.DataFrame(columns=["pairing_status"]),
            {},
        )
        self.assertEqual(samples["label_transport"].tolist(), ["other", "bitcoin_v2"])
        self.assertEqual(samples["observed_is_tcp"].tolist(), [0, 1])

    def test_timeline_keeps_udp_noise_and_labels_bitcoin_window(self):
        flows = pd.DataFrame(
            [
                {
                    "flow_id": flow_id,
                    "protocol": protocol,
                    "endpoint_a_ip": a,
                    "endpoint_b_ip": b,
                    "bitcoin_v2_candidate": False,
                    "duration_seconds": 1.0,
                    "entropy_first_payloads": 7.0,
                    "first_payload_a_to_b": 100,
                    "first_payload_b_to_a": 120,
                    "initial_flight_a_to_b": 100,
                    "initial_flight_b_to_a": 120,
                    "packet_count": 2,
                    "payload_packet_count": 2,
                }
                for flow_id, protocol, a, b in (
                    ("udp", "UDP", "10.0.0.9", "10.0.0.10"),
                    ("btc", "TCP", "10.0.0.1", "10.0.0.2"),
                )
            ]
        )
        sessions = pd.DataFrame(columns=["pairing_status", "transport_protocol", "flow_id"])
        flow_samples = build_flow_samples(
            "run", flows, {"a": "10.0.0.1", "b": "10.0.0.2"}, sessions, {}
        )
        timeline = pd.DataFrame(
            [
                {
                    "flow_id": flow_id,
                    "bucket_index": 0,
                    "timestamp": 10.0,
                    "seconds_from_flow_start": 0.0,
                    "phase": "handshake",
                    "direction": "a_to_b",
                    "scale_seconds": 0.25,
                    "observed_bytes": 100,
                    "packet_count": 1,
                    "burst_duration_seconds": 0.0,
                    "previous_bucket_gap_seconds": 0.0,
                    "burst_density_bytes_per_second": 400.0,
                }
                for flow_id in ("udp", "btc")
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            analysis = Path(directory)
            timeline.to_csv(analysis / "passive_timeline.csv", index=False)
            pd.DataFrame(columns=["flow_id", "timestamp", "event_type"]).to_csv(
                analysis / "passive_events.csv", index=False
            )
            logs = pd.DataFrame(
                [
                    {
                        "pairing_status": "matched",
                        "flow_id": "btc",
                        "timestamp": 10.0,
                        "label": "label_handshake",
                        "message_type": "version",
                    }
                ]
            )
            samples = build_event_samples("run", analysis, flow_samples, logs, sessions, 0.25)
        self.assertEqual(
            dict(zip(samples["flow_id"], samples["label_activity"])),
            {"udp": "other", "btc": "handshake"},
        )
