import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import pandas as pd

from bip324_traffic_ml.capture.pcap import PcapReader
from bip324_traffic_ml.capture.extract import extract_pcap
from bip324_traffic_ml.features import feature_frame
from helpers import ethernet, tiny_pcap, write_pcap


class CaptureTest(unittest.TestCase):
    def test_full_capture_no_port_filter_or_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pcap = root / "all.pcap"
            tiny_pcap(pcap)
            summary = extract_pcap(pcap, root / "features", "test")
            observations = pd.read_csv(root / "features/observations.csv")
            self.assertEqual(summary["flow_count"], 2)
            self.assertEqual(set(observations.observed_is_tcp), {0, 1})
            self.assertEqual(feature_frame(observations).shape[1], 34)
            self.assertFalse(any(c.startswith("label_") for c in observations))
            self.assertNotIn("endpoint_a_port", observations)
            tcp = observations[observations.observed_is_tcp.eq(1)]
            self.assertEqual(tcp.bytes_total.sum(), 178)
            self.assertEqual(tcp.initial_flight_a_to_b.iloc[0], 64)
            self.assertEqual(tcp.initial_flight_b_to_a.iloc[0], 64)

    def test_retransmission_and_reused_tuple(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = ethernet("10.0.0.1", "10.0.0.2", 100, 200, bytes(64), sequence=1)
            packets = [
                (1, ethernet("10.0.0.1", "10.0.0.2", 100, 200, flags=2, sequence=0)),
                (1.1, p),
                (1.2, p),
                (2, ethernet("10.0.0.1", "10.0.0.2", 100, 200, flags=2, sequence=100)),
                (2.1, ethernet("10.0.0.1", "10.0.0.2", 100, 200, bytes(32), sequence=101)),
            ]
            write_pcap(root / "test.pcap", packets)
            result = extract_pcap(root / "test.pcap", root / "out", "test")
            self.assertEqual(result["flow_count"], 2)
            obs = pd.read_csv(root / "out/observations.csv")
            self.assertEqual(obs.bytes_total.sum(), 96)

    def test_big_endian_gzip_and_vlan(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pcap"
            packet = ethernet("1.2.3.4", "5.6.7.8", 7, 8, bytes(7), protocol=17, vlan=True)
            write_pcap(path, [(1.5, packet)], endian=">")
            compressed = path.with_suffix(".pcap.gz")
            compressed.write_bytes(gzip.compress(path.read_bytes()))
            rows = list(PcapReader(compressed).packets())
            self.assertEqual(rows[0].payload_size, 7)
            self.assertEqual(rows[0].timestamp, 1.5)

    def test_pcapng_timestamp_resolution(self):
        def block(kind, body):
            body += bytes(-len(body) % 4)
            length = len(body) + 12
            return struct.pack("<II", kind, length) + body + struct.pack("<I", length)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pcapng"
            packet = ethernet("1.2.3.4", "5.6.7.8", 7, 8, bytes(7), protocol=17)
            section = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
            interface = block(
                1, struct.pack("<HHIHH", 1, 0, 65535, 9, 1) + b"\x09\x00\x00\x00" + bytes(4)
            )
            record = block(
                6, struct.pack("<IIIII", 0, 0, 1500000000, len(packet), len(packet)) + packet
            )
            path.write_bytes(section + interface + record)
            self.assertEqual(list(PcapReader(path).packets())[0].timestamp, 1.5)

    def test_truncated_input_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tiny_pcap(root / "capture.pcap")
            data = (root / "capture.pcap").read_bytes()
            (root / "capture.pcap").write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                extract_pcap(root / "capture.pcap", root / "out", "test")
            self.assertFalse((root / "out").exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tiny_pcap(root / "capture.pcap")
            out = root / "out"
            out.mkdir()
            (out / "valuable.csv").write_text("keep me")
            with self.assertRaises(FileExistsError):
                extract_pcap(root / "capture.pcap", out, "test")
            self.assertEqual((out / "valuable.csv").read_text(), "keep me")

    def test_linux_cooked_ipv6_and_fragment_reporting(self):
        import ipaddress

        udp = struct.pack("!HHHH", 111, 222, 12, 0) + b"test"
        header = struct.pack("!IHBB", 6 << 28, len(udp), 17, 64)
        header += ipaddress.ip_address("2001:db8::1").packed
        header += ipaddress.ip_address("2001:db8::2").packed
        cooked = b"\x86\xdd" + bytes(18)
        reader = PcapReader(Path("unused"))
        packet = reader._packet(1, 276, cooked + header + udp)
        self.assertEqual(packet.protocol, "UDP")
        self.assertEqual(packet.payload_size, 4)
        self.assertEqual(packet.src, ("2001:db8::1", 111))
        fragmented = bytearray(ethernet("1.2.3.4", "5.6.7.8", 111, 222, b"x", protocol=17))
        fragmented[20:22] = b"\x20\x00"
        self.assertIsNone(reader._packet(1, 1, bytes(fragmented)))
        self.assertEqual(reader.counts["ip_fragments"], 1)
