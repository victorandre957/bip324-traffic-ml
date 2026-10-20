"""PCAP -> passive flow statistics and directional payload windows, without logs.

State scales with flows and nonempty windows, not packet count. The v1 feature
contract intentionally preserves the historical high-water TCP byte accounting
and lexicographic endpoint orientation so existing datasets remain usable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd

from ..artifacts import file_identity, new_directory, write_json
from ..features import FEATURE_SCHEMA, build_windows, feature_frame
from .pcap import Packet, PcapReader

DIRECTIONS = ("a_to_b", "b_to_a")


def phase(age: float) -> str:
    return "handshake" if age < 1 else "post_handshake_initial" if age < 5 else "steady_state"


@dataclass(slots=True)
class Bucket:
    first: float
    last: float
    size: int = 0
    count: int = 0


@dataclass(slots=True)
class Flow:
    key: tuple
    session: int
    start: float
    end: float
    packets: int = 0
    payload_packets: int = 0
    first_payload: list[int] = field(default_factory=lambda: [0, 0])
    flights: list[int] = field(default_factory=lambda: [0, 0])
    flight_direction: int = -1
    turns: int = 0
    prefix: bytearray = field(default_factory=bytearray)
    syns: dict = field(default_factory=dict)
    highwater: dict = field(default_factory=dict)
    buckets: dict = field(default_factory=dict)

    @property
    def flow_id(self):
        protocol, a, b = self.key
        value = f"{protocol}|{a[0]}:{a[1]}|{b[0]}:{b[1]}|session={self.session}"
        return hashlib.sha1(value.encode()).hexdigest()[:12]

    def add(self, packet: Packet, scale: float):
        if packet.timestamp < self.end:
            raise ValueError(
                "out-of-order capture timestamps within a flow; sort the capture first"
            )
        self.end = packet.timestamp
        self.packets += 1
        direction = int(packet.src != self.key[1])
        if packet.flags & 2:
            self.syns.setdefault(packet.src, set()).add(packet.sequence)
        size = packet.payload_size
        if packet.protocol == "TCP" and size:
            highest = self.highwater.get(direction, packet.sequence)
            effective = max(0, packet.sequence + size - max(packet.sequence, highest))
            self.highwater[direction] = max(highest, packet.sequence + size)
            size = effective
        if not size:
            return
        self.payload_packets += 1
        if self.payload_packets <= 12:
            self.prefix.extend(packet.prefix)
        if not self.first_payload[direction]:
            self.first_payload[direction] = size
        if self.flight_direction < 0:
            self.flight_direction = direction
        elif direction != self.flight_direction:
            self.turns += 1
            self.flight_direction = direction
        if self.turns < 2:
            self.flights[direction] += size
        index = int((packet.timestamp - self.start) / scale)
        bucket = self.buckets.setdefault(
            (index, direction), Bucket(packet.timestamp, packet.timestamp)
        )
        bucket.last = packet.timestamp
        bucket.size += size
        bucket.count += 1

    def feature_row(self, run_id):
        counts = Counter(self.prefix)
        total = len(self.prefix)
        entropy = -sum(n / total * math.log2(n / total) for n in counts.values()) if total else 0.0
        return {
            "dataset": run_id,
            "flow_id": self.flow_id,
            "key": json.dumps(self.key),
            "start_time": self.start,
            "end_time": self.end,
            "duration_seconds": self.end - self.start,
            "packet_count": self.packets,
            "payload_packet_count": self.payload_packets,
            "entropy_first_payloads": entropy,
            "first_payload_a_to_b": self.first_payload[0],
            "first_payload_b_to_a": self.first_payload[1],
            "initial_flight_a_to_b": self.flights[0],
            "initial_flight_b_to_a": self.flights[1],
            "observed_is_tcp": int(self.key[0] == "TCP"),
        }

    def timeline(self, run_id, scale):
        previous = {}
        for (index, direction), bucket in sorted(self.buckets.items()):
            duration = bucket.last - bucket.first
            yield {
                "dataset": run_id,
                "flow_id": self.flow_id,
                "scale_seconds": scale,
                "bucket_index": index,
                "timestamp": self.start + index * scale,
                "seconds_from_flow_start": self.start + index * scale - self.start,
                "direction": DIRECTIONS[direction],
                "observed_bytes": bucket.size,
                "packet_count": bucket.count,
                "burst_duration_seconds": duration,
                "previous_bucket_gap_seconds": max(
                    0.0, bucket.first - previous.get(direction, bucket.first)
                ),
                "burst_density_bytes_per_second": bucket.size / max(duration, scale, 0.001),
                "phase": phase(bucket.first - self.start),
            }
            previous[direction] = bucket.last


def extract_pcap(pcap: Path, output_dir: Path, run_id: str, scale: float = 0.25) -> dict:
    """Write unlabeled observations.csv, flow_features.csv and passive_timeline.csv."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    reader = PcapReader(pcap)
    source = file_identity(pcap)
    with new_directory(output_dir) as target:
        flows, current, sessions = [], {}, Counter()
        last_progress = time.monotonic()
        for packet in reader.packets():
            endpoints = sorted((packet.src, packet.dst))
            key = (packet.protocol, *endpoints)
            flow = current.get(key)
            new_session = (
                flow is not None
                and packet.protocol == "TCP"
                and packet.flags & 2
                and packet.src in flow.syns
                and packet.sequence not in flow.syns[packet.src]
            )
            if flow is None or new_session:
                flow = Flow(key, sessions[key], packet.timestamp, packet.timestamp)
                sessions[key] += 1
                current[key] = flow
                flows.append(flow)
            flow.add(packet, scale)
            if time.monotonic() - last_progress >= 30:
                print(
                    f"[extract] {reader.counts['records']:,} records; {len(flows):,} flows",
                    flush=True,
                )
                last_progress = time.monotonic()
        if file_identity(pcap) != source:
            raise ValueError("capture changed during extraction")
        if not flows or not any(flow.buckets for flow in flows):
            raise ValueError("capture contains no usable TCP/UDP payload windows")
        flow_rows = pd.DataFrame(flow.feature_row(run_id) for flow in flows)
        timeline = pd.DataFrame(row for flow in flows for row in flow.timeline(run_id, scale))
        observations = build_windows(run_id, timeline, flow_rows, scale)
        feature_frame(observations)
        flow_rows.to_csv(target / "flow_features.csv", index=False)
        timeline.to_csv(target / "passive_timeline.csv", index=False)
        observations.to_csv(target / "observations.csv", index=False)
        summary = {
            "run_id": run_id,
            "feature_schema": FEATURE_SCHEMA,
            "scale_seconds": scale,
            "source": source,
            "flow_count": len(flows),
            "window_count": len(observations),
            "flows_without_payload": sum(not f.buckets for f in flows),
            "packet_counts": dict(reader.counts),
            "scope": "all observed TCP/UDP flows",
            "label_sources": [],
            "extractor_version": 1,
            "tcp_payload_accounting": "v1 high-water sequence accounting (legacy-compatible)",
        }
        write_json(target / "extraction_summary.json", summary)
    return summary
