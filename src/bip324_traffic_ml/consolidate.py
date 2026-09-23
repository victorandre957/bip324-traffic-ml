"""Pair passive analysis artifacts with lab-only validation labels."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_BY_EVENT = {
    "bip324_handshake": "label_handshake",
    "block_message": "label_block",
    "block_received": "label_block",
    "compact_block_message": "label_compact_block",
}

LABEL_BY_MESSAGE = {
    "inv": "label_inventory",
    "getdata": "label_request",
    "tx": "label_transaction",
}

BASELINE_BY_EVENT = {
    "bip324_handshake_candidate": "baseline_handshake",
    "block_arrival_candidate": "baseline_block",
    "compact_block_candidate": "baseline_compact_block",
    "inv_announcement_candidate": "baseline_inventory",
    "request_like_candidate": "baseline_request",
    "tx_like_candidate": "baseline_transaction",
    "transaction_relay_exchange_candidate": "baseline_transaction_sequence",
}

FLOW_LABEL_COLUMNS = ("label_bip324",)
EVENT_LABEL_COLUMNS = (
    "label_handshake",
    "label_block",
    "label_compact_block",
    "label_inventory",
    "label_request",
    "label_transaction",
    "label_transaction_sequence",
)

TIMELINE_VALUES = (
    "observed_bytes",
    "packet_count",
    "burst_duration_seconds",
    "previous_bucket_gap_seconds",
    "burst_density_bytes_per_second",
)


@dataclass(frozen=True)
class Endpoint:
    ip: str
    port: int


@dataclass(frozen=True)
class FlowKey:
    protocol: str
    endpoint_a: Endpoint
    endpoint_b: Endpoint

    @property
    def ips(self) -> frozenset[str]:
        return frozenset((self.endpoint_a.ip, self.endpoint_b.ip))

    @property
    def ports(self) -> frozenset[int]:
        return frozenset((self.endpoint_a.port, self.endpoint_b.port))


def parse_flow_key(raw_key: str) -> FlowKey:
    """Parse the structured flow key exported by traffic analysis."""
    value = ast.literal_eval(raw_key)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"invalid flow key: {raw_key}")
    protocol, endpoint_a, endpoint_b = value
    return FlowKey(
        protocol=str(protocol).upper(),
        endpoint_a=Endpoint(str(endpoint_a[0]), int(endpoint_a[1])),
        endpoint_b=Endpoint(str(endpoint_b[0]), int(endpoint_b[1])),
    )


def parse_address(raw_address: object) -> Endpoint | None:
    """Parse an IPv4 or bracketed IPv6 endpoint from validation artifacts."""
    if raw_address is None or pd.isna(raw_address):
        return None
    value = str(raw_address).strip()
    if not value:
        return None
    if value.startswith("[") and "]:" in value:
        ip, port = value[1:].rsplit("]:", 1)
    elif ":" in value:
        ip, port = value.rsplit(":", 1)
    else:
        return Endpoint(value, 0)
    try:
        return Endpoint(ip, int(port))
    except ValueError:
        return Endpoint(value, 0)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required analysis artifact not found: {path}")
    return pd.read_csv(path)


def _single_analysis_scope(frame: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in frame.columns or frame.empty:
        return frame
    all_flows = frame[frame["dataset"].astype(str).str.endswith("-all-flows")]
    return all_flows if not all_flows.empty else frame


def load_flows(analysis_dir: Path) -> pd.DataFrame:
    """Load one copy of every passive flow and expand its endpoints."""
    flows = _single_analysis_scope(_read_csv(analysis_dir / "flow_features.csv")).copy()
    parsed = flows["key"].map(parse_flow_key)
    flows["protocol"] = parsed.map(lambda key: key.protocol)
    flows["endpoint_a_ip"] = parsed.map(lambda key: key.endpoint_a.ip)
    flows["endpoint_a_port"] = parsed.map(lambda key: key.endpoint_a.port)
    flows["endpoint_b_ip"] = parsed.map(lambda key: key.endpoint_b.ip)
    flows["endpoint_b_port"] = parsed.map(lambda key: key.endpoint_b.port)
    return flows.drop_duplicates(subset=["flow_id"], keep="first")


def load_node_ips(lab_dir: Path) -> dict[str, str]:
    """Load controlled node identities for labeling, never for features."""
    metadata_path = lab_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"required lab artifact not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    addresses = metadata.get("bitcoin_node_p2p_addresses", {})
    return {
        str(node): str(details["pod_ip"])
        for node, details in addresses.items()
        if isinstance(details, dict) and details.get("pod_ip")
    }


def pair_connection_sessions(lab_dir: Path, flows: pd.DataFrame) -> pd.DataFrame:
    """Pair transient connection labels with exact captured TCP flows."""
    event_path = lab_dir / "connection-events.jsonl"
    columns = [
        "session_id",
        "node",
        "peer_id",
        "connection_type",
        "transport_protocol",
        "flow_id",
        "connected_timestamp",
        "disconnected_timestamp",
        "pairing_status",
    ]
    if not event_path.exists():
        return pd.DataFrame(columns=columns)

    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    end_by_session = {
        str(event["session_id"]): float(event["timestamp"])
        for event in events
        if event.get("action") in {
            "disconnected",
            "disconnect_unconfirmed",
            "disconnected_by_core",
            "capture_ended_while_connected",
        }
    }
    flow_endpoints = {
        str(row.flow_id): frozenset(
            (
                (str(row.endpoint_a_ip), int(row.endpoint_a_port)),
                (str(row.endpoint_b_ip), int(row.endpoint_b_port)),
            )
        )
        for row in flows.itertuples(index=False)
        if str(row.protocol) == "TCP"
    }
    flow_times = {
        str(row.flow_id): (float(row.start_time), float(row.end_time))
        for row in flows.itertuples(index=False)
    }
    rows = []
    for event in events:
        if event.get("action") != "connected":
            continue
        local = parse_address(event.get("local_address"))
        remote = parse_address(event.get("target_p2p_address"))
        connected_at = float(event["timestamp"])
        exact_endpoints = (
            frozenset(((local.ip, local.port), (remote.ip, remote.port)))
            if local is not None and remote is not None
            else None
        )
        candidates = [
            flow_id
            for flow_id, endpoints in flow_endpoints.items()
            if exact_endpoints == endpoints
            and _distance_from_interval(connected_at, *flow_times[flow_id]) <= 1.0
        ]
        observed_transport = event.get("transport_protocol")
        transport = (
            observed_transport
            if observed_transport in {"v1", "v2"}
            else event.get("requested_transport")
        )
        rows.append(
            {
                "session_id": str(event["session_id"]),
                "node": event.get("source"),
                "peer_id": event.get("peer_id"),
                "connection_type": event.get("connection_type"),
                "transport_protocol": transport,
                "flow_id": candidates[0] if len(candidates) == 1 else None,
                "connected_timestamp": connected_at,
                "disconnected_timestamp": end_by_session.get(
                    str(event["session_id"]),
                    float("inf"),
                ),
                "pairing_status": _pairing_status(len(candidates)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _pairing_status(candidate_count: int) -> str:
    if candidate_count == 1:
        return "matched"
    if candidate_count > 1:
        return "ambiguous"
    return "unmatched"


def _relevant_log_label(row: pd.Series) -> str | None:
    event_type = str(row.get("event_type", ""))
    message_type = str(row.get("message_type", "")).lower()
    return LABEL_BY_EVENT.get(event_type) or LABEL_BY_MESSAGE.get(message_type)


def pair_log_events(
    logs: pd.DataFrame,
    flows: pd.DataFrame,
    node_ips: dict[str, str],
    tolerance_seconds: float,
    session_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Associate relevant Core log events with endpoint- and time-matched flows."""
    flow_records = []
    for row in flows.itertuples(index=False):
        flow_records.append(
            {
                "flow_id": str(row.flow_id),
                "start_time": float(row.start_time),
                "end_time": float(row.end_time),
                "ips": frozenset((str(row.endpoint_a_ip), str(row.endpoint_b_ip))),
                "ports": frozenset((int(row.endpoint_a_port), int(row.endpoint_b_port))),
            }
        )

    paired_rows = []
    for log_index, row in logs.iterrows():
        label = _relevant_log_label(row)
        if label is None:
            continue
        timestamp = float(row["timestamp"])
        session_flow_id = _flow_from_session(row, timestamp, session_pairs)
        if session_flow_id is not None:
            paired_rows.append(
                {
                    "log_index": int(log_index),
                    "timestamp": timestamp,
                    "node": row.get("node"),
                    "peer_addr": row.get("peer_addr"),
                    "event_type": row.get("event_type"),
                    "message_type": row.get("message_type"),
                    "message_size": row.get("message_size"),
                    "label": label,
                    "flow_id": session_flow_id,
                    "pairing_status": "matched",
                    "pairing_reason": "session_peer_id_and_exact_endpoints",
                    "candidate_count": 1,
                    "time_distance_seconds": 0.0,
                }
            )
            continue
        node_ip = node_ips.get(str(row.get("node", "")))
        peer = parse_address(row.get("peer_addr"))
        if not node_ip or peer is None:
            paired_rows.append(_unmatched_log(log_index, row, label, "missing_endpoint"))
            continue

        expected_ips = frozenset((node_ip, peer.ip))
        candidates = []
        for flow in flow_records:
            if flow["ips"] != expected_ips:
                continue
            distance = _distance_from_interval(timestamp, flow["start_time"], flow["end_time"])
            if distance > tolerance_seconds:
                continue
            port_penalty = 0 if peer.port == 0 or peer.port in flow["ports"] else 1
            candidates.append((port_penalty, distance, flow))

        if not candidates:
            paired_rows.append(_unmatched_log(log_index, row, label, "no_flow_in_time_window"))
            continue
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        best_score = candidates[0][:2]
        equally_good = [candidate for candidate in candidates if candidate[:2] == best_score]
        status = "ambiguous" if len(equally_good) > 1 else "matched"
        selected = candidates[0][2]
        paired_rows.append(
            {
                "log_index": int(log_index),
                "timestamp": timestamp,
                "node": row.get("node"),
                "peer_addr": row.get("peer_addr"),
                "event_type": row.get("event_type"),
                "message_type": row.get("message_type"),
                "message_size": row.get("message_size"),
                "label": label,
                "flow_id": selected["flow_id"],
                "pairing_status": status,
                "pairing_reason": "endpoint_and_time",
                "candidate_count": len(candidates),
                "time_distance_seconds": best_score[1],
            }
        )
    return pd.DataFrame(paired_rows)


def _flow_from_session(
    log_row: pd.Series,
    timestamp: float,
    session_pairs: pd.DataFrame | None,
) -> str | None:
    if session_pairs is None or session_pairs.empty or pd.isna(log_row.get("peer_id")):
        return None
    try:
        peer_id = int(float(log_row.get("peer_id")))
    except (TypeError, ValueError):
        return None
    candidates = session_pairs[
        (session_pairs["pairing_status"] == "matched")
        & (session_pairs["node"].astype(str) == str(log_row.get("node")))
        & (pd.to_numeric(session_pairs["peer_id"], errors="coerce") == peer_id)
        & (session_pairs["connected_timestamp"] <= timestamp)
        & (session_pairs["disconnected_timestamp"] >= timestamp)
    ]
    return str(candidates.iloc[0]["flow_id"]) if len(candidates) == 1 else None


def _unmatched_log(log_index: int, row: pd.Series, label: str, reason: str) -> dict[str, object]:
    return {
        "log_index": int(log_index),
        "timestamp": float(row["timestamp"]),
        "node": row.get("node"),
        "peer_addr": row.get("peer_addr"),
        "event_type": row.get("event_type"),
        "message_type": row.get("message_type"),
        "message_size": row.get("message_size"),
        "label": label,
        "flow_id": None,
        "pairing_status": "unmatched",
        "pairing_reason": reason,
        "candidate_count": 0,
        "time_distance_seconds": None,
    }


def _distance_from_interval(timestamp: float, start: float, end: float) -> float:
    if start <= timestamp <= end:
        return 0.0
    return min(abs(timestamp - start), abs(timestamp - end))


def build_flow_samples(
    run_id: str,
    flows: pd.DataFrame,
    node_ips: dict[str, str],
    paired_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one passive feature row and validation label per TCP flow."""
    samples = flows[flows["protocol"] == "TCP"].copy()
    bitcoin_ips = set(node_ips.values())
    samples["run_id"] = run_id
    samples["label_bip324"] = (
        samples["endpoint_a_ip"].isin(bitcoin_ips)
        & samples["endpoint_b_ip"].isin(bitcoin_ips)
    ).astype(int)
    samples["label_source"] = "bitcoin_node_endpoints_default_v2"
    for session in paired_sessions[paired_sessions["pairing_status"] == "matched"].itertuples(
        index=False
    ):
        mask = samples["flow_id"].astype(str) == str(session.flow_id)
        samples.loc[mask, "label_bip324"] = int(str(session.transport_protocol) == "v2")
        samples.loc[mask, "label_source"] = "connection_events_observed_transport"
    samples["baseline_handshake"] = samples["bitcoin_v2_candidate"].astype(int)
    return samples


def build_event_samples(
    run_id: str,
    analysis_dir: Path,
    flows: pd.DataFrame,
    paired_logs: pd.DataFrame,
    paired_sessions: pd.DataFrame,
    scale_seconds: float,
) -> pd.DataFrame:
    """Create labeled 250 ms event windows from the passive timeline."""
    timeline = _single_analysis_scope(_read_csv(analysis_dir / "passive_timeline.csv")).copy()
    timeline = timeline[timeline["scale_seconds"].sub(scale_seconds).abs() < 1e-9]
    tcp_flow_ids = set(flows.loc[flows["protocol"] == "TCP", "flow_id"].astype(str))
    timeline = timeline[timeline["flow_id"].astype(str).isin(tcp_flow_ids)]
    if timeline.empty:
        raise ValueError(f"no TCP timeline rows found for scale {scale_seconds}")

    index_columns = ["flow_id", "bucket_index", "timestamp", "seconds_from_flow_start", "phase"]
    wide = timeline.pivot_table(
        index=index_columns,
        columns="direction",
        values=list(TIMELINE_VALUES),
        aggfunc="sum",
        fill_value=0,
    )
    wide.columns = [f"{name}_{direction}" for name, direction in wide.columns]
    samples = wide.reset_index()
    for name in TIMELINE_VALUES:
        for direction in ("a_to_b", "b_to_a"):
            column = f"{name}_{direction}"
            if column not in samples:
                samples[column] = 0.0

    samples["bytes_total"] = samples["observed_bytes_a_to_b"] + samples["observed_bytes_b_to_a"]
    samples["packets_total"] = samples["packet_count_a_to_b"] + samples["packet_count_b_to_a"]
    samples["byte_direction_ratio"] = (
        (samples["observed_bytes_a_to_b"] - samples["observed_bytes_b_to_a"]).abs()
        / samples["bytes_total"].clip(lower=1)
    )
    samples = _add_temporal_context(samples)

    static_columns = [
        "flow_id",
        "duration_seconds",
        "entropy_first_payloads",
        "first_payload_a_to_b",
        "first_payload_b_to_a",
        "initial_flight_a_to_b",
        "initial_flight_b_to_a",
        "packet_count",
        "payload_packet_count",
    ]
    samples = samples.merge(flows[static_columns], on="flow_id", how="left", validate="many_to_one")
    samples["run_id"] = run_id
    for label in EVENT_LABEL_COLUMNS:
        samples[label] = 0
    for baseline in BASELINE_BY_EVENT.values():
        samples[baseline] = 0

    v1_flow_ids = set(
        paired_sessions.loc[
            (paired_sessions["pairing_status"] == "matched")
            & (paired_sessions["transport_protocol"].astype(str) == "v1"),
            "flow_id",
        ].astype(str)
    )
    matched = paired_logs[
        (paired_logs["pairing_status"] == "matched")
        & ~paired_logs["flow_id"].astype(str).isin(v1_flow_ids)
    ].copy()
    assignments: list[tuple[int, str, float, str]] = []
    for log in matched.itertuples(index=False):
        sample_index = _nearest_sample_index(
            samples,
            str(log.flow_id),
            float(log.timestamp),
            maximum_distance_seconds=1.0,
        )
        if sample_index is not None:
            samples.loc[sample_index, str(log.label)] = 1
            assignments.append(
                (
                    sample_index,
                    str(log.flow_id),
                    float(log.timestamp),
                    str(log.message_type),
                )
            )
    _label_transaction_sequences(samples, assignments)
    _apply_passive_baselines(samples, analysis_dir)
    return samples


def _add_temporal_context(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.sort_values(["flow_id", "bucket_index"]).reset_index(drop=True)
    grouped = samples.groupby("flow_id", sort=False)
    for lag in (1, 2, 4):
        samples[f"bytes_total_lag_{lag}"] = grouped["bytes_total"].shift(lag).fillna(0)
        samples[f"bytes_total_lead_{lag}"] = grouped["bytes_total"].shift(-lag).fillna(0)
    samples["bytes_previous_4_buckets"] = grouped["bytes_total"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).sum()
    ).fillna(0)
    samples["packets_previous_4_buckets"] = grouped["packets_total"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).sum()
    ).fillna(0)
    return samples


def _nearest_sample_index(
    samples: pd.DataFrame,
    flow_id: str,
    timestamp: float,
    maximum_distance_seconds: float | None = None,
) -> int | None:
    candidates = samples[samples["flow_id"].astype(str) == flow_id]
    if candidates.empty:
        return None
    distances = (candidates["timestamp"] - timestamp).abs()
    sample_index = int(distances.idxmin())
    if (
        maximum_distance_seconds is not None
        and distances.loc[sample_index] > maximum_distance_seconds
    ):
        return None
    return sample_index


def _label_transaction_sequences(
    samples: pd.DataFrame,
    assignments: Iterable[tuple[int, str, float, str]],
) -> None:
    by_flow: dict[str, list[tuple[int, float, str]]] = {}
    for sample_index, flow_id, timestamp, message_type in assignments:
        by_flow.setdefault(flow_id, []).append((sample_index, timestamp, message_type.lower()))
    for events in by_flow.values():
        ordered = sorted(events, key=lambda event: event[1])
        for position, (sample_index, tx_time, message_type) in enumerate(ordered):
            if message_type != "tx":
                continue
            previous = ordered[:position]
            requests = [
                event
                for event in previous
                if event[2] == "getdata" and tx_time - event[1] <= 5.0
            ]
            if not requests:
                continue
            request_time = requests[-1][1]
            has_inventory = any(
                event[2] == "inv" and 0 <= request_time - event[1] <= 5.0
                for event in previous
            )
            if has_inventory:
                samples.loc[sample_index, "label_transaction_sequence"] = 1


def _apply_passive_baselines(samples: pd.DataFrame, analysis_dir: Path) -> None:
    events = _single_analysis_scope(_read_csv(analysis_dir / "passive_events.csv"))
    for event in events.itertuples(index=False):
        baseline = BASELINE_BY_EVENT.get(str(event.event_type))
        if baseline is None:
            continue
        sample_index = _nearest_sample_index(
            samples,
            str(event.flow_id),
            float(event.timestamp),
            maximum_distance_seconds=1.0,
        )
        if sample_index is not None:
            samples.loc[sample_index, baseline] = 1


def prepare_run(
    run_id: str,
    analysis_dir: Path,
    lab_dir: Path,
    output_dir: Path,
    scale_seconds: float = 0.25,
    pairing_tolerance_seconds: float = 1.0,
) -> dict[str, object]:
    """Build auditable flow and event datasets for one lab execution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flows = load_flows(analysis_dir)
    node_ips = load_node_ips(lab_dir)
    logs = _single_analysis_scope(_read_csv(analysis_dir / "log_events.csv"))
    paired_sessions = pair_connection_sessions(lab_dir, flows)
    paired_logs = pair_log_events(
        logs,
        flows,
        node_ips,
        pairing_tolerance_seconds,
        paired_sessions,
    )
    flow_samples = build_flow_samples(run_id, flows, node_ips, paired_sessions)
    event_samples = build_event_samples(
        run_id,
        analysis_dir,
        flows,
        paired_logs,
        paired_sessions,
        scale_seconds,
    )

    flow_samples.to_csv(output_dir / "flow_samples.csv", index=False)
    event_samples.to_csv(output_dir / "event_samples.csv", index=False)
    paired_logs.to_csv(output_dir / "log_flow_pairing.csv", index=False)
    paired_sessions.to_csv(output_dir / "connection_flow_pairing.csv", index=False)
    pairing_counts = paired_logs["pairing_status"].value_counts().to_dict()
    summary = {
        "run_id": run_id,
        "analysis_dir": str(analysis_dir.resolve()),
        "lab_dir": str(lab_dir.resolve()),
        "scale_seconds": scale_seconds,
        "pairing_tolerance_seconds": pairing_tolerance_seconds,
        "flow_samples": len(flow_samples),
        "event_samples": len(event_samples),
        "pairing_counts": {str(key): int(value) for key, value in pairing_counts.items()},
        "connection_pairing_counts": {
            str(key): int(value)
            for key, value in paired_sessions["pairing_status"].value_counts().to_dict().items()
        },
        "labels": {
            label: int(event_samples[label].sum())
            for label in EVENT_LABEL_COLUMNS
        },
        "ground_truth_policy": "Bitcoin Core logs and lab metadata label samples only.",
        "feature_policy": "Only passive flow and packet-timing columns are eligible for training.",
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
