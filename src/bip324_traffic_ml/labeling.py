"""Ground truth from Core logs and lab metadata; never model inputs."""

from __future__ import annotations

import ast
from bisect import bisect_left
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PREPARATION_VERSION = 3


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

ACTIVITY_BY_LABEL = {
    "label_handshake": "handshake",
    "label_block": "block",
    "label_compact_block": "compact_block",
    "label_inventory": "inventory",
    "label_request": "request",
    "label_transaction": "transaction",
    "label_transaction_sequence": "transaction",
}
EVENT_LABEL_COLUMNS = (
    "label_handshake",
    "label_block",
    "label_compact_block",
    "label_inventory",
    "label_request",
    "label_transaction",
    "label_transaction_sequence",
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


def load_noise_ip_families(lab_dir: Path) -> dict[str, str]:
    """Load lab-only noise families for evaluation, never for model features."""
    ip_map_path = lab_dir / "ip-map.txt"
    if not ip_map_path.exists():
        return {}
    families: dict[str, str] = {}
    for line in ip_map_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[0].startswith("noise-generator-noise-"):
            continue
        family = _noise_family_from_pod_name(fields[0])
        if family is not None:
            families[fields[5]] = family
    return families


def _noise_family_from_pod_name(name: str) -> str | None:
    suffix = name.removeprefix("noise-generator-noise-")
    for prefix, family in (
        ("obfs4-", "obfs4_shaped"),
        ("streaming-", "udp_streaming"),
        ("torrent-", "bittorrent"),
        ("https-", "https"),
        ("http-", "http"),
        ("tor-", "tor"),
    ):
        if suffix.startswith(prefix):
            return family
    return None


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
        if event.get("action")
        in {
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
        if hasattr(row, "protocol") and row.protocol != "TCP":
            continue
        flow_records.append(
            {
                "flow_id": str(row.flow_id),
                "start_time": float(row.start_time),
                "end_time": float(row.end_time),
                "ips": frozenset((str(row.endpoint_a_ip), str(row.endpoint_b_ip))),
                "ports": frozenset((int(row.endpoint_a_port), int(row.endpoint_b_port))),
            }
        )

    flows_by_ips: dict[frozenset[str], list[dict[str, object]]] = {}
    for flow in flow_records:
        flows_by_ips.setdefault(flow["ips"], []).append(flow)

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
        for flow in flows_by_ips.get(expected_ips, []):
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
    return pd.DataFrame(
        paired_rows,
        columns=[
            "log_index",
            "timestamp",
            "node",
            "peer_addr",
            "event_type",
            "message_type",
            "message_size",
            "label",
            "flow_id",
            "pairing_status",
            "pairing_reason",
            "candidate_count",
            "time_distance_seconds",
        ],
    )


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
    noise_ip_families: dict[str, str],
) -> pd.DataFrame:
    """Create one passive feature row and validation label per observed IP flow."""
    samples = flows.copy()
    samples["observed_is_tcp"] = samples["protocol"].eq("TCP").astype(int)
    bitcoin_ips = set(node_ips.values())
    samples["run_id"] = run_id
    samples["label_bip324"] = (
        samples["protocol"].eq("TCP")
        & samples["endpoint_a_ip"].isin(bitcoin_ips)
        & samples["endpoint_b_ip"].isin(bitcoin_ips)
    ).astype(int)
    samples["label_source"] = samples["label_bip324"].map(
        {0: "lab_background_topology", 1: "bitcoin_node_endpoints_default_v2"}
    )
    samples["label_transport"] = samples["label_bip324"].map({0: "other", 1: "bitcoin_v2"})
    for session in paired_sessions[paired_sessions["pairing_status"] == "matched"].itertuples(
        index=False
    ):
        mask = samples["flow_id"].astype(str) == str(session.flow_id)
        transport = str(session.transport_protocol)
        if transport in {"v1", "v2"}:
            samples.loc[mask, "label_bip324"] = int(transport == "v2")
            samples.loc[mask, "label_transport"] = f"bitcoin_{transport}"
            samples.loc[mask, "label_source"] = "connection_events_observed_transport"
    if "bitcoin_v2_candidate" in samples:
        samples["baseline_handshake"] = samples["bitcoin_v2_candidate"].astype(int)
    samples["noise_family"] = samples.apply(
        lambda row: _noise_family_for_flow(
            str(row.endpoint_a_ip), str(row.endpoint_b_ip), noise_ip_families
        ),
        axis=1,
    )
    samples["traffic_class"] = samples["noise_family"].map(
        lambda family: f"noise_{family}" if family != "not_noise" else "unlabeled_non_bitcoin"
    )
    samples.loc[samples["label_transport"] == "bitcoin_v2", "traffic_class"] = "bitcoin_v2"
    samples.loc[samples["label_transport"] == "bitcoin_v1", "traffic_class"] = "bitcoin_v1"
    return samples


def _noise_family_for_flow(
    endpoint_a_ip: str, endpoint_b_ip: str, noise_ip_families: dict[str, str]
) -> str:
    families = {
        family
        for ip in (endpoint_a_ip, endpoint_b_ip)
        if (family := noise_ip_families.get(ip)) is not None
    }
    if not families:
        return "not_noise"
    if len(families) == 1:
        return next(iter(families))
    return "mixed_noise"


def label_windows(
    observations: pd.DataFrame,
    flow_samples: pd.DataFrame,
    paired_logs: pd.DataFrame,
    paired_sessions: pd.DataFrame,
    tolerance_seconds: float = 1.0,
) -> pd.DataFrame:
    """Attach targets to passive windows. All joins are explicitly many-to-one."""
    metadata = ["flow_id", "noise_family", "traffic_class", "label_transport"]
    samples = observations.merge(
        flow_samples[metadata], on="flow_id", how="left", validate="many_to_one"
    )
    for label in EVENT_LABEL_COLUMNS:
        samples[label] = 0
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
    sample_lookup = _sample_lookup(samples)
    assignments: list[tuple[int, str, float, str]] = []
    for log in matched.itertuples(index=False):
        sample_index = _nearest_sample_index(
            samples,
            str(log.flow_id),
            float(log.timestamp),
            maximum_distance_seconds=tolerance_seconds,
            lookup=sample_lookup,
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
    samples["label_activity"] = activity_labels(samples)
    return samples


def activity_labels(samples: pd.DataFrame) -> pd.Series:
    """Assign one ground-truth class per observed window; retain ambiguous rows for audit."""
    labels = samples[list(ACTIVITY_BY_LABEL)].fillna(0).astype(bool).copy()
    # These specific labels imply their generic parent, rather than conflicting with it.
    labels.loc[labels["label_compact_block"], "label_block"] = False
    labels.loc[labels["label_transaction_sequence"], "label_transaction"] = False
    active_count = labels.sum(axis=1)
    result = (
        samples["label_transport"]
        .map({"other": "other", "bitcoin_v1": "other", "bitcoin_v2": "bitcoin_v2_other"})
        .fillna("ambiguous")
    )
    for column, activity in ACTIVITY_BY_LABEL.items():
        result.loc[
            (samples["label_transport"] == "bitcoin_v2") & labels[column] & (active_count == 1)
        ] = activity
    result.loc[(samples["label_transport"] == "bitcoin_v2") & (active_count > 1)] = "ambiguous"
    # A full-block transfer commonly shares a 250 ms window with the getdata
    # that requested it. In the validated coverage captures, block message
    # bytes dominate every such window, so retain the block class. Keep a
    # simultaneous handshake ambiguous because its phase is a distinct event.
    result.loc[
        (samples["label_transport"] == "bitcoin_v2")
        & labels["label_block"]
        & ~labels["label_handshake"],
    ] = "block"
    return result


def _sample_lookup(samples: pd.DataFrame) -> dict[str, tuple[list[float], list[int]]]:
    lookup: dict[str, tuple[list[float], list[int]]] = {}
    for flow_id, group in samples.groupby("flow_id", sort=False):
        ordered = group.sort_values("timestamp")
        lookup[str(flow_id)] = (
            ordered["timestamp"].astype(float).tolist(),
            ordered.index.astype(int).tolist(),
        )
    return lookup


def _nearest_sample_index(
    samples: pd.DataFrame,
    flow_id: str,
    timestamp: float,
    maximum_distance_seconds: float | None = None,
    lookup: dict[str, tuple[list[float], list[int]]] | None = None,
) -> int | None:
    times, indices = (lookup or _sample_lookup(samples)).get(flow_id, ([], []))
    if not times:
        return None
    position = bisect_left(times, timestamp)
    choices = [p for p in (position - 1, position) if 0 <= p < len(times)]
    nearest = min(choices, key=lambda p: abs(times[p] - timestamp))
    if (
        maximum_distance_seconds is not None
        and abs(times[nearest] - timestamp) > maximum_distance_seconds
    ):
        return None
    return indices[nearest]


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
                event for event in previous if event[2] == "getdata" and tx_time - event[1] <= 5.0
            ]
            if not requests:
                continue
            request_time = requests[-1][1]
            has_inventory = any(
                event[2] == "inv" and 0 <= request_time - event[1] <= 5.0 for event in previous
            )
            if has_inventory:
                samples.loc[sample_index, "label_transaction_sequence"] = 1
