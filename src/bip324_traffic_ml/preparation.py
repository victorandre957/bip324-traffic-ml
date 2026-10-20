"""Incremental orchestration: accepted lab capture -> passive data -> labeled dataset.

Completed datasets are immutable. Cached CSVs do not require the original PCAP,
analysis repository, or logs to remain available. Rebuild into a new directory.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re

import pandas as pd

from .artifacts import file_identity, new_directory, write_json
from .capture.extract import extract_pcap
from .capture.logs import BitcoinCoreLogReader
from .features import FEATURE_SCHEMA, build_windows, feature_frame
from .labeling import (
    EVENT_LABEL_COLUMNS,
    PREPARATION_VERSION,
    _single_analysis_scope,
    build_flow_samples,
    label_windows,
    load_flows,
    load_node_ips,
    load_noise_ip_families,
    pair_connection_sessions,
    pair_log_events,
)


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", run_id):
        raise ValueError(f"unsafe run ID: {run_id!r}")
    return run_id


def accepted_lab_run(lab_dir: Path) -> dict:
    try:
        metadata = json.loads((lab_dir / "metadata.json").read_text())
        quality = json.loads((lab_dir / "capture-quality.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"missing/invalid lab metadata in {lab_dir}") from error
    if metadata.get("status") != "finished" or quality.get("accepted") is not True:
        raise ValueError(f"unfinished or rejected capture: {lab_dir}")
    return metadata


def completed_dataset(
    output: Path, run_id: str, lab_dir: Path, scale: float, tolerance: float
) -> dict | None:
    if not output.exists():
        return None
    try:
        summary = json.loads((output / "preparation_summary.json").read_text())
        if (
            summary["run_id"] != run_id
            or summary["lab_dir"] != str(lab_dir.resolve())
            or summary["scale_seconds"] != scale
            or summary["pairing_tolerance_seconds"] != tolerance
        ):
            raise ValueError("source or preparation settings differ")
        for name in (
            "event_samples.csv",
            "flow_samples.csv",
            "log_flow_pairing.csv",
            "connection_flow_pairing.csv",
        ):
            if not (output / name).is_file() or (output / name).stat().st_size == 0:
                raise ValueError(f"missing artifact: {name}")
        sample = pd.read_csv(output / "event_samples.csv", nrows=1)
        feature_frame(sample)
        if "label_activity" not in sample or sample.empty:
            raise ValueError("missing activity labels or empty dataset")
        if summary.get("artifacts"):
            for name, identity in summary["artifacts"].items():
                if file_identity(output / name) != identity:
                    raise ValueError(f"artifact changed: {name}")
        return summary
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(
            f"existing dataset preserved at {output}; use a NEW output directory to rebuild ({error})"
        ) from error


def prepare_from_features(
    run_id: str,
    features_dir: Path,
    lab_dir: Path,
    output_dir: Path,
    scale: float = 0.25,
    tolerance: float = 1.0,
) -> dict:
    """Label existing passive artifacts; legacy analysis CSVs remain importable."""
    validate_run_id(run_id)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("pairing tolerance must be finite and nonnegative")
    cached = completed_dataset(output_dir, run_id, lab_dir, scale, tolerance)
    if cached is not None:
        print(f"[prepare] keeping completed dataset {run_id}", flush=True)
        return cached
    accepted_lab_run(lab_dir)
    extraction_manifest = features_dir / "extraction_summary.json"
    if extraction_manifest.is_file():
        extracted = json.loads(extraction_manifest.read_text())
        if extracted["run_id"] != run_id or extracted["scale_seconds"] != scale:
            raise ValueError("extracted features have a different run ID or window scale")
    with new_directory(output_dir) as target:
        flows = load_flows(features_dir)
        flows["observed_is_tcp"] = flows["protocol"].eq("TCP").astype(int)
        node_ips = load_node_ips(lab_dir)
        if not node_ips:
            raise ValueError(
                "lab metadata has no Bitcoin node addresses; cannot label training data"
            )
        sessions = pair_connection_sessions(lab_dir, flows)
        log_paths = sorted(set(lab_dir.glob("*-debug.log")) | set(lab_dir.glob("debug*.log")))
        if log_paths:
            events = BitcoinCoreLogReader().parse_many(run_id, log_paths)
            logs = pd.DataFrame(
                [asdict(event) for event in events],
                columns=[
                    "dataset",
                    "node",
                    "timestamp",
                    "event_type",
                    "direction",
                    "message_type",
                    "message_size",
                    "peer_id",
                    "peer_addr",
                    "raw",
                ],
            )
        elif (features_dir / "log_events.csv").is_file():
            logs = _single_analysis_scope(pd.read_csv(features_dir / "log_events.csv"))
        else:
            raise ValueError(
                "Core debug logs are required for training labels, but not for extraction/inference"
            )
        if logs.empty:
            raise ValueError(
                "no Core log events parsed; refusing to label every window as background"
            )
        paired = pair_log_events(logs, flows, node_ips, tolerance, sessions)
        flow_samples = build_flow_samples(
            run_id, flows, node_ips, sessions, load_noise_ip_families(lab_dir)
        )
        # IP ownership alone is not proof of v2 (RPC also runs between lab nodes).
        uncertain = flow_samples["label_source"].eq(
            "bitcoin_node_endpoints_default_v2"
        ) & flow_samples["label_bip324"].eq(1)
        v2_evidence = set(
            paired.loc[
                paired["pairing_status"].eq("matched") & paired["label"].eq("label_handshake"),
                "flow_id",
            ]
        )
        verified_by_log = uncertain & flow_samples["flow_id"].isin(v2_evidence)
        flow_samples.loc[verified_by_log, "label_source"] = "matched_v2_handshake_log"
        uncertain &= ~verified_by_log
        flow_samples.loc[uncertain, "label_transport"] = "unknown"
        flow_samples.loc[uncertain, "label_bip324"] = -1
        flow_samples.loc[uncertain, "label_source"] = "unverified_node_connection"
        flow_samples.loc[uncertain, "traffic_class"] = "unknown"
        if extraction_manifest.is_file():
            observations = pd.read_csv(features_dir / "observations.csv")
            if set(observations["run_id"].astype(str)) != {run_id}:
                raise ValueError("observations have a different run ID")
        else:
            timeline = _single_analysis_scope(pd.read_csv(features_dir / "passive_timeline.csv"))
            observations = build_windows(run_id, timeline, flow_samples, scale)
        feature_frame(observations)
        samples = label_windows(observations, flow_samples, paired, sessions, tolerance)
        flow_samples.to_csv(target / "flow_samples.csv", index=False)
        samples.to_csv(target / "event_samples.csv", index=False)
        paired.to_csv(target / "log_flow_pairing.csv", index=False)
        sessions.to_csv(target / "connection_flow_pairing.csv", index=False)
        summary = {
            "preparation_version": PREPARATION_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "run_id": run_id,
            "analysis_dir": str(features_dir.resolve()),
            "lab_dir": str(lab_dir.resolve()),
            "scale_seconds": scale,
            "pairing_tolerance_seconds": tolerance,
            "flow_samples": len(flows),
            "event_samples": len(samples),
            "unverified_node_flows": int(uncertain.sum()),
            "pairing_counts": paired["pairing_status"].value_counts().to_dict(),
            "connection_pairing_counts": sessions["pairing_status"].value_counts().to_dict(),
            "activity_classes": samples["label_activity"].value_counts().to_dict(),
            "labels": {label: int(samples[label].sum()) for label in EVENT_LABEL_COLUMNS},
            "ground_truth_policy": "Core logs and lab metadata label only; unknown node connections are ambiguous.",
            "feature_policy": "Explicit allowlist of 34 passive PCAP-derived measurements.",
            "scope": "all observed TCP/UDP flows; windows require payload",
        }
        # Identity paths point to the final output directory, not the staging directory.
        summary["artifacts"] = {}
        for name in (
            "event_samples.csv",
            "flow_samples.csv",
            "log_flow_pairing.csv",
            "connection_flow_pairing.csv",
        ):
            identity = file_identity(target / name)
            identity["path"] = str((output_dir / name).resolve())
            summary["artifacts"][name] = identity
        write_json(target / "preparation_summary.json", summary)
    return summary


def find_pcap(lab_dir: Path) -> Path:
    paths = sorted(
        p
        for p in lab_dir.iterdir()
        if p.name.endswith((".pcap", ".pcapng", ".pcap.gz", ".pcapng.gz"))
    )
    preferred = [p for p in paths if p.name.startswith("isp-capture.")]
    choices = preferred or paths
    if len(choices) != 1:
        raise ValueError(f"expected one bridge capture in {lab_dir}; found {len(choices)}")
    return choices[0]


def prepare_lab(
    lab_dir: Path,
    output_dir: Path,
    features_dir: Path,
    run_id: str | None = None,
    scale: float = 0.25,
    tolerance: float = 1.0,
    reuse_features: bool = False,
) -> dict:
    metadata = accepted_lab_run(lab_dir)
    run_id = validate_run_id(run_id or str(metadata.get("seed") or lab_dir.name))
    cached = completed_dataset(output_dir, run_id, lab_dir, scale, tolerance)
    if cached is not None:
        print(f"[prepare] keeping completed dataset {run_id}", flush=True)
        return cached
    if not reuse_features:
        manifest = features_dir / "extraction_summary.json"
        pcap = find_pcap(lab_dir)
        if features_dir.exists():
            if not manifest.is_file():
                raise ValueError(
                    f"existing legacy features preserved: {features_dir}; use --reuse-analysis or a new features directory"
                )
            prior = json.loads(manifest.read_text())
            if (
                prior["source"] != file_identity(pcap)
                or prior["scale_seconds"] != scale
                or prior["run_id"] != run_id
            ):
                raise ValueError(
                    "existing extraction belongs to different inputs/settings; use a new features directory"
                )
        else:
            print(f"[extract] {run_id}: reading full capture {pcap}", flush=True)
            extract_pcap(pcap, features_dir, run_id, scale)
    print(f"[label] {run_id}: pairing logs with passive observations", flush=True)
    return prepare_from_features(run_id, features_dir, lab_dir, output_dir, scale, tolerance)


def run_id_for_lab(seed: str, lab_dir: Path, output_root: Path, selected: dict) -> str:
    validate_run_id(seed)
    for candidate in (f"{seed}-{lab_dir.name}", seed):
        summary = output_root / candidate / "preparation_summary.json"
        if summary.is_file():
            previous = json.loads(summary.read_text())
            if previous.get("lab_dir") == str(lab_dir.resolve()):
                return candidate
    if seed not in selected and not (output_root / seed).exists():
        return seed
    candidate = validate_run_id(f"{seed}-{lab_dir.name}")
    if candidate in selected or (output_root / candidate).exists():
        raise ValueError(f"dataset ID collision: {candidate}; existing data preserved")
    return candidate


def prepare_accepted(
    lab_root: Path,
    output_root: Path,
    features_root: Path,
    seed_prefix: str | None = None,
    scale: float = 0.25,
    tolerance: float = 1.0,
    reuse_features: bool = False,
) -> dict:
    if not lab_root.is_dir():
        raise ValueError(f"lab directory not found: {lab_root}")
    selected, prepared, skipped = {}, [], []
    for lab in sorted(lab_root.glob("run-*")):
        try:
            metadata = accepted_lab_run(lab)
        except ValueError:
            continue
        seed = str(metadata.get("seed") or lab.name)
        if seed_prefix and not seed.startswith(seed_prefix):
            continue
        run_id = run_id_for_lab(seed, lab, output_root, selected)
        selected[run_id] = lab
    if not selected:
        raise ValueError("no accepted captures found")
    for run_id, lab in selected.items():
        output = output_root / run_id
        if completed_dataset(output, run_id, lab, scale, tolerance) is not None:
            print(f"[prepare] keeping completed dataset {run_id}", flush=True)
            skipped.append(run_id)
            continue
        prepared.append(
            prepare_lab(
                lab, output, features_root / run_id, run_id, scale, tolerance, reuse_features
            )
        )
    return {
        "prepared_runs": len(prepared),
        "skipped_runs": len(skipped),
        "skipped_seeds": skipped,
        "runs": prepared,
    }
