"""Pure passive feature transforms shared by preparation, training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SCHEMA = "passive-windows-v1"
PHASES = ("handshake", "post_handshake_initial", "steady_state")
TIMELINE_VALUES = (
    "observed_bytes",
    "packet_count",
    "burst_duration_seconds",
    "previous_bucket_gap_seconds",
    "burst_density_bytes_per_second",
)
FEATURE_COLUMNS = (
    "seconds_from_flow_start",
    "burst_density_bytes_per_second_a_to_b",
    "burst_density_bytes_per_second_b_to_a",
    "burst_duration_seconds_a_to_b",
    "burst_duration_seconds_b_to_a",
    "observed_bytes_a_to_b",
    "observed_bytes_b_to_a",
    "packet_count_a_to_b",
    "packet_count_b_to_a",
    "previous_bucket_gap_seconds_a_to_b",
    "previous_bucket_gap_seconds_b_to_a",
    "bytes_total",
    "packets_total",
    "byte_direction_ratio",
    "bytes_total_lag_1",
    "bytes_total_lead_1",
    "bytes_total_lag_2",
    "bytes_total_lead_2",
    "bytes_total_lag_4",
    "bytes_total_lead_4",
    "bytes_previous_4_buckets",
    "packets_previous_4_buckets",
    "duration_seconds",
    "entropy_first_payloads",
    "first_payload_a_to_b",
    "first_payload_b_to_a",
    "initial_flight_a_to_b",
    "initial_flight_b_to_a",
    "packet_count",
    "payload_packet_count",
    "observed_is_tcp",
    "phase_handshake",
    "phase_post_handshake_initial",
    "phase_steady_state",
)


def feature_frame(frame: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Select an allowlist, never discover numeric columns from a labeled table.

    Phase columns are fixed even if a partition contains no early windows.
    Missing/invalid measurements are errors, not silently converted labels or zeros.
    """
    values = frame.copy(deep=False)
    if "phase" in values:
        invalid = ~values["phase"].isin(PHASES)
        if invalid.any():
            raise ValueError("unknown or missing passive phase")
        values = values.assign(
            **{f"phase_{phase}": values["phase"].eq(phase).astype(int) for phase in PHASES}
        )
    missing = set(FEATURE_COLUMNS) - set(values)
    if strict and missing:
        raise ValueError(f"missing passive features: {', '.join(sorted(missing))}")
    columns = [name for name in FEATURE_COLUMNS if name in values]
    if not columns:
        raise ValueError("dataset has no passive numeric features")
    numeric = values[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("passive features contain NaN or infinity")
    return numeric


def build_windows(
    run_id: str, timeline: pd.DataFrame, flows: pd.DataFrame, scale_seconds: float = 0.25
) -> pd.DataFrame:
    """Pivot directional payload buckets and add retrospective flow context."""
    if scale_seconds <= 0 or not np.isfinite(scale_seconds):
        raise ValueError("scale_seconds must be finite and positive")
    timeline = timeline[timeline["scale_seconds"].sub(scale_seconds).abs() < 1e-9].copy()
    if timeline.empty:
        raise ValueError(f"no payload windows found for scale {scale_seconds}")
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
        samples["observed_bytes_a_to_b"] - samples["observed_bytes_b_to_a"]
    ).abs() / samples["bytes_total"].clip(lower=1)
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
        "observed_is_tcp",
    ]
    samples = samples.merge(flows[static_columns], on="flow_id", how="left", validate="many_to_one")
    samples["run_id"] = run_id
    return samples


def _add_temporal_context(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.sort_values(["flow_id", "bucket_index"]).reset_index(drop=True)
    grouped = samples.groupby("flow_id", sort=False)
    for lag in (1, 2, 4):
        samples[f"bytes_total_lag_{lag}"] = grouped["bytes_total"].shift(lag).fillna(0)
        samples[f"bytes_total_lead_{lag}"] = grouped["bytes_total"].shift(-lag).fillna(0)
    samples["bytes_previous_4_buckets"] = (
        grouped["bytes_total"]
        .transform(lambda values: values.shift(1).rolling(4, min_periods=1).sum())
        .fillna(0)
    )
    samples["packets_previous_4_buckets"] = (
        grouped["packets_total"]
        .transform(lambda values: values.shift(1).rolling(4, min_periods=1).sum())
        .fillna(0)
    )
    return samples
