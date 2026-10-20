"""Compatibility imports for pre-0.2 notebooks. New code uses features/labeling."""

from .labeling import (  # noqa: F401
    PREPARATION_VERSION,
    EVENT_LABEL_COLUMNS,
    activity_labels,
    build_flow_samples,
    pair_connection_sessions,
    pair_log_events,
    parse_address,
    parse_flow_key,
)
from .features import build_windows


def build_event_samples(
    run_id, analysis_dir, flow_samples, paired_logs, paired_sessions, scale_seconds
):
    import pandas as pd
    from .labeling import _single_analysis_scope, label_windows

    timeline = _single_analysis_scope(pd.read_csv(analysis_dir / "passive_timeline.csv"))
    windows = build_windows(run_id, timeline, flow_samples, scale_seconds)
    return label_windows(windows, flow_samples, paired_logs, paired_sessions)


def prepare_run(
    run_id, analysis_dir, lab_dir, output_dir, scale_seconds=0.25, pairing_tolerance_seconds=1.0
):
    from .preparation import prepare_from_features

    return prepare_from_features(
        run_id, analysis_dir, lab_dir, output_dir, scale_seconds, pairing_tolerance_seconds
    )
