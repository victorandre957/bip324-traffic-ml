"""Compatibility entry points. Training implementations live in training/."""

from .features import feature_frame
from .training.data import ACTIVITY_CLASSES, load_datasets as _load_datasets
from .training.xgboost import XGBoostConfig, train_xgboost

REQUIRED_ACTIVITY_CLASSES = frozenset(ACTIVITY_CLASSES)


def _feature_frame(frame):
    """Legacy exploratory API; production training requires the complete schema."""
    return feature_frame(frame, strict=False)


def train_activity_model(
    dataset_paths, output_dir, test_runs=None, seed=42, allow_incomplete_classes=False
):
    return train_xgboost(
        dataset_paths,
        output_dir,
        test_runs or [],
        XGBoostConfig(seed=seed),
        allow_incomplete_classes,
    )
