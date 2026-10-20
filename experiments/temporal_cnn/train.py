"""Backward-compatible script; maintained training code lives inside the package."""

from bip324_traffic_ml.training.temporal import TemporalCNN, neighbor_indices  # noqa: F401


def main():
    import sys
    from bip324_traffic_ml.cli import main as cli_main

    cli_main(["train", "--model", "temporal-cnn", *sys.argv[1:]])


if __name__ == "__main__":
    main()
