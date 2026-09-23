# BIP324 Traffic ML

Experimental supervised models for passive Bitcoin P2P v2 traffic analysis.

The project trains XGBoost classifiers from packet sizes, directions, timing,
and flow-level measurements exported by `bip324-traffic-analysis`. Bitcoin Core
logs are used only to create evaluation labels. They are never model features.

This is an early Warnet result, not evidence that the models generalize to
mainnet traffic.

## Repository boundary

| Project | Responsibility |
| --- | --- |
| `bip324-traffic-lab` | Generate controlled captures and validation logs |
| `bip324-traffic-analysis` | Extract passive flow, burst, and timeline features |
| `bip324-traffic-ml` | Pair labels, train models, and compare them with heuristics |

The projects remain independent. This repository reads their exported
artifacts and does not invoke either one.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

Python 3.11 or newer is required.

## Prepare data

Analyze each lab run in `bip324-traffic-analysis` first:

```bash
python bip324-traffic-analysis/run_analysis.py \
  --data-dir bip324-traffic-lab/results/run-YYYYMMDDHHMMSS \
  --output bip324-traffic-analysis/results/ml-training-001 \
  --compare-bitcoin-port-filter
```

Then pair the passive output with labels from the same run:

```bash
bip324-traffic-ml prepare \
  --run-id ml-training-001 \
  --analysis-dir ../bip324-traffic-analysis/results/ml-training-001 \
  --lab-dir ../bip324-traffic-lab/results/run-YYYYMMDDHHMMSS \
  --output-dir data/ml-training-001
```

For several runs, use a manifest:

```csv
run_id,analysis_dir,lab_dir
ml-training-001,../bip324-traffic-analysis/results/ml-training-001,../bip324-traffic-lab/results/run-20260908010101
ml-training-002,../bip324-traffic-analysis/results/ml-training-002,../bip324-traffic-lab/results/run-20260908020202
ml-training-003,../bip324-traffic-analysis/results/ml-training-003,../bip324-traffic-lab/results/run-20260908030303
```

```bash
bip324-traffic-ml prepare-many --manifest runs.csv --output-dir data
```

Every preparation writes the samples, pairing audit, and a summary of matched
and unmatched labels. Ambiguous events are retained in the audit and excluded
from training labels.

## Train

Hold out complete runs. Windows from a held-out run must never occur in the
training partition.

```bash
bip324-traffic-ml train-all data/*/event_samples.csv \
  --test-runs ml-training-019,ml-training-020 \
  --output-dir models/events
```

Train the flow-level BIP324 classifier separately:

```bash
bip324-traffic-ml train data/*/flow_samples.csv \
  --target label_bip324 \
  --test-runs ml-training-019,ml-training-020 \
  --output-dir models/bip324-flow
```

Each model directory contains `metrics.json`, `test_predictions.csv`,
`feature_importance.csv`, and the serialized XGBoost model.

## Documentation

- [Methodology](docs/methodology.md)
- [Preliminary results](docs/preliminary-results.md)

Generated captures, prepared datasets, and trained models are intentionally
ignored by Git. Publish only small, reviewed result summaries.
