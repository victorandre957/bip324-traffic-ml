# BIP324 Traffic ML

Classify passive TCP/UDP traffic as Bitcoin v2 activity or `other`, using one
multiclass model. This repository owns **PCAP extraction, labeling, features,
training, evaluation and inference**. Only traffic generation belongs to
`bip324-traffic-lab`. The statistical analysis repository is no longer required.

The experiment uses packet sizes, timing and direction. Core logs label the
training data; they are never classifier inputs. Results describe an offline
Warnet experiment, not validated detection on mainnet.

## Pipeline and code

```text
Traffic lab -> PCAP ----------------> extract -> observations.csv -> predict
            -> Core logs / metadata                    |              ^
                        |                              v              |
                        +---------------------------> label -> train -+
                                                               |
                                                        held-out evaluation
```

| Stage | Code | Responsibility |
| --- | --- | --- |
| Capture reading | `capture/pcap.py` | Stream PCAP/PCAPNG, including gzip; parse TCP/UDP without filtering ports or lab identities |
| Passive extraction | `capture/extract.py` | Track bidirectional sessions and aggregate payload windows |
| Feature contract | `features.py` | Build windows/context; select exactly 34 allowed numeric inputs |
| Ground truth | `capture/logs.py`, `labeling.py` | Parse Core logs and associate events/sessions with observations |
| Preparation | `preparation.py`, `artifacts.py` | Incremental orchestration and atomic outputs; preserve completed datasets |
| Data and splits | `training/data.py` | Load all supplied datasets, reject duplicates, hold out complete captures |
| Models | `training/xgboost.py`, `training/temporal.py` | Multiclass XGBoost and temporal CNN |
| Evaluation | `evaluation.py` | Shared window, flow and noise confusion metrics |
| Inference | `inference.py` | Predict from observations without logs or lab metadata |
| CLI | `cli.py` | Arguments and dispatch only |

## Environment

Use the existing `fellowship/.venv`. Commands below run from `fellowship`:

```bash
source .venv/bin/activate
python -m pip install -e ./bip324-traffic-ml
```

Python 3.11+ is required. PyTorch is optional for XGBoost; if it is already
installed in this environment, keep it. Otherwise install the `cnn` extra in the
same environment: `python -m pip install -e './bip324-traffic-ml[cnn]'`.

## Prepare new accepted lab captures

```bash
python -m bip324_traffic_ml prepare-accepted \
  --lab-root bip324-traffic-lab/results-training-v2 \
  --seed-prefix ml-v2 \
  --features-root bip324-traffic-ml/artifacts/features \
  --output-dir bip324-traffic-ml/data
```

This skips completed datasets **even if their original PCAP or old extraction
has been deleted**. New accepted captures are extracted and labeled. Reused
seeds get a capture-directory suffix, preserving earlier datasets. The old
`--analysis-root` spelling remains an alias; it does not invoke the analysis repo.

Existing outputs are never replaced. Invalid/incomplete outputs cause an
explicit error; rebuild into a **new output directory**. There is no destructive
`--force` mode. A temporary directory is published only after success. Concurrent
writers to the same destination are prevented by an exclusive lock. After a hard
process kill, inspect the PID in the sibling `.NAME.lock` and remove that lock
only once its writer is confirmed stopped; unfinished temporary directories are
not training datasets.

Already prepared `data/*/event_samples.csv` files remain directly trainable.
The raw captures, existing datasets and model directories need no migration.

## Extract and label separately

Extraction works with any supported capture and requires no logs:

```bash
python -m bip324_traffic_ml extract \
  --pcap /path/to/isp-capture.pcap \
  --run-id capture-new \
  --output-dir bip324-traffic-ml/artifacts/features/capture-new
```

It writes `observations.csv` (model-ready passive inputs and observation IDs),
`flow_features.csv`, `passive_timeline.csv` and `extraction_summary.json`.
The summary reports skipped packet categories and flows without payload.

For supervised training only, attach labels in a separate step:

```bash
python -m bip324_traffic_ml label \
  --lab-dir /path/to/accepted-lab-run \
  --run-id capture-new \
  --features-dir bip324-traffic-ml/artifacts/features/capture-new \
  --output-dir bip324-traffic-ml/data/capture-new
```

`prepare-lab` combines those two steps. `label` also accepts legacy
`flow_features.csv`, `passive_timeline.csv` and `log_events.csv` exports, without
executing the old analysis code. `--reuse-analysis` / `--reuse-features` select
that path in the preparation commands. `flow_samples.csv` and pairing reports
are for auditing; the activity model trains on `event_samples.csv`.

## Train on all existing data

Passing the data directory discovers **all** its `event_samples.csv` files,
including runs whose raw captures were removed. The output must be a new path.

```bash
python -m bip324_traffic_ml train bip324-traffic-ml/data \
  --model xgboost \
  --test-runs ml-v2-coverage-009,ml-v2-coverage-010 \
  --threads 8 \
  --output-dir bip324-traffic-ml/models/activity-v3-xgboost
```

Use the same data and holdout for the CNN:

```bash
python -m bip324_traffic_ml train bip324-traffic-ml/data \
  --model temporal-cnn \
  --test-runs ml-v2-coverage-009,ml-v2-coverage-010 \
  --device auto --epochs 32 --batch-size 1024 --threads 8 \
  --output-dir bip324-traffic-ml/models/activity-v3-cnn
```

`auto` uses CUDA when available, otherwise CPU. Context is constructed per batch
to avoid allocating a nine-window copy of the entire dataset. Defaults preserve
the earlier architectures. `train-activity` and the old CNN script remain aliases.

The model predicts `other`, `bitcoin_v2_other`, `handshake`, `block`,
`compact_block` (`cmpctblock`), `inventory` (`inv`), `request` (`getdata`) or
`transaction` (`tx`). Legacy v1 is `other`; transaction sequences become
`transaction`. Ambiguous targets are excluded. All classes must be present on
both sides unless `--allow-incomplete-classes` explicitly allows a preliminary run.

Every training output contains:

- the model, ordered feature names and class names;
- a manifest with hyperparameters, random seed, library versions, input SHA-256
  checksums, class counts and complete train/test run IDs;
- window predictions and a multiclass confusion matrix;
- precision/recall/F1, macro F1, balanced accuracy and error rates;
- Bitcoin false-positive/false-negative rates, with counts and denominators;
- flow votes, flow confusion and false positives by noise family.

The holdout above has already guided previous comparisons. Keep it for continuity,
but use a separate validation set for tuning and new untouched captures for final
claims. The commands fit fixed hyperparameters/epochs; they never choose a model
using test scores. No full retraining is performed automatically during setup.

## Predict without ground truth

After `extract`, run:

```bash
python -m bip324_traffic_ml predict \
  --observations bip324-traffic-ml/artifacts/features/capture-new/observations.csv \
  --model-dir bip324-traffic-ml/models/activity-v3-xgboost \
  --output-dir bip324-traffic-ml/artifacts/predictions/capture-new
```

Use the CNN model directory in the same command to infer with that architecture.
Predictions include each window's activity and confidence plus a connection vote.
Existing multiclass XGBoost models with the same 34-feature schema are supported.
New CNN checkpoints use safe, tensor/list-only serialization; historical checkpoints
remain untouched. See [migration notes](docs/migration-v0.2.md).

## Checks and methodology

```bash
python -m unittest discover -s bip324-traffic-ml/tests -v
```

Tests use temporary synthetic captures/datasets, not the real data directory.
They cover full-PCAP extraction, labels, duplicate rejection, train-only
normalization, output preservation, and training/inference round trips for both
architectures (CNN tests skip if PyTorch is absent).

- [Methodology and measurement limitations](docs/methodology.md)
- [Migration and compatibility](docs/migration-v0.2.md)
- [Historical preliminary results](docs/preliminary-results.md)
- [Historical CNN results](experiments/temporal_cnn/RESULTS.md)
