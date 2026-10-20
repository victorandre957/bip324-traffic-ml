# Migration to 0.2

## Preserved data

No existing PCAP, dataset, analysis artifact or model is migrated in place.
`data/*/event_samples.csv` remains the training interface. Legacy `bitcoin_v1`
and `transaction_sequence` labels are normalized in memory. Existing extraction
CSVs can be labeled using `label --features-dir ...`; no analysis executable is
called. Data whose raw captures were removed remain trainable.

`prepare-accepted` keeps completed datasets regardless of missing raw sources or
extractor-version changes. It checks source identity, requested settings, basic
schema and output presence. For new datasets it also checks recorded output size
and modification time. Changes that require relabeling need a **new destination**.
`--force` was removed to prevent overwriting prior research data.

## Command changes

| Previous | Current |
| --- | --- |
| `prepare-lab --analysis-dir ...` | Same command works; `--features-dir` is the preferred spelling |
| `prepare-accepted --analysis-root ...` | Alias retained; `--features-root` is preferred |
| `--reuse-analysis` | Alias for `--reuse-features`; imports completed CSVs only |
| `--analysis-repo`, `--analysis-python` | Removed: no external extractor process |
| `prepare --analysis-dir ...` | Alias for `label --features-dir ...` |
| `train-activity ...` | Alias for `train --model xgboost ...` |
| `experiments/temporal_cnn/train.py ...` | Thin wrapper around `train --model temporal-cnn ...` |
| `train --target ...`, `train-all`, `prepare-many` | Removed: one activity model and incremental directory discovery are the supported workflow |

Pass `data/` to training to include every prepared dataset. `--test-runs` is
explicitly required. Use new model-output directories; completed models cannot be
replaced by training. No full training is triggered by installation or preparation.

## Model compatibility

The order and semantics of the 34 passive features are versioned as
`passive-windows-v1`. Existing multiclass XGBoost models with matching feature and
class sidecars work with `predict`. Earlier binary flow/event models remain on disk
but are not accepted by the new multiclass inference interface.

New CNN checkpoints contain tensors and primitive lists and load with
`weights_only=True`. Historical CNN checkpoints contain pickled NumPy arrays and
are preserved, but the new inference command does not unpickle them. They can
still be used with a trusted historical implementation/environment. Retraining is
optional and always writes a new model directory.

The rewrite separates extraction, labeling, features and models; it does not
retroactively alter historical metrics. New preparation also stops treating every
node-to-node TCP flow as v2 without observed transport/handshake evidence. That
labeling change is recorded as preparation version 3 and does not rewrite version-2
CSVs. Use a fresh data root when comparing labeling policies.

## Verification performed for this refactor

The full existing dataset was loaded read-only: 1,496,179 windows and 262,654
flows, including 73,130 ambiguous windows. Synthetic tests cover extraction,
log pairing, preserving prior outputs, both training backends and unlabeled
inference. These small CPU tests verify the pipeline; they do not replace a new
full model evaluation or claim improved predictive performance.

A 20,000-record prefix of an existing real capture was also compared with the
previous extractor: all 34 feature columns matched across 196 flows and 510
windows (floating-point tolerance 1e-6). This checks compatibility on a bounded
sample, not every PCAP/link-layer edge case. Existing data/artifact/model files
were checked against the pre-refactor file-size and modification-time inventory.
