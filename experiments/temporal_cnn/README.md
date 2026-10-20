# Temporal CNN

The maintained implementation is now in
[`training/temporal.py`](../../src/bip324_traffic_ml/training/temporal.py).
The script in this directory is a compatibility wrapper; historical result
summaries remain in [RESULTS.md](RESULTS.md).

The CNN uses the same 34 passive inputs and eight classes as XGBoost, with a
nine-bucket flow context. Normalization is fitted only on training captures.
Contexts are gathered per batch to reduce RAM use. This remains an offline model
because the feature set contains future windows and complete-flow totals.

```bash
source .venv/bin/activate
python -m bip324_traffic_ml train bip324-traffic-ml/data \
  --model temporal-cnn \
  --test-runs ml-v2-coverage-009,ml-v2-coverage-010 \
  --device auto --epochs 32 --batch-size 1024 --threads 8 \
  --output-dir bip324-traffic-ml/models/activity-v3-cnn
```

`auto` uses CUDA when available, otherwise CPU. Choose a new output path for
each experiment. The previous holdout has already guided comparisons; final
validation needs untouched captures. See the [main README](../../README.md)
for inference, optional dependencies and the shared evaluation outputs.
