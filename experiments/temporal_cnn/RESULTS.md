# Temporal CNN result after 32 epochs

The CNN completed 32 epochs on CUDA with the same 18 training runs, two
held-out runs (`ml-v2-coverage-009` and `010`), 34 passive feature columns,
1,325,246 labeled training windows, and 97,803 labeled test windows as the
XGBoost activity model. Another 73,130 ambiguous windows were excluded as
targets but could appear as unlabeled sequence context. Both models use
completed-flow and future-window measurements, so this is an offline PCAP
comparison.

| Held-out metric | XGBoost | Temporal CNN |
| --- | ---: | ---: |
| Macro F1 | 0.764 | 0.791 |
| Balanced accuracy | 0.819 | 0.843 |
| Window accuracy | 98.58% | 98.83% |
| Bitcoin-window false positives | 55 | 73 |
| Bitcoin-window false negatives | 20 | 10 |
| Bitcoin-connection false positives | 2 | 2 |
| Bitcoin-connection false negatives | 0 | 0 |
| Handshake F1 | 0.909 | 0.906 |
| Block F1 | 0.667 | 0.730 |
| Request F1 | 0.397 | 0.444 |
| Inventory F1 | 0.898 | 0.927 |
| Transaction F1 | 0.739 | 0.736 |

The CNN now exceeds XGBoost in macro F1 and finds the same number of Bitcoin
connections with the same two false-positive flows. Its per-window Bitcoin
false-negative count falls from 20 to 10, while false positives rise from 55
to 73. Sixty of its 73 false positives are obfs4-shaped noise, compared with
26 of 55 for XGBoost. On the mainnet-like block-load run `009`, block recall
rises from 34% to 46%, but that run has only 50 labeled block windows. The
other held-out run has 19. These small samples do not establish a reliable
block improvement outside the lab.

The user reports a batch size of 1,024 for this run. The training code version
used here recorded the 32 epochs and CUDA device but did not yet save batch
size; that setting cannot be independently checked from the saved artifact.
The batch size and epoch count both changed from the previous eight-epoch CNN
run, which used the same output directory and was overwritten. Its previously
recorded macro F1 was 0.757, but the former prediction files are gone.

There is no separate validation split or early stopping in this run. Because
the same held-out captures have been inspected repeatedly, these scores are
exploratory, not an unbiased model-selection estimate. Future tuning should use
validation captures inside the training set and a new, untouched final test set
with different noise, topology, and network conditions. Later runs will save
batch size, learning rate, and seed in both the checkpoint and metrics.

The one-epoch CPU pipeline check remains in
`models/coverage-all-activity-temporal-cnn-cpu-probe/` and is not used for this
comparison. Full reports are in `models/coverage-all-activity/` and
`models/coverage-all-activity-temporal-cnn/`; model output directories are
ignored by Git.
