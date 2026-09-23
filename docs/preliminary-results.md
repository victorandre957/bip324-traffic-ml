# Preliminary results

## Status

These results are a pipeline and feasibility check from three ten-minute
Warnet captures. They are not a final benchmark and should not be generalized
to Bitcoin mainnet.

The experiment trained on two complete runs and tested on the third, repeating
the process so each run was held out once. Across the three captures, 30,839
event windows and 11,238 TCP flows were prepared. Label pairing associated
15,474 relevant log events with captured flows; 25 events remained unmatched.

## Leave-one-run-out result

| Target | XGBoost mean F1 | Heuristic mean F1 |
| --- | ---: | ---: |
| Block | 0.80 | 0.003 |
| Compact block | 0.74 | 0.006 |
| Handshake event | 0.85 | 0.48 |
| Inventory | 0.73 | 0.32 |
| Request | 0.74 | 0.36 |
| Transaction | 0.75 | 0.38 |
| Transaction sequence | 0.78 | 0.38 |

The separate flow-level BIP324 classifier had a mean F1 of approximately 0.91,
compared with 0.57 for the handshake heuristic.

## BIP324-only check

The newest run was also evaluated after restricting the test rows to known
BIP324 flows. Labels were used only to select this evaluation subset, not as
features.

| Target | XGBoost F1 | Heuristic F1 |
| --- | ---: | ---: |
| Block | 0.86 | 0.05 |
| Compact block | 0.84 | 0.03 |
| Handshake event | 0.93 | 0.93 |
| Inventory | 0.72 | 0.45 |
| Request | 0.76 | 0.48 |
| Transaction | 0.77 | 0.55 |
| Transaction sequence | 0.75 | 0.56 |

This check suggests that the event result is not explained only by separating
Bitcoin flows from background traffic. It does not rule out other lab-specific
shortcuts.

## Interpretation

The preliminary models outperform the current heuristics on these held-out lab
runs, particularly where fixed size thresholds create many false positives.
The result is consistent across the three run-level splits, which is enough to
justify collecting a larger dataset.

It is too early to claim a generally better BIP324 detector. Complete flow
duration has high importance in several models, the lab topology is repeated,
and only one pilot used the final noise profile without generator defects. The
next evaluation will use 20 clean runs and a four-run test partition that
remains untouched during model development.

## Reproduction outline

For each pilot:

1. Run `bip324-traffic-analysis` with `--compare-bitcoin-port-filter`.
2. Run `bip324-traffic-ml prepare` with the matching analysis and lab paths.
3. Train on two `event_samples.csv` files and pass the remaining run id through
   `--test-runs`.
4. Repeat for all three held-out runs and average each target's F1.

The generated `metrics.json` and `test_predictions.csv` files retain the exact
confusion-matrix counts for each split.
