# Explaining feature influence with SHAP

## What is explained

`explain` loads an already fitted **multiclass XGBoost** model and attributes each
raw class score to the 34 passive inputs. It never fits or updates the model.
The exact native XGBoost TreeSHAP implementation calculates contributions;
`shap.Explanation` and SHAP plotting functions render them. No new PCAP extraction
or training is required for prepared data.

For sample `i` and class `c`:

```text
raw_score[i, c] = base_value[i, c] + sum_j SHAP[i, j, c]
```

A positive value increases that class's raw score; a negative value decreases it.
These are **raw margins**, not probability percentage points. Multiclass
probabilities follow from softmax across the complete vector of class scores.
Every batch checks the sum against the booster's actual margins.

The reference is `tree_path_dependent`: training path cover stored in the
booster defines the background, rather than a newly chosen test background.
This choice explains the fitted model with its historical training distribution.

## Run using the existing model

From `fellowship`, use the shared `.venv`:

```bash
source .venv/bin/activate
python -m pip install -e './bip324-traffic-ml[explain]'

python -m bip324_traffic_ml explain bip324-traffic-ml/data \
  --model-dir bip324-traffic-ml/models/coverage-all-activity \
  --max-samples 1024 --error-samples 5 --threads 4 \
  --output-dir bip324-traffic-ml/artifacts/explanations/coverage-all-xgboost-01
```

A new trained XGBoost directory can replace `--model-dir`. The default cohort is
that model's recorded test runs. `--runs run-a,run-b` selects a different supplied
holdout, including new captures. Recorded training runs are rejected. Keep the
same frozen test cohort for comparisons; use separate validation data when
choosing features based on these explanations.

Computation runs on CPU and batches the selected examples; it does not need the
RTX. The command predicts all supplied unambiguous held-out windows, then explains
a bounded sample. `--max-samples` limits the population sample. `--error-samples`
adds up to that many diagnostic examples per error/activity/noise group. The seed
is reproducible; increase the sample for a more stable estimate. `--no-plots`
exports numerical outputs without SHAP/Matplotlib installed. Existing outputs
cannot be overwritten: choose a new directory for another experiment.

## Avoiding a misleading global ranking

`other` dominates this dataset. Pure uniform sampling can miss rare blocks,
transactions and handshakes. The population sample is therefore uniform **within
each true class**, with capped balanced quotas and no replacement. True labels
select the diagnostic sample only; they never become feature columns or booster
inputs.

CSV rankings correct the class oversampling with each row's weight:

```text
weight = count_of_class_in_full_holdout / (full_holdout_size * sampled_class_count)
```

Weights sum to one. Extra examples chosen because they are errors have zero
population weight unless already in the population sample. Their inclusion
cannot artificially raise their influence in the global ranking. These are
approximate population summaries from a bounded sample, not confidence intervals.
The SHAP beeswarm itself shows the **unweighted balanced sample**, as its title
states; read the weighted global CSV/plot for overall population importance.

## Outputs

| File | Meaning |
| --- | --- |
| `report.md` | Reading guide, top features per activity and full-cohort error counts |
| `global_importance.csv` / `.png` | Population-weighted mean absolute SHAP, equally averaged over output scores |
| `importance_by_output_class.csv` | Which features influence each class score across the full mixture of traffic |
| `importance_within_true_class.csv` | Which features influence an activity's own score among samples actually labeled with that activity |
| `group_importance.csv` / `.png` | Signed contributions summed within size/timing/opening/context groups, then summarized by magnitude |
| `beeswarm_CLASS.png` | Direction and spread of effects; red/blue indicate high/low feature values |
| `error_contrast_importance.csv` | Features supporting a mistaken class over the actual class, grouped by false-positive noise family, missed activity or activity confusion |
| `waterfall_*.png` | Local predicted-minus-actual score explanation for one diagnostic example per error group |
| `explained_samples.csv` | Run/flow/window IDs, actual/predicted class, confidence, sample membership and population weights |
| `explained_features.csv` | The exact allowed measurements used in those explanations |
| `shap_values.npz` | Per-window/feature/class values, bias and raw scores; loads with `allow_pickle=False` |
| `explanation_summary.json` | Sample settings, cohort sizes, additivity error, source identities and library versions |

`bitcoin_detection_full_population` and `error_population_counts` use all supplied
unambiguous held-out windows, not the balanced explanation sample. A missing
trained class can be reported as an error but has no output score of its own to
explain. Such contrasts are omitted and listed in `unseen_true_classes`.

## Interpreting the result in this experiment

- Strong opening-exchange contributions can explain handshake detection without
  giving the model IPs or ports. This alone does not establish mainnet detection
  or prove survival under a new randomized-port distribution.
- Strong future-window or completed-flow contributions show reliance on offline
  information. They cannot support claims about a real-time ISP classifier.
- In a noise false positive, a positive predicted-minus-`other` contribution shows
  what pushed the model toward Bitcoin. Negative contributions opposed that error.
- Bytes, packet counts, rates and context are correlated. SHAP can divide credit
  between substitutes. Grouped sums aid reading but do not remove correlation.
- Use these results to propose grouped ablations on validation data. A feature
  removal experiment and fresh evaluation are needed to show a performance effect.

The default `--method shap` supports the XGBoost architecture. Tree-model
attributions must not be presented as explanations of the CNN.

## Grouped permutation for either model

Use `--method permutation` to measure score dependence on related input groups:

```bash
python -m bip324_traffic_ml explain bip324-traffic-ml/data \
  --method permutation \
  --model-dir bip324-traffic-ml/models/coverage-updated-20261027-cnn \
  --max-samples 2048 --repeats 3 --batch-size 1024 --device cuda \
  --output-dir bip324-traffic-ml/artifacts/explanations/cnn-permutation-01
```

This uses a class-balanced sample reweighted to the supplied holdout prevalence,
then shuffles each feature group between samples with the model frozen.
Donors are uniform within this balanced sample; prevalence weights correct the
score, rather than changing the donor distribution. For the
CNN, every offset in the input context is shuffled jointly for that group.
Missing-context patterns are carried with donor group values. This can produce
inputs outside the observed distribution; unchanged correlated groups can also
substitute for the perturbed group. Interpret the result as dependence under a
particular perturbation, not as a causal effect, feature-removal result, or SHAP.

`permutation_importance.csv` gives the mean and standard deviation of macro F1
and accuracy drops across the permutations. The baseline is the weighted sampled
score, not the full-holdout metric. Negative drops are retained. The standard
deviation describes permutation variability, not a confidence interval over
independent captures. New version-2 CNN checkpoints are supported; earlier weights
remain available in their original directories.

References: [SHAP tabular examples](https://shap.readthedocs.io/en/latest/tabular_examples.html),
[TreeExplainer semantics](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html),
[XGBoost contribution API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict).
