# Methodology

## Research question

The experiment asks whether passive measurements can classify BIP324 flows and
traffic events more accurately than the current size- and timing-based
heuristics. It does not decrypt BIP324 or recover exact message contents.

The initial model is XGBoost. The available input is tabular and the dataset is
still small, so a tree-based baseline is easier to inspect and evaluate than an
LLM or a larger sequence model.

## Data path

1. `bip324-traffic-lab` creates a controlled Warnet run containing Bitcoin v1,
   Bitcoin v2, HTTP, HTTPS, Tor, obfs4-shaped, BitTorrent, and UDP traffic.
2. A bridge-interface sniffer records the mixed traffic without participating
   in the Bitcoin connections.
3. `bip324-traffic-analysis` extracts passive flow and 250 ms timeline
   features from the PCAP.
4. This project associates Bitcoin Core log events and connection lifecycle
   records with the corresponding captured flow.
5. Complete lab runs, rather than individual windows, are assigned to training
   or test partitions.

Logs and lab metadata are ground truth for the synthetic experiment. They are
not detector inputs.

## Features

Eligible features describe information visible in the passive capture:

- packet and byte counts by direction;
- burst duration, density, and gaps;
- bytes and packets in neighboring 250 ms windows;
- elapsed time and estimated phase within a flow;
- initial payload-flight sizes and payload entropy;
- complete flow duration and packet counts.

The last group makes the current classifier retrospective: it can analyze a
completed capture, but it is not yet an online classifier. Lead-window features
also use packets that follow the classified window. A future online evaluation
must remove complete-flow and future-window measurements.

The following columns are excluded from model features:

- IP addresses and ports;
- flow, run, and dataset identifiers;
- Bitcoin Core labels and Warnet metadata;
- heuristic decisions and BIP324 detector flags;
- cleartext hints extracted during validation.

## Labels and pairing

Flow labels identify v1 and v2 connections. Transient sessions are paired by
their observed TCP four-tuple and connection interval. Persistent lab flows are
identified from the controlled Bitcoin node endpoints.

Event labels cover handshake, block, compact block, inventory, request,
transaction, and transaction relay sequence. Log events are matched by peer,
flow interval, and timestamp. A transient session's peer id is used when an
exact session pairing is available.

Unmatched and ambiguous events are written to pairing reports. They are not
silently assigned to the nearest flow.

## Evaluation

The primary comparison uses the same test rows for XGBoost and the existing
heuristic. Reported metrics include precision, recall, F1, average precision,
ROC AUC, and the four confusion-matrix counts.

The preliminary experiment used leave-one-run-out evaluation over three lab
runs. A second evaluation restricted the newest held-out run to known BIP324
flows. This checks whether an event model merely learned to separate Bitcoin
from background traffic.

The planned larger experiment uses 20 independent seeds. Sixteen runs will be
available for training and validation, while four complete runs remain held
out until the final evaluation. Threshold selection and model changes must not
use those four runs.

## Limitations

- Warnet traffic is synthetic and is not a representative sample of the
  Internet or Bitcoin mainnet.
- The three preliminary runs used different noise-generator revisions. Only
  the third used the final calibrated profile.
- Nodes, topology, Bitcoin Core build, and traffic generators are shared across
  runs, which may expose stable lab-specific patterns.
- Positive block and handshake samples are still limited.
- High scores can reflect repeatable properties of this lab rather than
  general BIP324 behavior.
- The model currently uses retrospective features and should not be described
  as a real-time detector.

## Research background

The feature design follows two observations from related work:

- Rasteh et al., *Encrypted Internet Traffic Classification Using a Supervised
  Spiking Neural Network* ([arXiv:2101.09818](https://arxiv.org/abs/2101.09818)),
  motivates combining packet sizes with short temporal neighborhoods for
  encrypted traffic classification.
- Ndolo and Tschorsch, *Security Analysis of Bitcoin's V2 Transport Protocol*
  ([arXiv:2605.19715](https://arxiv.org/abs/2605.19715)), studies what remains
  observable from encrypted Bitcoin v2 traffic and motivates evaluating
  length- and timing-derived message evidence.

These papers motivate the experiment; they do not validate the results in this
repository.
