# Methodology

## Task and observation boundary

One multiclass model predicts a Bitcoin v2 activity or `other` for each observed
250 ms payload window. It does not decrypt messages. Both XGBoost and the temporal
CNN use the same feature contract, labeling rules and held-out captures.

The lab generates traffic and records PCAPs, Core logs and connection metadata.
This project reads the full PCAP without consulting ports, IP maps or heuristic
Bitcoin detectors to select traffic. Extraction is independent of labeling.
Every supported TCP/UDP flow is recorded; payload-free flows have no activity
windows and appear in the extraction summary rather than receiving invented
class predictions. Non-IP/non-TCP/UDP packets are outside this model's scope.

## Transformations and feature contract

1. Parse PCAP/PCAPNG records sequentially, including gzip. Supported link layers
   are Ethernet (including VLAN), Linux SLL/SLL2 and raw IPv4/IPv6.
2. Group packets by protocol and bidirectional endpoint tuple. Endpoint A is
   lexicographically smaller. A new TCP SYN sequence distinguishes tuple reuse;
   retransmitted SYNs stay in the same session. UDP tuples span the capture.
3. Aggregate nonempty directional payload buckets relative to the flow start.
4. Join the two directions, add neighboring-window measurements and static flow
   statistics, then select the fixed allowlist in `features.py`.

The 34 inputs are:

- **Current window (7):** directional payload bytes and packet counts, total
  bytes/packets and absolute byte-direction imbalance.
- **Burst measurements (6):** duration, byte rate and preceding burst gap in each
  direction. Byte rate uses `max(burst_duration, window_scale, 0.001)`.
- **Neighboring observations (8):** byte totals 1, 2 and 4 observed rows before
  and after; byte/packet totals over the previous four observations. These shifts
  follow observed buckets, not empty intervals of wall-clock time.
- **Connection age (4):** elapsed seconds plus fixed phase flags for the first
  second, the following four seconds and later traffic. `handshake` here is a
  time-range name, not a ground-truth label.
- **Opening exchange (5):** first payload and initial sending-turn size in each
  direction; entropy of up to 64 captured bytes from each of the first 12 effective
  payload packets. Raw bytes never enter training.
- **Complete flow (3):** duration, total packets and effective payload packets.
- **Transport (1):** TCP versus UDP.

IPs, ports, absolute timestamps, flow/run IDs, Core logs, node roles, noise
families, heuristic flags and raw payload are excluded. IPs/ports identify flows
and pair ground truth but never become numeric model inputs. Unknown numeric
columns cannot silently enter training. Missing/invalid features cause an error.

The feature schema preserves historical `passive-windows-v1` semantics so prepared
data remain usable. In particular, TCP effective bytes use a sequence high-water
mark, **not full reassembly**: reordered segments and 32-bit sequence wrap may be
miscounted. IP fragments are counted and skipped; encrypted/unsupported IPv6
extension chains are not decoded. Snaplen truncation is reported: wire payload
size can be inferred from headers, but entropy only uses captured bytes. Capture
framing truncation and unsupported link types fail explicitly. Timestamps must
be nondecreasing within each flow.

Whole-flow totals and future context make these **offline** models. They cannot
be advertised as real-time detectors without rebuilding a causal feature set.
The CNN applies signed `log1p`, training-only mean/std normalization and clipping,
then uses the center window and up to four adjacent buckets per side. Context
never crosses captures, flows or missing buckets. Ambiguous windows can supply
passive context but never supervised targets.

## Ground truth

Core logs and connection records label observations only after extraction.
Sessions match exact TCP endpoints and connection time; log events use node,
peer and time. Matching failures and ambiguous flow associations stay in audit
reports. Relevant events map to the nearest observed bucket within the configured
tolerance (default one second). This is approximate flow/time supervision, not
packet-level decrypted truth; block transfers can span more than one window.

New preparation requires observed connection transport or matched v2-handshake
log evidence to positively label a TCP connection between Bitcoin nodes. Node
IP ownership alone is insufficient; unverified node connections are ambiguous.
Outside those node connections, the controlled lab topology supplies background
labels. This assumption is specific to lab labeling and is never used for
inference on an arbitrary PCAP.

The eight classes are `other`, `bitcoin_v2_other`, `handshake`, `block`,
`compact_block`, `inventory`, `request`, and `transaction`. V1 windows map to
`other`. Relay sequences contribute to `transaction`. Compact-block evidence
supersedes generic block evidence. Overlapping activities are ambiguous except
that a full-block label dominates its overlapping request/relay labels, unless a
handshake is also present. Original binary label flags remain available for audit.
`bitcoin_v2_other` means no specific event matched that window, not absence of a
Bitcoin message. Historical prepared labels are not rewritten automatically.

## Training and evaluation

Hold out whole captures before fitting. Duplicate files, run IDs and duplicate
capture paths in manifests are rejected. The caller names the holdout explicitly.
Normalization and clipped inverse-frequency class weights use training rows only.
Model parameters, seeds, data checksums, library versions and split IDs are saved.

Both models report the same multiclass metrics, Bitcoin detection errors and
confusion by background family. A classifier that predicts Bitcoin everywhere
will have a noise false-positive rate of 100%; accuracy alone is insufficient.
Undefined rates (no positive or negative examples) are `null`, not misleading zeros.

Connection detection takes a majority vote after mapping all activity predictions
except `other` to Bitcoin v2. A tie votes v2, preserving the original rule. Scored
flow metrics use only windows with unambiguous targets; unlabeled inference votes
over all observed payload windows. Window errors and connection errors therefore
have different denominators and interpretations.

The test set does not select epochs, thresholds or hyperparameters. Use a separate
validation cohort for tuning and a fresh untouched test cohort for final claims.
The historical holdout has already influenced comparisons and remains exploratory.

## Limits

Warnet topology, kernel, Core build and noise generators can produce lab-specific
signals. Preserving the old feature schema does not make new captures statistically
identical: the new extractor includes all supported flows, including infrastructure
traffic that the old analysis may have scoped out using the lab IP map. Evaluate
this distribution change explicitly. Neither encryption alone nor high lab scores
establish a reliable fingerprint across real Internet traffic. Validation still
needs randomized ports, different generators/topologies and untouched captures.
