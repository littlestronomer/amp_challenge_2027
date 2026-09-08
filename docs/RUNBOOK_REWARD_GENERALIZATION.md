# Family-held-out predictor benchmark

Implemented for the next SSH experiment; no real benchmark results exist yet.
This fixes the evaluation design for **freshly trained reward heads**, not the
already-trained heads. Nothing promotes a model, changes generator defaults,
rewrites a label CSV, or re-scores the six 50k libraries.

## What this experiment measures

The reconstructed AUROCs matched the frozen-run summaries, but activity had
193/403 validation rows with an exact training sequence, and 351/403 with a
nearest-training Levenshtein ratio greater than 0.8. Hemolysis and panel also had
substantial near-overlap (665/951 and 1667/2213 rows). Matching those AUROCs
established numerical consistency, not out-of-family performance.

The new protocol measures a frozen 35M representation with two fresh head
architectures: a linear baseline and the production-shaped MLP (the same dense
layer applied twice). Both use the same inputs, train/validation/calibration/test
membership, and three predeclared seeds. No 15B model is needed for this test.

| Partition | Approximate fraction | Allowed use |
|---|---:|---|
| Train | 60% | Fit head weights and training-only class weights/prevalence baselines |
| Validation | 15% | Select each seed's epoch by macro AUROC, early stopping |
| Calibration | 10% | Fit one positive temperature to the fixed three-seed mean-logit ensemble |
| Test | 15% | Final measurements only, via a separate explicit unlock command |

Fractions refer to the joint sequence graph, not each task. Whole families are
assigned using a label-blind, fixed-seed largest-first procedure; realized task
sizes and class balance can differ markedly. No split-seed search is performed.

### Curation and family boundary

- Uppercase canonical 8–50-residue peptides only; never silently truncate inputs.
- Collapse repeat evidence for a sequence/output to one observation. Conflicting
  0/1 labels for an output are **masked**, not resolved by CSV order, a majority
  vote, or choosing the active label. Excluded rows and reasons are recorded.
- For binary activity, this defines a **consensus-only measured-context task**.
  Opposite labels across organisms can be genuine biology, not erroneous data.
  Such sequences are excluded from this binary benchmark, not declared inactive.
  The organism-specific panel remains important; its conflicts are masked per
  sequence/genus, leaving other observed genera usable. This change of population
  means new scores must not be directly compared to the old random-split AUROCs.
- Jointly group sequences from all three CSVs, including valid excluded sequences
  as graph bridges. Identical peptides cannot enter train for one task and test
  for another.
- Build connected components of the **all-pairs Levenshtein-ratio >= 0.8 graph**.
  This includes transitive families that greedy leader clustering can miss.
  Levenshtein ratio is normalized indel similarity, not aligned biological identity.
- Independently check every possible cross-partition threshold violation, using
  only an exact length upper bound to skip impossible edges. Every pair across
  any two partitions must have ratio **strictly below 0.8**, including exactly-0.8
  pairs in the prohibited set. No approximate neighbor search.
- If a giant component or missing classes makes the split unusable, preparation
  reports the problem and training stops. Do not lower the boundary, break a
  family, or search split seeds to obtain a favorable score.

## Commands on the SSH machine, in order

Use one writer per output directory, ideally inside your existing tmux session.
No dependency upgrade is required. The commands use the existing uv environment
and the hash-pinned label CSVs already audited on the SSH machine.

### 1. Pull the code

```bash
cd /home/istke/Documents/Project/amp_challenge_2027
git pull --ff-only origin main
```

### 2. Prepare and verify the split (CPU only)

```bash
uv run --no-sync python -u scripts/prepare_reward_generalization.py \
  --reconstruction sweep_results/reward-reconstruction-v1 \
  --out sweep_results/reward-generalization-data-v1
```

This pins the resolved 35M backbone commit from the completed reconstruction;
it does not reuse its head weights or its validation split. All three recorded
backbone revisions must agree. The input CSV hashes are checked before curation.
Pairwise grouping and verification are quadratic in the number of unique
sequences, with bounded graph memory; expect CPU work and progress messages.

Inspect the support table before spending GPU time:

```bash
uv run --no-sync python -c '
import json
from pathlib import Path
root = Path("sweep_results/reward-generalization-data-v1")
print(json.dumps(json.loads((root / "preflight.json").read_text()), indent=2))
print((root / "support.csv").read_text())
print((root / "curation.json").read_text())
'
```

Continue only if `eligible_for_training` is true and `violations` is zero.
The preflight requires both classes for every training output, at least one
defined validation output per task, and nonempty task partitions. It does not
guarantee adequate test power. Look at family counts and per-output support too.
Test support is descriptive; it must not be used to optimize assignments.

### 3. Train the predeclared controls and seal them

Optional read-only preflight:

```bash
uv run --no-sync python scripts/benchmark_reward_generalization.py train \
  --prepared sweep_results/reward-generalization-data-v1 \
  --out sweep_results/reward-generalization-fit-v1 \
  --list
```

Actual experiment, on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_reward_generalization.py train \
  --prepared sweep_results/reward-generalization-data-v1 \
  --out sweep_results/reward-generalization-fit-v1
```

This computes frozen train/validation/calibration embeddings once per unique
sequence and shares them across tasks. It trains 18 small heads: 3 tasks ×
2 architectures × 3 seeds. The backbone stays in frozen evaluation mode.
The MLP head has dropout; neither architecture fits feature normalization.
Both use training-only positive class weights, AdamW, and the fixed recipe in
`experiments/reward_generalization_v1.json`. No test embeddings are produced.

All seeds remain in the ensemble; there is no best-seed promotion. Temperature
is fitted on calibration data only, with equal weighting of outputs having both
classes. Unsupported outputs are disclosed; if none are supported, T=1 is kept
and calibration is marked not fitted. Boundary grid optima are flagged. Scalar
temperature cannot correct every calibration problem, particularly an intercept
shift from class-weighted training; the final report includes raw and calibrated
metrics rather than assuming calibration improved them.

The command ends with `ALL fits sealed` and writes `sealed.json`, head weights,
validation histories/logits, calibration logits, and exact head/config hashes.
Do not change any hyperparameters after looking at final-test performance.

### 4. Unlock the final test, once the fits are sealed

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_reward_generalization.py test \
  --prepared sweep_results/reward-generalization-data-v1 \
  --training sweep_results/reward-generalization-fit-v1 \
  --out sweep_results/reward-generalization-test-v1 \
  --unlock-test
```

The command refuses an incomplete or altered training seal. It performs no
optimization, temperature fitting, checkpoint selection, or deployment changes.
It evaluates both architectures and all seeds, with the fixed ensemble as the
primary readout. Test labels are available in the prepared artifact for
reproducibility; the unlock is a workflow guard, not cryptographic blinding.

### 5. Share the results

```bash
uv run --no-sync python -c '
import json
from pathlib import Path
root = Path("sweep_results/reward-generalization-test-v1")
print((root / "results.csv").read_text())
print((root / "paired_architecture_deltas.json").read_text())
for task in ["activity", "panel", "hemolysis"]:
    print("\nPER-OUTPUT MLP:", task)
    print((root / task / "mlp/per_output_metrics.csv").read_text())
'
```

`results.csv` includes macro AUROC/AP, Brier score, log loss, ECE, prevalence
baselines, family counts, and family-resampled AUROC intervals. Per-output CSVs
contain observed positive/negative and family support. Masked panel entries
never enter metrics. Undefined AUROC/AP values are null, never chance values.
`metrics.json` also contains each seed's uncalibrated measurements and raw vs
calibrated ensemble results. Prediction tables preserve sequence/family/output
alignment and masks, and can support future read-only analyses.

The 1,000 bootstrap draws resample **families**, using identical draws for the
MLP-minus-linear comparison. The set of test-evaluable outputs stays fixed;
single-class resamples are omitted and counted, not allowed to alter the macro
denominator. Intervals are withheld when fewer than 80% or 100 draws are valid.
Rare-genus performance can remain inconclusive. Reliability-bin Wilson intervals
are descriptive iid-row intervals, not family-level intervals.

## Integrity and resuming

Use the same command to resume on the same code, environment, inputs and
settings. Completed feature caches and heads are hash-checked and reused.
An interrupted, incomplete head is retrained deterministically from its seed;
this small-head trainer does not resume optimizer state mid-epoch. Completed
test cells are reused, and the final test report can be reopened without fitting.

Changed code, protocol, batch size, inputs, or runtime require a new output
directory; do not edit manifests to bypass checks. The final test additionally
requires the same code/runtime as sealed training, to prevent an altered forward
implementation from silently reinterpreting the saved heads. Keep that checkout
and environment through both stages. Never reuse an output
directory concurrently. Keep all outputs outside `checkpoint/`; deployment
directories are explicitly protected. New models are experimental controls,
not automatically loadable/promoted production checkpoints.

## Interpreting the next result

1. Check label exclusions, realized partition sizes, and per-output family/class
   support before interpreting aggregate scores.
2. Compare the fresh MLP with the fresh linear control on identical test samples,
   and compare Brier/log loss with training-prevalence constants. A lower score
   than the old random validation is not evidence of a regression: the population,
   split and training protocol changed.
3. Examine uncertainty and raw-vs-calibrated metrics. A small positive AUROC delta
   is not convincing if its paired family interval spans zero. Do not equate a
   sigmoid output with an experimentally calibrated probability of safety.
4. This is an **internal family-held-out benchmark conditional on one split and
   one pretrained representation**. It does not certify historical independence,
   rule out ESM pretraining exposure, account for all assay/source confounding, or
   validate activity of the generated top-100. The old heads have seen overlapping
   data and cannot be fairly added as supposedly held-out test controls.
5. Once the test is inspected, later changes require a new genuinely untouched
   evaluation source or a preregistered nested evaluation—not repeated tuning on
   this test. An external/source/time-held-out dataset and prospective wet-lab
   measurements remain necessary for strong generalizability claims about the
   complete peptide-generation method. Keep the current hybrid/heads as deployed
   controls until the predictor evidence and a separate selection study justify
   a change.
