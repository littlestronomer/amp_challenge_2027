# Provisional reconstruction of the deployed predictors' validation results

## What is established, and what is assumed

The SSH artifact audit found exact tensor matches between each deployed head
and a member under its frozen-training run:

| Predictor | Frozen run | Member | Recorded seed | Stored T | Recorded AUROC |
|---|---|---:|---:|---:|---:|
| Activity | `checkpoint/reward_binary_frozen` | 2 | 44 | 0.98 | 0.7860 |
| Panel | `checkpoint/reward_panel_frozen` | 1 | 43 | 0.92 | 0.7995 |
| Hemolysis | `checkpoint/reward_hemo_frozen` | 0 | 42 | 0.95 | 0.8460 |

Matching both the frozen run's promoted file and its winning member is expected.
The prior panel helper's default `--member 2` is **not** the deployed panel's
winning member. A measurement made with that default is not a replay of the
matched member's validation split.

The user supplied current hashes for three candidate label CSVs. Searches in
the frozen-run directories and `runs` found no saved predictor split manifests
or training logs establishing original dataset identity. In particular:

- The current hashes pin this experiment's inputs; they do not prove these
  bytes were used in the September 5 training runs.
- The matching members' seeds are recorded, but the historical split mode,
  dataset membership/order, trainer version and original backbone revision
  are not established by the member summaries.
- This audit assumes the current trainer's **random 80/20** split: binary
  stratified by class, panel shuffled after the current loader's aggregation.
  It preserves loaded row order, duplicate binary rows and current panel
  conflict handling; it does not clean/repartition data to improve results.
- The ESM revision comes from the completed top-100 comparison, not a verified
  original training-time model manifest. It is explicitly pinned for this run.

These assumptions are recorded in
[`experiments/reward_reconstruction_v1.json`](../experiments/reward_reconstruction_v1.json).
No alternative seeds/splits are searched to find an AUROC match. This is an
explicit **reconstruction check**, not an exact historical validation replay,
new independent holdout, or biological safety assessment.

## 1. Pull and run the CPU preflight on SSH

Keep the existing environment and the completed audit/source directories.
Do not rerun or modify their old manifests after updating code.

```bash
cd /home/istke/Documents/Project/amp_challenge_2027
git pull --ff-only origin main

uv run --no-sync python scripts/eval_reward_reconstruction.py \
  --out sweep_results/reward-reconstruction-v1 \
  --list
```

The preflight reads the current pinned CSVs and reconstructs memberships to
print train/validation counts. It verifies the artifact audit, deployed head
hashes/configs, expected matched member and recorded seed/temperature/AUROC,
and the comparison's backbone provenance. It neither writes nor loads ESM.

Defaults import `sweep_results/reward-artifact-audit-v2` and
`sweep_results/epoch58-top100-v1`. A missing/changed input is a stop condition:
investigate, do not edit old integrity markers or substitute a different head.

## 2. Run the three bounded 35M inference jobs

If the stated reconstruction assumptions are acceptable, run:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/eval_reward_reconstruction.py \
  --out sweep_results/reward-reconstruction-v1 \
  --accept-reconstruction
```

Choose a free GPU and use `tmux` if needed. The order is hemolysis, activity,
then panel. Only reconstructed validation examples are inferred; no 50k
libraries, generator training or 650M/15B evaluations are launched. Exact
validation-to-training sequence-similarity checks run on CPU after inference.

`--accept-reconstruction` acknowledges the unverified assumptions. It never
turns the output into verified historical validation. Temperatures remain
0.95/0.98/0.92; no calibration is refitted and no deployed artifact is written.

For a hemolysis-only first pass, add `--tasks hemolysis` to both commands and
use a separate output such as `sweep_results/reward-reconstruction-hemo-v1`.
The default batch size is 64, with FP32 inference and the production head
architecture/tokenization/pooling. `--batch-size` and `--device` are explicit
protocol settings; changing them requires a fresh output directory.

## 3. Print the comparison

```bash
uv run --no-sync python -c '
import pandas as pd
root = "sweep_results/reward-reconstruction-v1"
cols = ["task", "member", "seed", "temperature", "recorded_val_auroc",
        "macro_auroc", "auroc_delta_from_record", "discrepancy_requires_investigation",
        "macro_average_precision", "macro_brier", "macro_log_loss",
        "macro_training_prevalence_baseline_brier",
        "macro_training_prevalence_baseline_log_loss", "macro_ece_10_equal_width",
        "validation_rows", "validation_rows_with_exact_training_sequence",
        "validation_rows_with_nearest_ratio_gt_0_8"]
print(pd.read_csv(f"{root}/results.csv")[cols].to_string(index=False))
for task in ["hemolysis", "activity", "panel"]:
    print(f"\n{task.upper()}: UNCALIBRATED VS STORED TEMPERATURE")
    print(pd.read_csv(f"{root}/{task}/per_output_metrics.csv").to_string(index=False))
'
```

Send those tables for review. If only one task was requested, adjust the final
loop accordingly. The result table updates after completed tasks; identical
reruns verify/skip completed tasks and reuse marked raw-logit caches when
post-inference reporting was interrupted. Tampered outputs fail rather than
being silently overwritten. No concurrent writers to an output directory.

## Output and interpretation

Each task directory contains:

- `split_assignments.csv`: explicit membership/order identifiers into the
  **currently loaded records**, targets and masks. This is a newly reconstructed
  split record, not a recovered historical manifest.
- `predictions.npz`, `backbone.json`, `inference.json`: raw logits with a verified
  completion marker and explicit model revision/runtime settings.
- `validation_predictions.csv`: sequence, output label/mask, raw logits,
  probabilities before/after the stored temperature, properties, and nearest
  reconstructed training-sequence similarity. Masked panel entries remain
  visible but never enter metrics. Properties for nonstandard alphabets are null.
- `split_audit.json`: current-input row counts, duplicate/label-conflict notes,
  exact cross-split overlap and full-reference Levenshtein similarity counts.
- `per_output_metrics.csv`, `reliability_bins.csv`, `metrics.json`: measured
  support, AUROC/AP, calibration metrics and explicit provenance limitations.

AUROC and AP rank raw logits, so a positive fixed temperature does not change
their ordering. Historical training computed AUROC from sigmoid outputs;
saturation/rounding, batch size and runtime can cause numerical differences.
Binary AUROC is the single-output `macro_auroc`; panel AUROC averages only
outputs with both classes, with the number of defined outputs recorded.
Empty/single-class strata have null rank metrics, not fallback AUROC=0.5.

Brier and log loss compare the probabilities against observed labels, and
against a constant predictor using **reconstructed training prevalence**.
Baseline log loss clips prevalence to `[1e-7, 1-1e-7]`; baseline Brier uses the
unclipped prevalence. Panel calibration summaries are equal-output macro means,
not pooled label counts. Per-output supports are necessary context.

Reliability uses 10 equal-width bins and ECE with row-count weighting. Bins
with fewer than 30 observations are flagged. Wilson intervals are descriptive
**iid-row** intervals, not family-level uncertainty estimates: duplicates,
near-neighbours and multiple panel observations can violate independence.
Neither ECE nor a stored temperature certifies calibration on generated peptides.

An absolute AUROC discrepancy >0.001 is an investigation trigger, not a
scientific performance cutoff. Matching the rounded recorded AUROC does not
validate historical data provenance. Do not tune seeds or temperature to close
a gap. Inspect data, loader/split assumptions, masks and runtime first.

The legacy trainer selected checkpoints and fitted temperature on validation
data, so these are not independent generalization estimates even if exact
historical provenance were recovered. High cross-split similarity is additional
reason for caution. No risk-weight sweep, model promotion or new training is
automatically enabled by a favourable reconstruction result.
