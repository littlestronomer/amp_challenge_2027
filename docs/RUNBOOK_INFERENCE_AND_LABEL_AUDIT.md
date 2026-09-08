# Inference parity and measurement-bound audit

This iteration repairs inference metadata packaging and adds diagnostic gates.
It does **not** retrain models, promote a new selector, reproduce historical
validation, or establish biological safety. Existing label datasets and completed
generalization experiments must remain unchanged.

## What changed

The three audited head SHA256s select metadata bundled under `src/amp_challenge_2027`.
Fresh clones consequently use temperatures 0.98 / 0.92 / 0.95 rather than shared
historical configs. Existing SSH per-artifact configs are preserved; conflicting
metadata for a known head is rejected. Unknown heads retain legacy config lookup.
No checkpoint files are added at paths occupied by SSH's untracked configs.

Production head loading also rejects missing non-backbone parameters. Optional
revision arguments support fixed-revision probe replay. Ordinary generation is
still permissive about unavailable components; this is not yet a strict submission
release workflow. Run the parity gate before any new scoring experiment.

## SSH commands, in order

From the existing repository, with no local code conflicts:

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/audit_inference_parity.py \
  --out sweep_results/inference-metadata-v1

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/audit_inference_parity.py \
  --reconstruction sweep_results/reward-reconstruction-v1 \
  --device cuda \
  --out sweep_results/inference-probes-v1

uv run --no-sync python scripts/audit_measurement_bounds.py \
  --input data/raw/dbaasp/activity.csv --column concentration \
  --out sweep_results/activity-bounds-v1

uv run --no-sync python scripts/audit_measurement_bounds.py \
  --input data/raw/dbaasp/hemolysis_raw.csv --column value \
  --out sweep_results/hemolysis-bounds-v1
```

Stop if a command fails. Each output directory must be new; reports are never
overwritten. No dependency sync or model training is needed. The GPU probe uses
the reconstruction's immutable backbone revision, verifies source completion
hashes, and compares the first eight validation records per task against saved
probabilities at absolute/relative tolerance 1e-5. A mismatch needs investigation,
not a loosened tolerance. This is a smoke test, not full-library parity.

Share the three parity lines and the counts from both bound audits. Detailed
reports are `report.json` in each output directory, including source hashes and
raw-row evidence. Unsupported values include ranges; these are not guessed into
numeric values. MIC label implications are calculated only for explicit µM/uM
strings. Censored hemolysis values are inventoried, not interpreted as safety
labels or actual HC50 measurements.

## Next decision gate

Use the raw-row findings to design versioned, bound-aware label reconstruction
with target/assay filtering, sequence-level conflicts, and molecular modification
metadata. Do not overwrite pinned CSVs or rerun an already-inspected test as a
new holdout. Legacy parsers intentionally remain unchanged for reproducibility;
new training must wait for that label reconstruction and its audit.

After labels are resolved, compare heads under a declared family-disjoint
protocol. The existing test results support predictive signal, but neither the
selected generated peptides' generalization nor safety is guaranteed. A 15B
backbone experiment is deferred until these gates pass.
