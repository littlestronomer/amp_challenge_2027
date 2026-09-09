# Fixed-population label ablation (development only)

This experiment does not promote heads and has no test-evaluation command.
Defaults: panel and hemolysis, frozen 35M backbone and immutable revision inherited
from the original family benchmark, production-shaped MLP, seeds 42/43/44, and
the existing epoch/patience/optimizer recipe. All three heads are retained.

## Run on SSH

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/run_label_ablation.py prepare \
  --prepared sweep_results/reward-generalization-data-v1 \
  --candidates sweep_results/molar-candidates-v1 \
  --tasks panel hemolysis \
  --out sweep_results/label-ablation-data-v1
```

Inspect `eligible` and `failures`. Preparation checks both classes in EVERY
output/arm/train/validation/calibration partition. This is a minimum computability
check, not a guarantee of statistical power. Per-output counts and family support
are in `support.csv`. Label transitions on common keys are in `transitions.csv`.
If support fails, stop and share the failures; do not change splits or silently
drop outputs. A missing candidate task file also blocks preparation.

If eligible:

```bash
uv run --no-sync python scripts/run_label_ablation.py train \
  --prepared sweep_results/label-ablation-data-v1 \
  --out sweep_results/label-ablation-fit-v1 --list

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/run_label_ablation.py train \
  --prepared sweep_results/label-ablation-data-v1 \
  --out sweep_results/label-ablation-fit-v1 --device cuda
```

This fits 24 heads. Frozen embeddings are shared across all arms and cached by
development partition. There are no test records in the generated training
dataset and no test embeddings/predictions. Completed heads resume with matching
manifests and hashes; interrupted cells rerun deterministically. Code or parameter
changes require a new output directory. Original snapshots remain unchanged.

## What is held fixed

| Arm | Training population | Training labels |
|---|---|---|
| original | Original curated benchmark training records | Original |
| subset_original | Common observed sequence/output keys | Original |
| subset_candidate | Exactly the same common keys | Candidate |
| full_candidate | All mapped molar-only candidate training records | Candidate |

Validation and calibration in **all four arms** use the same common observed
keys and candidate labels. The original arm is freshly trained; old published
scores with different evaluation populations are not used as the baseline.
Per-output masks are intersected, not merely peptide IDs. No sequence changes
partition. Unmapped candidates block preparation. Test members are excluded.

Class weights are recomputed by the existing fitter for each arm. Thus changes
include the associated class-weight response to filtering/relabeling. Epochs are
selected on the fixed validation population; a single temperature is fitted on
the separate common calibration set after averaging all three seed logits.

## Readout

```bash
uv run --no-sync python - <<'PY'
import pandas as pd
root = 'sweep_results/label-ablation-fit-v1'
for name in ['validation_summary.csv', 'validation_deltas.csv']:
    print('\n' + name)
    print(pd.read_csv(f'{root}/{name}').to_string(index=False))
PY
```

`validation_seeds.csv` reports uncalibrated per-seed metrics. Ensemble metrics,
calibration parameters and per-genus metrics are also saved. Deltas compare
subset_original−original (filtering), subset_candidate−subset_original
(relabeling), and full_candidate−subset_candidate (candidate expansion).
If labels are unchanged, the relabeling arms should be identical under deterministic
training. Do not characterize a filtering improvement as corrected-label benefit.

These are development results on candidate-defined labels, NOT independent
biological ground truth. Validation is also used for early stopping. No confidence
interval or fresh-test claim is produced. The old test remains previously inspected.
Activity is intentionally opt-in (`--tasks activity` with a separate output): its
small common population and different target definition require separate review.
