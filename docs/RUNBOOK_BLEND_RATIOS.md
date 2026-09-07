# Epoch-58 + charge-conditioned blend sweep

Run on the SSH experiment machine from the repository directory, in `tmux`.
The code is developed/tested/pushed locally; no full experiment runs locally.
Nothing here trains a model, loads the primary generator, ranks a top-100,
or changes the shipped checkpoints/submission.

## 1. Pull and test

```bash
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run --no-sync pytest -q \
  tests/test_blend_ratio_sweep.py tests/test_blend.py
nvidia-smi
```

No new dependencies are needed. Keep the environment used for the completed
checkpoint sweeps. Commands below assume GPU 1 is available; change its physical
index if needed. Device (`cuda`), ESM model (650M), raw candidate count (100,000),
temperature (1.0), top-p (0.9), repetition penalty (1.3), and library size (50,000)
are inherited from the source sweep, not independently reset to new defaults.

## 2. Validate the seed-42 plan

```bash
uv run --no-sync python scripts/sweep_blend_ratios.py \
  --source-sweep sweep_results/checkpoints-seed44-v2 \
  --primary-case ckpt_epoch58 \
  --seeds 42 --ratios 7:1 3:1 1:1 \
  --secondary-cache sweep_results/epoch58-conditioned-cache-v1 \
  --out sweep_results/epoch58-blends-v1 --list
```

This checks source manifests/hashes, completed control measurements, reference,
runtime versions, tracked evaluator compatibility with the source commit,
and the charge-conditioned checkpoint config/weight file.
It does not load a model, sample, evaluate, or write files.

The secondary defaults to `checkpoint/generator_blend`. It must have
`conditioning: "charge"`; an unconditional checkpoint is rejected.
Do not use the old hybrid's `pool.fasta` as the conditioned component: that
file is already a mixture and does not retain independent component streams.

## 3. Run the three ratios

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/sweep_blend_ratios.py \
  --source-sweep sweep_results/checkpoints-seed44-v2 \
  --primary-case ckpt_epoch58 \
  --seeds 42 --ratios 7:1 3:1 1:1 \
  --secondary-cache sweep_results/epoch58-conditioned-cache-v1 \
  --out sweep_results/epoch58-blends-v1
```

| Result case | Epoch-58 count | Conditioned count | Fraction |
|---|---:|---:|---|
| `p7_s1` | 43,750 | 6,250 | 87.5 / 12.5 |
| `p3_s1` | 37,500 | 12,500 | 75 / 25 |
| `p1_s1` | 25,000 | 25,000 | 50 / 50 |

The primary pool is read from the source sweep and never regenerated. The
conditioned generator draws 100,000 candidates once per requested generation
seed, using reference-distributed charge bins; the clean pool is cached.
Every ratio uses the same two pools for that seed. There is no learned scorer
or fallback sampler, so classifier calibration metadata is not required.

Quotas are exact AFTER deduplication. Shared peptides are assigned only once,
favoring the primary prefix but reserving shared candidates for the secondary
when necessary to satisfy its quota. Sources in `membership.csv` are selection
attributions, not proof that only one generator can produce that peptide.
Insufficient unique candidates fail instead of silently changing the ratio.
The production hybrid's historical interleave handles duplicates differently;
this is a comparison of resulting libraries, not an isolated causal estimate
of changing only the generator checkpoint.

Outputs:

- `results.csv`: fresh measurements for the three new blends; updated per cell.
- `controls.csv`: the source sweep's hybrid and standalone epoch-58 measurements,
  explicitly marked `recorded_source_control`. They are not reevaluated or new
  independent replicates. Their hashes, code provenance, reference and runtime
  are retained/checked. Changed tracked evaluation code or lockfile is rejected;
  do not reuse controls after changing the evaluation protocol.
- `p*/seed42/library.fasta`, `membership.csv`, `blend_stats.json`: generated
  library, source assignment per peptide, exact counts and component-pool hashes.
- Per-cell `evaluation.log`, `metrics.csv`, `metrics.json`, and completion hashes.
- `epoch58-conditioned-cache-v1/seed42/pool.fasta`, `generation.log`, and hashes.

Rerun the identical command to resume completed stages. Do not delete caches,
run concurrent writers against the same output/cache, or change code or runtime
mid-experiment. A changed recipe requires a new output directory; changes to
conditioned generation code, weights, runtime, or sampling also require a new
cache directory. Merely changing ratios or the primary source does not invalidate
the shared conditioned cache when its own recipe is unchanged.

`--generate-only` creates the blends without evaluation. Remove that flag on an
otherwise identical rerun to evaluate. The two source controls must already have
completed evaluations even for this mode.

## 4. Print results and compare

```bash
uv run --no-sync python -c '
import pandas as pd
root = "sweep_results/epoch58-blends-v1"
df = pd.concat([pd.read_csv(f"{root}/controls.csv"), pd.read_csv(f"{root}/results.csv")])
cols = ["case", "seed", "FBD", "MMD", "Precision", "Recall", "Conformity score", "Diversity", "Authenticity", "FKEA"]
print(df[cols].to_string(index=False))
'
```

Compare against BOTH controls. Lower FBD/MMD alone do not establish improved
potency or safety, and gains in one metric do not cancel losses in another by
an established competition weighting. Keep the shipped hybrid unchanged.

## 5. Confirm only a chosen ratio on seeds 43 and 44

After reviewing step 4, use the confirmation source sweep and select the exact
winning weights with `--ratios`. For example, ONLY IF `3:1` is selected:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/sweep_blend_ratios.py \
  --source-sweep sweep_results/checkpoint58-confirm-v1 \
  --primary-case ckpt_epoch58 \
  --seeds 43 44 --ratios 3:1 \
  --secondary-cache sweep_results/epoch58-conditioned-cache-v1 \
  --out sweep_results/epoch58-blend-confirm-v1
```

This reuses the existing epoch-58 pools for these seeds. Their conditioned pools
are generated only if not already cached. No primary resampling or retraining.
Confirmation uses generation seeds, not independent training replicates.

After confirming a candidate, use the [paired top-100 runbook](RUNBOOK_TOP100_COMPARISON.md)
to compare ranking and predicted hemolysis risk without changing library bytes.
