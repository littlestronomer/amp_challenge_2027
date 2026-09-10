# On-policy specialist distillation

This is an isolated experiment, not a production change. A student initialized
from the original primary generator learns from its matching frozen GRPO teacher.
The teacher need not be larger: this tests transfer of its learned specialty.
It does not assume that the teacher is superior on whole-library metrics.

## Three fixed variants

| Variant | Objective |
| --- | --- |
| `specialist` | Forward token KL from the frozen GRPO teacher on fresh student-generated prefixes. |
| `anchored` | Specialist loss plus equally weighted forward token KL from the frozen baseline on fresh baseline-generated prefixes. |
| `coverage` | Anchored loss plus an on-policy score-function penalty for deviation from baseline mean features. |

All variants initialize from the same baseline, use the seed-matched teacher, and
take at most 160 updates of 32 fresh student sequences at learning rate 1e-6.
The teacher supplies its full next-token distribution, not just selected samples
or a scalar reward. Prefix sampling is detached from the distillation gradient.
There is one optimizer update per fresh batch; no stale-rollout reuse. Forward
KL is teacher-to-student, averaged over non-PAD tokens within each sequence.
The teachers are frozen and dropout is disabled in all generators.

Both baseline-anchored variants use another 32 baseline draws per step. The
coverage variant also draws 1,024 baseline sequences once to fix its target:
128 random Fourier features of unit-normalized, pinned 35M ESM embeddings,
plus length/50 and 20 amino-acid fractions. The RBF bandwidth is fixed from the
baseline sample. A leave-one-out estimator excludes self-pairs; its sampled
discrepancy may be negative. Its policy gradient uses full sequence log
probability, not a length-normalized score or group-centered reward. A bounded
multiplier starts at .1 and updates by .5 times (discrepancy minus .01), clipped
to [0,10]. This is an exploratory moment regularizer, **not** a hard constraint,
an official FBD/MMD surrogate validated on this task, or constrained GRPO.

Every variant has the same fixed 32-baseline-sequence drift probe. An attempted
update exceeding .05 forward token KL is rolled back and training stops. This
small probe is a guard, not a global distribution guarantee. Early stops remain
in the comparison; do not silently restart with a relaxed limit.

Training never loads the reward or evaluation heads. Coverage loads only the
pinned frozen 35M backbone. No 650M or 15B model is used during training. Original
teacher training remains an additional cost. At full completion, new training
draws per seed are 5,152 specialist, 10,272 anchored, and 11,296 coverage. Equal
student-update budgets are not equal compute budgets.

## Run on SSH

Run from `~/Documents/Project/amp_challenge_2027`. Existing dependencies suffice.
Keep all six completed `grpo-generator-seed{42,43,44}-v1` and
`raft-generator-seed{42,43,44}-v1` source runs, their original baseline, reference,
parity artifacts, and evaluator files unchanged. The existing shared provenance
validator checks all six, although RAFT is not an OPD teacher.

First pull and perform a read-only preflight:

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/train_opd.py \
  --variant anchored --seed 42 \
  --out sweep_results/opd-anchored-seed42-v1 --list
```

Train the nine students sequentially. Stop the loop on any failure:

```bash
for variant in specialist anchored coverage; do
  for seed in 42 43 44; do
    CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/train_opd.py \
      --variant "$variant" --seed "$seed" \
      --out "sweep_results/opd-${variant}-seed${seed}-v1" || break 2
  done
done
```

Do not rerun this whole loop over completed outputs: training deliberately refuses
existing directories. Run only missing cells if interrupted between runs. An
interrupted training cell is preserved; do not overwrite it. If a run fails
mid-cell, share the traceback before deciding how to create a new experiment
series. Model/input hashes and completion markers are checked throughout.

Each output contains `policy/`, `history.csv`, `training_samples.csv`,
`guard_reference.csv`, `run.json`, `status.json`, and `complete.json`. Coverage
also stores its feature projection, target and baseline reference samples.
No deployment checkpoint, production generator, or existing experiment is edited.

## Fresh paired evaluation

After all nine training cells finish, these commands compare five methods over
three paired seeds: baseline, GRPO teacher, specialist OPD, anchored OPD, coverage
OPD. The baseline is one fixed model sampled three times, not three independently
trained baselines. All training settings, code and runtime must match across the
nine OPD cells. The directory naming above is the evaluator's expected layout.

```bash
uv run --no-sync python scripts/evaluate_opd.py sample \
  --out sweep_results/opd-evaluation-v1 --list

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/evaluate_opd.py sample \
  --out sweep_results/opd-evaluation-v1

uv run --no-sync python -u scripts/evaluate_opd.py audit \
  --out sweep_results/opd-evaluation-v1

uv run --no-sync python scripts/evaluate_opd.py report \
  --out sweep_results/opd-evaluation-v1
```

Default budget is 8,192 raw draws per cell: 122,880 total, scored with the existing
frozen 35M reward heads and evaluation-only ensemble. Fresh sampling seeds
60042/60043/60044 are distinct from training streams and earlier evaluations.
Use the same GPU/software, full masked categorical sampler, and batches of 32
for every cell. Completed sampling cells are verified and skipped on retry;
mid-cell sampling recovery is not supported. `--method` and `--seed` can select
individual sampling/audit cells; `report` requires the complete comparison.

The CPU audit uses the existing scale-validation implementation. It reports raw
qualifying yield, reference novelty, length strata, and greedy similarity-separated
qualifying yield at nested 2,048/8,192 draws. It selects diagnostic top100 sets
from distinct non-reference samples under distance floors .50/.55/.60/.65, with
the fixed 2,000-candidate shortlist and reference-similarity checks. Evaluation
risk is never used to rank those sets. A shortfall is reported without relaxing
constraints; a greedy shortfall does not prove no feasible set exists.

Read and send back the following:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("sweep_results/opd-evaluation-v1/report")
for name in ["status.csv", "paired_growth_summary.csv", "frontier.csv", "matched_frontier.csv"]:
    print(f"\n{name}")
    print(pd.read_csv(root / name).to_string(index=False))
print((root / "report.json").read_text())
PY
```

`paired_growth.csv` retains per-seed differences against baseline and teacher;
`paired_growth_summary.csv` reports their descriptive mean/std/count. Nested
draw prefixes are not independent replicates. `matched_frontier.csv` reports
quality deltas only when both sets contain 100 and achieved distance differs by
at most .01; otherwise deltas are missing, not zero. Inspect absolute frontiers
and `lengths.csv` too. Do not call a lower risk score an improvement if activity,
diversity support or length coverage deteriorates materially.

## Decision and optional full scale

The initial question is whether anchoring retains a useful portion of the
teacher's qualifying-yield gain while recovering diversity support relative to
specialist-only transfer. Prefer a consistent tradeoff across all three seeds,
not the best seed. No automatic promotion or statistical significance claim is
made. This pilot does not settle whole-library preservation.

After reviewing the pilot, a separate full-scale experiment can use the same
frozen endpoints with 100,000 draws per cell. This is 1.5 million draws plus
15 official library evaluations, so **do not launch it as part of the first test**:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/evaluate_opd.py sample \
  --draws 100000 --out sweep_results/opd-scale-v1

uv run --no-sync python -u scripts/evaluate_opd.py audit \
  --out sweep_results/opd-scale-v1

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/evaluate_opd.py evaluate \
  --out sweep_results/opd-scale-v1

uv run --no-sync python scripts/evaluate_opd.py report \
  --out sweep_results/opd-scale-v1
```

Full-scale libraries are the first 50,000 distinct non-reference sequences in
draw order, without activity/risk filtering. Frontier candidates then come only
from that library. Incomplete libraries are reported, not filled by extra draws.
The official evaluation uses the existing strict 650M suite and seed2027;
inspect FBD, MMD, conformity, diversity, precision/recall alongside proxy gains.
It is separate from training and skips pilot cells lacking a complete library.
The larger run shares the pilot sampling prefix and is not independent evidence.

Neither fresh sampling nor distillation fixes predictor label/generalization
limitations. Evaluation shares inspected data/backbone lineage and is not a new
biological test set. Any further tuning informed by these reports is a new
development iteration, not untouched-test validation. A mixture-inference
comparator and constrained GRPO remain separate future experiments; neither is
implemented by this runner.
