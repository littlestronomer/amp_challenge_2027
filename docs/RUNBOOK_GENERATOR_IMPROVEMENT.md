# Fresh-generation improvement pilot

Purpose: improve the generator's fresh output, not just rerank an unchanged pool.
This adds diverse, on-policy reward-ranked maximum-likelihood fine-tuning
(RAFT-style). It is **not** token-distribution teacher distillation and does not
establish experimentally low hemolysis.

## Protocol fixed before running

- Start from `checkpoint/generator`, exactly as in the GRPO pilot. This is the
  primary generator, not the production two-generator hybrid.
- Reuse verified, frozen deployed activity/hemolysis heads and their pinned 35M
  ESM backbone. Never use the evaluation-only ensemble to choose training samples,
  tune rewards, or select intermediate checkpoints.
- Default teaching gates: activity >= .6 and reward risk <= .5. These are
  exploratory probability gates, not validated biological thresholds.
- Each round draws 1,024 fresh sequences from the current policy. Rank eligible,
  non-reference samples by activity minus risk. Retain up to 128, rejecting any
  with normalized-indel similarity >= .8 to an already retained sequence.
  Length-bin caps are 1.5 times their share of the current raw pool; this is a
  heuristic, not proof of family coverage or baseline-length preservation.
- Fit two epochs in batches of 16: mean per-sequence masked-token NLL, plus .25
  NLL on replay and .05 KL to the frozen initial model on those sampled prefixes.
  Replay is a fixed 256 draws from the starting generator, not another training
  corpus. The model is warm-started; dropout stays off, with gradients on.
- Stop without relaxing gates if fewer than eight diverse teaching examples
  survive. Stop and roll back an update exceeding the .05 batch-prefix KL guard.
  This is not a bound on KL everywhere in sequence space. No automatic restart.
- Save the fixed endpoint or guarded early endpoint. Never choose a checkpoint
  using the evaluation ensemble. Outputs are isolated; existing output dirs fail.
- Same strict deterministic runtime as corrected GRPO, within the same hardware
  and software environment. No cross-platform bitwise reproducibility claim.

## SSH commands

Run from the repository. No dependency changes are required.

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/pilot_generator_improvement.py \
  --checkpoint checkpoint/generator \
  --steps 5 --lr 1e-6 --seed 42 \
  --out sweep_results/raft-generator-seed42-v1 --list
```

The preflight verifies local generator, parity and evaluator artifacts without
loading GPU weights or creating the output directory. Run all three prespecified
seeds unchanged; do not tune on each seed's evaluation results:

```bash
for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/pilot_generator_improvement.py \
    --checkpoint checkpoint/generator \
    --steps 5 --lr 1e-6 --seed "$seed" \
    --out "sweep_results/raft-generator-seed${seed}-v1" || break
done

uv run --no-sync python scripts/summarize_generator_improvement.py \
  --runs sweep_results/raft-generator-seed42-v1 \
         sweep_results/raft-generator-seed43-v1 \
         sweep_results/raft-generator-seed44-v1 \
  --out sweep_results/raft-generator-comparison-v1
```

Run the longer GRPO comparator with identical raw-evaluation metrics, gates,
starting checkpoint, and 5,120 planned training reward-scored draws:

```bash
for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/pilot_generator_improvement.py \
    --method grpo --checkpoint checkpoint/generator \
    --steps 80 --groups 8 --group-size 8 --lr 1e-6 --seed "$seed" \
    --out "sweep_results/grpo-generator-seed${seed}-v1" || break
done

uv run --no-sync python scripts/summarize_generator_improvement.py \
  --runs sweep_results/grpo-generator-seed42-v1 \
         sweep_results/grpo-generator-seed43-v1 \
         sweep_results/grpo-generator-seed44-v1 \
  --out sweep_results/grpo-generator-comparison-v1
```

Both methods use 2,048 fresh evaluation draws and common held-out sampling seeds.
GRPO reward and legacy top100 selection retain their fixed .6 activity floor;
the configurable yield gates only change reporting (and RAFT teaching). Keep all
defaults above to avoid mismatched objectives. The 5,120 training draws match
only if neither run stops early. RAFT replay adds 256 **unscored** initial-policy
draws; training FLOPs, gradient updates and wall time are not matched. Each run
also retains the baseline selection control with evaluation draws plus actual
reward-scored training draws. Do not mix old V2 summaries with this new protocol.

## Inspect the outcome

```bash
uv run --no-sync python -c '
import pandas as pd
for method in ["raft", "grpo"]:
    root = f"sweep_results/{method}-generator-comparison-v1"
    print(f"\n{method.upper()}")
    df = pd.read_csv(f"{root}/paired_summary.csv")
    print(df[df["comparison"].str.endswith("equal_eval")].to_string(index=False))
    print(open(f"{root}/report.json").read())
'
```

Primary metrics use **all raw draws as the denominator**, including repeats and
failures. `raw_evaluation_yield_per_1000` counts distinct non-reference sequences
passing the activity gate and evaluation-only risk gate; `raw_joint_yield_per_1000`
requires both risk heads to pass. Activity is still the training predictor:
this is not fully independent multi-objective validation. The evaluator shares
backbone/data lineage with reward models. Exact-reference exclusion is not the
challenge's full novelty check.

Compare raw yields to `baseline_selection_equal_eval` (the baseline's raw pool,
before selection). Unequal-sized pools change uniqueness fractions and unique
yield denominators: the matched-total control is primarily for **selected**
performance and resource trade-offs. Review paired per-seed values, not just
three-seed means. Low teaching support, KL stops, null top100 results and zero
updates are visible, not silently excluded. A shortfall is a completed diagnostic
run, not an improvement.

Raw diagnostics include evaluation risk mean/p75, unique novel fraction, length,
and mean pairwise distance of the first 256 draws (including duplicates). This
is not an all-library diversity or family-coverage estimate. Selected top100
scores remain secondary. All pool predictions and teaching membership are saved.

Advance only if fresh evaluation/joint yield improves consistently without
material diversity/novelty collapse; investigate scorer disagreement first.
Then generate 50k from the candidate endpoint and rerun official 650M library
metrics and full novelty checks under the existing runbooks. Do not swap in an
ESM 15B embedder and interpret its distances as comparable to official metrics.
Neither predictor gains nor three training seeds guarantee generalizability:
the inspected development evaluator is not a fresh test or experimental assay.
No production generator, selector or submission is promoted automatically.
