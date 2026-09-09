# Exploratory GRPO versus selection-only

## Deterministic replication (v2)

The v1 pilot's second RNG reset unintentionally restored warning-only
determinism. Preserve its results, but do not pool them with v2. V2 reasserts
strict algorithms after seed resets, disables TF32, and forces math-only SDPA
for forward and backward passes. Runtime guards reject changes back to permissive
settings. An unsupported deterministic operation fails rather than continuing
with a warning. Math attention can be slower and use more GPU memory.

Run the unchanged 20-step/1e-6 pilot for all three prespecified seeds in new
directories. Do not increase optimization pressure based on the v1 result:

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/pilot_grpo.py \
  --checkpoint checkpoint/generator \
  --out sweep_results/grpo-pilot-seed42-v2 --list

for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/pilot_grpo.py \
    --checkpoint checkpoint/generator \
    --out "sweep_results/grpo-pilot-seed${seed}-v2" \
    --seed "$seed" || break
done
```

Stop on errors; never disable strict mode to continue. There is no guarantee of
bitwise identity across different GPU models, PyTorch/CUDA versions or platforms.
GPU details and deterministic settings are recorded in each actual run manifest.
Local tests verify identical CPU synthetic outputs and model tensors on repeated
same-seed execution; CUDA repeatability remains a remote check.

After all three complete:

```bash
uv run --no-sync python scripts/summarize_grpo_pilot.py \
  --runs sweep_results/grpo-pilot-seed42-v2 \
         sweep_results/grpo-pilot-seed43-v2 \
         sweep_results/grpo-pilot-seed44-v2 \
  --out sweep_results/grpo-pilot-comparison-v2
```

This verifies completion hashes, seed membership, matching code/configs/artifacts
and runtime, and matched-total draw budgets. It rejects v1 inputs. Outputs include
per-seed results, paired GRPO-minus-baseline deltas, and descriptive mean/sample
standard deviation. Three seeds do not establish statistical significance or
biological safety. Selection shortfalls and KL stops remain visible in per_seed.csv;
paired deltas are left unavailable when either selection is incomplete, and the
number of available pairs is reported. The summary never silently discards a weak
seed or promotes a policy.

This pilot optimizes **predicted** activity/hemolysis, not experimentally verified
safety. It does not change deployed weights or create a submission. The start is
one exported generator, not the historical hybrid blend. Reward heads are the
audited deployed heads from the successful inference parity report, not the newly
trained ablation heads. The latter provide an evaluation-only hemolysis ensemble.

## SSH commands

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/pilot_grpo.py \
  --checkpoint checkpoint/generator \
  --out sweep_results/grpo-pilot-seed42-v2 --list

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/pilot_grpo.py \
  --checkpoint checkpoint/generator \
  --out sweep_results/grpo-pilot-seed42-v2 --seed 42
```

Defaults require `sweep_results/inference-probes-v1/report.json`, completed
`sweep_results/label-ablation-fit-v1`, and `data/antibacterial.fasta`. A pinned 35M
backbone is shared between reward heads and the post-run evaluator. Missing or
mismatched artifacts stop the run; components are never dropped. `--list` checks
artifact metadata/hashes without loading models or writing outputs.

Start with this one bounded smoke run. If successful, run seeds43/44 with new
output directories and the identical declared settings, before drawing conclusions.
Do not select a best seed. Default settings: 20 rollout batches, four groups of
eight samples, two clipped updates per batch, learning rate1e-6, KL coefficient.05,
sampled-prefix KL stop.05, clip.2. Token losses are averaged per sequence. Constant
reward groups have zero policy advantage. Dropout is disabled during both sampling
and gradient calculations. The frozen starting generator supplies reference KL.
Groups have the same unconditional BOS prompt (checkpoint default conditioning).
This is a GRPO-style exploratory implementation, not a claim of optimized GRPO
hyperparameters for this domain.

Sampling uses temperature1 and full categorical probabilities over legal residues
plus EOS, with length8–50. Training recomputes exactly this masked distribution;
no top-p/top-k/repetition processor creates a mismatched behavior likelihood.
EOS is forced after50 residues. A proposed update exceeding the KL stop on its
sampled prefixes is rolled back and training stops; a pre-update exceedance also
stops. This is not a global distribution-distance guarantee.

Reward: activity − risk − 2×max(.6−activity,0), minus1 for batch duplicates and
minus1 for exact reference matches. Scores must be finite in[0,1]. These are proxy
values; .6 is a pilot predictor threshold, not a calibrated biological guarantee.
All selection arms enforce activity>=.6, unique sequences and no exact reference
match. Selection ranks activity−risk except the activity-only diagnostic control.
This is not a reproduction of the existing composite top100 selector.

## Budget and output interpretation

Default training scores640 draws; GRPO evaluation scores2048 fresh draws. The
matched-total selection baseline samples2688 draws from the original model.
If the KL stop reduces training draws, its budget shrinks accordingly. This
matches generator draws scored by the reward models, NOT training FLOPs, and
does not count the separate post-run evaluator as optimization reward.

`summary.csv` contains:

- baseline_activity_control: first2048 baseline draws, activity ranking;
- baseline_selection_equal_eval: same2048 draws, activity−risk ranking;
- baseline_selection_matched_total: all baseline draws, activity−risk ranking;
- grpo_selection:2048 draws from the updated policy, same activity−risk ranking.

Each arm reports pool risk/activity, uniqueness, selected score means, mean
length, and normalized-indel pairwise distance. `selection_complete=false` is a
failed selection budget/threshold outcome, not permission to relax the threshold.
No exact match is permitted, but the stricter submission similarity constraint
and official library metrics are NOT evaluated here.

`evaluation_risk` uses all three original-data hemolysis MLPs from the ablation,
with that ensemble's frozen temperature. They are loaded after RL and never used
for reward or checkpoint selection. They share backbone/data lineage with the
reward predictor: disagreement is informative, agreement is not independent
biological validation. It does not supply an independent activity evaluator.

```bash
uv run --no-sync python - <<'PY'
import pandas as pd
print(pd.read_csv('sweep_results/grpo-pilot-seed42-v2/summary.csv').to_string(index=False))
PY
```

Share the summary plus any KL-stop/shortfall. Inspect activity and diversity losses
alongside risk. Improvement only in reward risk is not success. The saved `policy/`
is an experimental endpoint, not automatically promoted or chosen using evaluator
scores. `training_samples.csv`, `history.csv`, scored pools, selected CSVs,
`run.json`, and completion hashes support auditing. Partial runs do not resume:
keep them and use a new directory. No existing experiment/config/dataset is changed.
