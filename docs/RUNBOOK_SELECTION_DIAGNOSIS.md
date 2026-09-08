# Cached selection diagnosis and predictor artifact inventory

This implements the first delivery of [the next-iteration plan](PLAN_NEXT_ITERATION.md):
Phases 0/1A, the artifact-identity portion of 1B, and the bounded P0/P1 experiment.
The combined experiment includes the diagnostic control replays. It does not
change deployed ranking, generate peptides, load neural backbones, retrain
classifiers, or run library FBD/MMD again.

Full validation/calibration reproduction needs original training data and split
evidence. Pool hemolysis scoring, safety-weight tuning, fresh-seed confirmation,
and any promotion remain separate, gated follow-ups, not implemented switches.

## 1. Pull and verify the frozen six-cell cache

Run these commands on the SSH machine. Keep the existing environment; do not
run an unnecessary `uv sync` or change any of the per-artifact classifier configs.

```bash
cd /home/istke/Documents/Project/amp_challenge_2027
git pull --ff-only origin main

uv run --no-sync python scripts/sweep_top100_selection.py \
  --source sweep_results/epoch58-top100-v1 \
  --out sweep_results/selection-normalization-v1 \
  --list
```

`--list` hashes/validates every cache and prints 12 proposed cells without writing.
It requires exactly `hybrid` and `p3_s1` at generation seeds 42, 43, 44, plus the
comparison's `backbones.json` and `reference_cache.json`. Scores stay aligned to
the saved FASTA, never a sorted/reconstructed sequence list. Current local head
files are not needed for cached selection; their original identities are in the
source manifest. The separate artifact audit checks today's deployed heads.

If a source is missing or damaged, stop and inspect it. Do not regenerate the
old experiments or edit their manifests to make the check pass.

## 2. Inventory the three deployed predictors (CPU)

```bash
uv run --no-sync python -u scripts/audit_reward_artifacts.py \
  --source sweep_results/epoch58-top100-v1 \
  --out sweep_results/reward-artifact-audit-v1
```

This uses Torch only to inspect head tensors, not to load ESM. It requires
explicit per-artifact configs and compares their hashes with the original
comparison. It inspects the two reward directories and their immediate
`member*` subdirectories; it does not search the whole machine.

`provenance_incomplete` is an expected, honest result until the original
training records are established. `invalid_artifact` exits with status 2 and
records the failure: investigate it before making new neural predictions or
starting a risk-guided experiment. A fixed-cache P0/P1 analysis can still
diagnose the previously recorded scores, but does not certify the current head.

Read the report:

```bash
uv run --no-sync python -m json.tool \
  sweep_results/reward-artifact-audit-v1/inventory.json
```

If original runs are elsewhere, rerun with one `--training-run /actual/run/path`
per known directory, and a **new** output name. Additional original label/split/log
files can be recorded with repeatable `--record /actual/file/path` arguments.
These records are inventoried by hash, not automatically certified as belonging
to a head. Tensor matches identify weights even if Torch serialization differs.
Multiple matches remain ambiguous; a matching rounded AUROC does not pick a
winner, split, or training seed. Shared `config.json`/`members.json` may have
been overwritten by another training task.

The report explicitly lists the evidence needed before a bounded validation
replay. It does not fit a temperature, invent a new holdout from training data,
or interpret predicted risk as measured percent hemolysis.

## 3. Run the controlled normalization ablation (CPU)

```bash
uv run --no-sync python -u scripts/sweep_top100_selection.py \
  --source sweep_results/epoch58-top100-v1 \
  --out sweep_results/selection-normalization-v1
```

No `CUDA_VISIBLE_DEVICES` is needed. Use `tmux` for a long SSH session: exact
novelty screens still compare each shortlist with the full reference on CPU.
There are no downloads or new 35M/650M inference calls.

- P0: existing per-library scaling. All six replays must produce byte-identical
  original `top.fasta` files before **any** P1 cell runs.
- P1: fit scaling once on the full seed-42 hybrid library, serialize its
  means/stds/denominators and reuse them for every cell. Its seed-42 hybrid
  control must also be byte-identical to the original top.
- Five weights, shortlist 2,000, plausibility, exact novelty <= 0.8, ranking
  seed 42 and the production diversity selector are unchanged.
- Source 50k FASTAs remain read-only. New top lists must be 100 unique, valid,
  plausible and novel members of their own library. Source library hashes,
  rather than redundant library copies, identify their membership.

The standalone `scripts/analyze_selection.py` accepts the same arguments but
only runs P0. Use it with its own output directory if only diagnostics are
wanted. Running it first is **not necessary** for the combined P0/P1 command.

Identical reruns resume verified completed cells. Changed source files,
normalization protocol, code/runtime or input records require a fresh output
directory. Do not run concurrent writers. A parity failure is a stop condition,
not permission to relax the gate. Root summaries are written once all requested
cells finish; per-cell `summary.json`/`complete.json` show partial progress.

## 4. Read the measurements

```bash
uv run --no-sync python -c '
import pandas as pd
root = "sweep_results/selection-normalization-v1"
df = pd.read_csv(f"{root}/results.csv")
cols = ["policy", "case", "seed", "matches_original_top_bytes",
        "activity_mean", "panel_mean_probability", "breadth_mean", "mdr_mean",
        "top_mean_pairwise_distance", "reference_similarity_max",
        "hemo_risk_known_count", "hemo_risk_status", "hemo_risk_mean"]
print(df[cols].to_string(index=False))
print("\nP1 MINUS P0 WITHIN EACH LIBRARY")
print(pd.read_csv(f"{root}/policy_paired_deltas.csv").to_string(index=False))
print("\nCANDIDATE MINUS HYBRID WITHIN EACH POLICY")
print(pd.read_csv(f"{root}/case_paired_deltas.csv").to_string(index=False))
print("\nEFFECTIVE COMPONENT WEIGHTS")
print(pd.read_csv(f"{root}/effective_weights.csv").to_string(index=False))
'
```

Send these tables and the predictor inventory for the next decision. Additional
files explain where a change originates:

| Output | Meaning |
|---|---|
| `normalization_anchor.json` | Frozen anchor library/score hashes and fitted scales |
| `component_distribution.csv` | Raw quantiles/variance/ties/saturation at all five stages, including each panel output |
| `effective_weights.csv` | Both `weight / denominator` and the coefficient after division by total weight |
| `selection_funnel.csv` | Library → plausible → shortlist → novel shortlist → top counts |
| `component_contributions.csv` | Weighted, standardized contributions to the final average; may be negative |
| `rank_correlations.csv` | Spearman association with composite rank within each stage; constant values are undefined |
| `baseline_results.csv` | Six replay controls, not six new experimental replicates |
| `seed_summary.csv` | Generation-seed mean/sample SD; missing-seed measurements do not receive a pooled mean |
| `<policy>/<case>/seedN/top_scores.csv` | Sequence-aligned raw scores, panel probabilities, risk coverage and novelty |
| `<policy>/<case>/seedN/stages.npz` | Ordered zero-based row indices into that cell's original library FASTA |

`duplicate_fraction` is `1 - number_of_unique_values / n`;
`tied_observation_fraction` counts all observations whose exact value occurs
more than once. They are different quantities. `fraction_at_one` is most
useful for thresholded breadth/MDR, not as a saturation definition for every
continuous predictor.

Missing hemolysis measurements are deliberate: old caches score only the old
top-100, not all 50k sequences. A new top reuses risks only for exact sequence
matches to its own cell's original top. If even one risk is missing, the new
top's mean/p75/max are null, with `hemo_risk_known_count` reporting coverage.
Blank values/NaN when Pandas reads these CSVs mean **not measured**, not zero risk.
Do not compare a partial risk average against a complete control. No full-pool
risk cache or later safety-weight sweep is enabled by this delivery.

Seeds 42–44 are exploratory: we already inspected them and used them to choose
this iteration. Higher selection-surrogate scores are not independent activity
validation. Sequence pairwise distance is not seqme library Diversity. Cached
FBD/MMD remain unchanged because the 50k library bytes remain unchanged.

## ESM-2 15B: a separate benchmark, not the next default

The intended model is [facebook/esm2_t48_15B_UR50D](https://huggingface.co/facebook/esm2_t48_15B_UR50D).
Its masked-language-model objective does not supply a calibrated AMP/hemolysis
classifier or a drop-in autoregressive generator. The official model table
lists 5,120-dimensional 15B embeddings versus 480 for our frozen 35M heads;
changing only the backbone name cannot load the existing heads correctly.
[Official ESM model table](https://github.com/facebookresearch/esm#pre-trained-models).

Recommendation: finish the cached diagnosis and establish predictor evidence
first. If a larger-backbone question remains, test one of two explicit hypotheses:

1. **Embedding robustness:** profile a small fixed set first, then compare
   matched reference/generated samples within each embedding protocol. Recompute
   both sides; do not compare absolute FBD/MMD values across 650M and 15B spaces.
   Predeclare sampling, pooling, layer, dtype and covariance regularization;
   a small-sample 5,120-dimensional FBD can be statistically ill-conditioned.
2. **Predictor improvement:** train a new task-specific head on frozen larger
   embeddings and evaluate/calibrate with a documented leakage-controlled split.
   Keep a matched 35M baseline. This is a separate training experiment, not a
   way to reuse the existing 480-input head or its temperature. A 650M/3B pilot
   is a reasonable resource-cost checkpoint before committing to 15B.

Using the approximate 15B parameter count, weights alone need about **30 GB
at two bytes per parameter**, or **60 GB in FP32**, before activations, loading
overhead and other runtime allocations. These are arithmetic estimates, not
a guarantee that a particular GPU can run it. Official ESM documentation warns
about 15B out-of-memory failures and gives a CPU-offloading example.
[Official large-model inference guidance](https://github.com/facebookresearch/esm#cpu-offloading-for-inference-with-large-models).

Our current `eval_official.py` does not expose dtype, offloading or sharding
controls. Do **not** substitute `--esm-model ...15B...` into the full sweep as
the first trial. No 15B download or experiment has been launched. Before a
separate pilot, report hardware with:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
free -h
```
