# Generator experiments: checkpoint choice and library membership

For the next experiment after checkpoint confirmation, see the
[fixed-ratio epoch-58 blend runbook](RUNBOOK_BLEND_RATIOS.md).

Run the experiments on `istke-compute-1`. Develop, test, commit and push code
locally, then pull on the training machine. The commands below write to new
experiment directories and do not promote or change the shipped submission.

## 1. Pull and check the environment

From the existing AMP repository on the SSH machine:

```bash
git pull --ff-only origin main
uv sync --locked --extra ml --extra seqme
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run --no-sync pytest -q
```

The CUDA package remains selected by the lockfile; the tests above use CPU.
Full generation/evaluation below uses GPU 1. Check `nvidia-smi` before starting
and use another free GPU if necessary. Do not overlap with an existing job on
the same GPU. A `tmux` session keeps the foreground sweep running if SSH drops.

## 2. Inspect the checkpoint sweep before running it

The historical seed44 training run is recorded at
`checkpoint/generator-e100_p10-seed44`. It must contain its original
`config.json`, `model.pt`, and `checkpoints/ckpt_epoch*.pt` archives.

```bash
uv run --no-sync python scripts/sweep_checkpoints.py \
  --checkpoint-dir checkpoint/generator-e100_p10-seed44 \
  --max-snapshots 4 --include-hybrid \
  --out sweep_results/checkpoints-seed44-v1 --list
```

By default, this selects four spaced archived epochs, the run's inference
model, and the shipped 75/25 hybrid as an explicit baseline. The printed plan
shows exactly what was found. If archives were removed, the inference model
and hybrid are still runnable, but there is no historical trajectory to sweep.
Use `--epochs 20 40 60` only when those exact archives exist; missing requested
epochs cause an error. `--max-snapshots 0` evaluates inference only, plus the
hybrid if requested.

## 3. First experiment: compare saved checkpoints

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/sweep_checkpoints.py \
  --checkpoint-dir checkpoint/generator-e100_p10-seed44 \
  --max-snapshots 4 --include-hybrid \
  --out sweep_results/checkpoints-seed44-v1
```

Defaults: 50,000 sequences, 100,000 raw candidates per generator, generation
seed 42, temperature 1.0, top-p 0.9, repetition penalty 1.3, and
`facebook/esm2_t33_650M_UR50D` for local evaluation. The historical checkpoint
and current inference model are sampled alone; blending happens ONLY in the
explicit `hybrid` cell. This avoids the normal entry point's automatic blend
when evaluating alternative checkpoints.

Every cell contains:

- `library.fasta`: exactly 50,000 valid, unique, non-reference sequences.
- `pool.fasta`: all clean candidates, retained for the selection experiment.
- `generation.log`, `evaluation.log`, `generation_stats.json`.
- `metrics.csv`, `metrics.json`: local seqme results.
- Hash manifests for inputs and completed outputs.

`results.csv` is updated after each completed cell. The sweep stops on failed
generation, insufficient clean candidates, or incomplete metrics. Rerunning
the identical command reuses verified completed stages. `--generate-only`
can populate pools first; remove that flag to evaluate the same run later.
Changes to code, checkpoints, reference, seeds or evaluation parameters require
a NEW `--out` directory so results cannot be silently mixed. Do not run two
processes writing the same experiment directory.

### Recovering the seqme 0.5.1 evaluation failure

The locked seqme does not export `Amphiphilicity`. The initial strict evaluator
incorrectly required the auxiliary `ConformityScore(amp+charge)` readout.
The fix keeps this readout optional and still requires all 12 supported
metrics, including the core charge/hydrophobicity/hydrophobic-moment conformity
score. The HF unauthenticated-download warning is unrelated to this failure.

After pulling the fix, keep the failed run intact and import its verified
generation into a new run (do not edit `run.json` to bypass provenance checks):

```bash
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/sweep_checkpoints.py \
  --checkpoint-dir checkpoint/generator-e100_p10-seed44 \
  --max-snapshots 4 --include-hybrid \
  --reuse-generated-from sweep_results/checkpoints-seed44-v1 \
  --out sweep_results/checkpoints-seed44-v2
```

`--reuse-generated-from` is an explicit choice to reuse datasets produced by
older code; use it only after evaluation-only fixes, not generation changes.
It checks weights, reference, sampling/settings, runtime versions and output
hashes, copies only completed generation stages, and records the original
code provenance in `generation_source.json`. All metrics are recomputed.
The source is read-only and is still needed for verified reruns of this command;
keep the reuse flag when resuming. Missing cells are generated normally.
Original generation logs remain in the source directory.

For this recovery, use `checkpoints-seed44-v2` instead of `checkpoints-seed44-v1`
in the library-selection commands below.

These are library experiments, not submission pairs: they do not produce
`top.fasta`. The original submission entry point and checkpoints are unchanged.
The 650M metrics are our local comparison protocol, not the organizers' full
aggregation score. Independent potency and synthesizability axes remain to be
added before concluding that qualification performance has improved.

## 4. Second experiment: select stronger library members

The baseline remains the actual hybrid library from step 3. Candidate pools
contain the additional generated peptides, so we can test membership changes
without another training run.

First verify `checkpoint/reward/classifier_panel_config.json` on the training
machine. The tracked shared `config.json` is stale. The per-artifact file must
come from the actual frozen training run and its calibrated promoted member:
`task=panel`, `unfreeze_layers=0`, `checkpoint_format=head-only`, and its real
positive temperature. Do not reconstruct a temperature or AUROC from rounded
prose in the experiment record. The script refuses missing/stale metadata.
The gitignore now allows all three per-artifact classifier configs to be
committed once their provenance has been reconciled with the weights.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/sweep_library_selection.py \
  --baseline sweep_results/checkpoints-seed44-v1/hybrid/seed42/library.fasta \
  --pools sweep_results/checkpoints-seed44-v1/hybrid/seed42/pool.fasta \
          sweep_results/checkpoints-seed44-v1/inference/seed42/pool.fasta \
  --strengths 0 0.25 0.5 \
  --out sweep_results/library-membership-v1
```

Add promising archived-checkpoint pools to `--pools` after the first sweep.
No implicit glob expansion or automatic winner promotion occurs.

Selection keeps EXACT counts for each joint `(length, charge_bin)` in the
baseline. Within each bin, strength 0.25 allows replacing at most 25% of its
members with higher-scoring candidates (rounded down). Strength 0 copies the
baseline FASTA byte-for-byte. This limits distribution shifts but does not
guarantee preserved embedding diversity, hydrophobicity, or biological activity;
those must be checked in the results.

Default scores are mean calibrated panel probabilities. They are selection
surrogates, not independent validation. `selection_score_mean` and
`selection_score_p10` should improve by construction, so do not count that as
evidence of experimental potency. To use an independent scoring source, pass
`--scores path/to/scores.csv` with unique `sequence,score` rows (higher is
better), covering every clean pool and baseline member. Missing/duplicate rows
and non-finite scores fail immediately. Scores are cached in `pool_scores.csv`.

## 5. Confirm finalists and decide what to train next

Re-evaluate only promising epochs with `--seeds 42 43 44` in a new directory.
Compare against the hybrid generated with each matching seed. Keep the
incumbent until the broader evaluation supports a change; FBD alone is not an
adoption criterion. Validate the final top-100 after any eventual promotion.

New SFT runs have corrected checkpoint semantics:

- Root `model.pt` is the best evaluated validation checkpoint; `selection.json`
  records the chosen step and loss.
- `last/model.pt` is the final training state, always saved separately.
- With no validation evaluation, the root explicitly records `last_no_validation`.
- Patience counts consecutive non-improving evaluations and resets on improvement.
- `--split-seed` fixes the data split independently of initialization seed.
- `training_manifest.json` records data/split hashes and architecture config; resume
  rejects different inputs or legacy runs without this manifest. Historical
  checkpoints remain usable for generation/sweeps.

If saved checkpoints cannot answer the question, start a new controlled run:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/train_generator.py sft \
  --data data/processed/generative.csv \
  --seed 42 --split-seed 42 \
  --residual block_attnres --num-layers 6 --hidden-size 384 --num-heads 6 \
  --epochs 100 --patience 10 --eval-every 500 --save-every 2000 \
  --precision bf16 \
  --out-dir checkpoint/generator-fixed-split-seed42-v1 \
  --log-dir runs/generator-fixed-split-seed42-v1
```

For initialization replicates, change `--seed` and the two output directories
while keeping `--split-seed 42`. The split is still random at sequence level;
these fixes do not establish independence between homologous families or
out-of-sample biological generalization. Historical metrics remain measurements
of the historical artifacts, not results of this corrected training procedure.
Resume supports bookkeeping continuity, not bitwise equivalence to an
uninterrupted training run (RNG/data-loader state is not restored).
