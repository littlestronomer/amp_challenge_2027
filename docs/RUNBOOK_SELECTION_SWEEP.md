# Runbook — selection-side improvement sweep (Tier 1)

How to reproduce the blended/reranked library experiments on `istke-compute-1`
after pulling this branch. Everything here runs **without any model
re-training**: pools are pre-generated candidate sets, and all heavy steps are
scoring + selection + seqme evaluation.

## What was added (on this branch)

| piece | purpose |
|---|---|
| `src/amp_challenge_2027/score.py` | component scorers: activity classifier, property-conformity KDE density, ESM kNN precision proxy, z-score combiner |
| `src/amp_challenge_2027/pipeline.py` | shared pool loading / cleaning / composite scorer wiring |
| `src/amp_challenge_2027/metrics_official.py` | Phase-1 protocol metric builder shared by eval CLI and sweep |
| `src/amp_challenge_2027/select.py` | scalability: score-ranked shortlist (default 2000) before novelty/FPS screens |
| `generate.py` | `--pool` ingestion, `--w-*` composite weights, cheap oversampling rounds |
| `scripts/blend_libraries.py` | offline blend + rerank of pool FASTAs → submission pair |
| `scripts/sweep_selection.py` | weight-grid × pool-mix sweep with official metrics |

Design notes:

- The **conformity** component is a numpy port of the ConformityScore idea
  (Gaussian KDE, silverman bandwidths, density-quantile score) over
  `(modlamp charge, KD hydrophobicity, Eisenberg moment)`. It uses a seeded
  reference subsample (`--conformity-sample`, default 12000) so it stays fast.
- The **precision proxy** is mean cosine similarity to the 5 nearest reference
  ESM-2 embeddings. Reference embeddings are computed once and cached under
  `data/cache/ref_embeddings_<hash>.npy`; delete that file after changing
  `--precision-esm`.
- Components that can't run in an environment are dropped automatically and
  weights renormalize, so every command below also works on a torch-less clone
  (just weaker scoring).
- Same seed ⇒ byte-identical outputs everywhere (validator contract).

## Step 0 — pull and sync

```bash
git pull origin main
uv sync --extra ml --extra seqme        # GPU box; light env also works for smoke tests
pytest -q                                # sanity: all pass
```

## Step 1 — collect candidate pools

Each generator checkpoint contributes one raw-candidate FASTA. You likely
already have these as the libraries of previous submissions; reuse them:

```bash
ls generate/submission-e100_p10-seed44/library.fasta   # seed44 (coverage champion)
ls generate/submission-v2/library.fasta                # v2 / seed42 (balanced)
# seed51 (precision/conformity extreme): regenerate once if absent, see below
```

If a pool is missing, regenerate its clean library with the standard entry
point and use that output as the pool:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml python -m amp_challenge_2027.generate \
    --checkpoint checkpoint/generator-e100_p10-seed51 \
    --out-dir generate/pool-seed51 \
    --temperature 1.0 --top-p 0.9 --repetition-penalty 1.3
```

(Any extra flags you used originally — e.g. charge conditioning — apply here
too.) Optionally cap each source (`--pool-cap-per-source N`) if you want equal
contributions instead of union-ranked.

## Step 2 — the sweep

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml --extra seqme python scripts/sweep_selection.py \
    --pools generate/submission-e100_p10-seed44/library.fasta \
            generate/submission-v2/library.fasta \
            generate/pool-seed51/library.fasta \
    --device cuda \
    --grid '1,0.5,0.5;1,0.5,0;1,0,0.5;0,0.5,0.5;1,0.25,0.75;1,0.75,0.25;1,1,1' \
    --mixes '0;0,1;0,1,2' \
    --out sweep_results/selection
```

- Component scores are computed exactly once; each cell only re-weights,
  re-selects, writes `<out>/<cell>/{library,top}.fasta` and evaluates.
- Results land in `sweep_results/selection/results.csv` plus a Pareto table on
  stdout (sorted by FBD).
- Success criterion: a cell with FBD < ~0.35, Recall ≥ ~0.85 AND
  Precision ≥ ~0.89, Conformity ≥ ~0.55 beats the incumbent seed44 library.

## Step 3 — embedder-fidelity check

The protocol numbers above use `esm2_t6_8M`. Re-score the top 2–3 cells with
the 650M embedder to make sure the ranking isn't an artifact of the small
model:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml --extra seqme python scripts/eval_official.py \
    --library sweep_results/selection/<best-cell>/library.fasta \
    --esm-model facebook/esm2_t33_650M_UR50D --device cuda \
    --out sweep_results/selection/<best-cell>/metrics_650M.csv
```

Record both scores in `PHASE1_RESULTS.md`.

## Step 4 — wire the winner into the entry point

Once a cell wins, either

```bash
cp sweep_results/selection/<best-cell>/{library,top}.fasta generate/
```

for immediate validation, or bake it in as the default pipeline invocation
(checkpoint list + weights as `generate.py` defaults) so bare `uv run generate`
reproduces it — required for the final submission.

## Local smoke test (dev machine, no GPU)

```bash
uv run python scripts/sweep_selection.py \
    --pools generate/library.fasta --smoke --out /tmp/sweep-smoke --no-eval
uv run python scripts/blend_libraries.py --pools generate/library.fasta \
    --out-dir /tmp/blend-smoke --library-size 250 --top-k 20
uv run generate --n-sequences 300 --out-dir /tmp/gen-smoke
```

## Known limitations / follow-ups (not in scope here)

- The validator installs no extras (`uv sync`), so bare `uv run generate` still
  falls back to the seeded sampler on a fresh clone — the torch-dependency
  decision (Tier 3) is still open.
- The activity component needs `checkpoint/reward/classifier.pt` (present on
  SSH, absent on the dev machine).
- RL trainer bugs (grad-accum wipe, dropout during rollouts, SFT-overwriting)
  remain unfixed — Tier 2.
