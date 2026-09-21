# Frozen-pool hemolysis sensitivity runbook

Run on the SSH machine after pulling the implementation commit on the existing
feature branch. Keep the current artifacts and use the exact directories below.
The first command is read-only and checks source hashes, protocol and model
metadata without loading a model.

## Optional strict repeatability bundle

The reported strict runs wrote to the same output directory, so their hash
agreement is user-reported rather than a durable two-run artifact. To create a
portable comparison with separate manifests, generate into two fresh folders
and then bundle the results:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync generate --strict \
  --out-dir generate/strict-run-1
CUDA_VISIBLE_DEVICES=1 uv run --no-sync generate --strict \
  --out-dir generate/strict-run-2
uv run --no-sync python scripts/compare_strict_generation_runs.py \
  --run1 generate/strict-run-1 --run2 generate/strict-run-2 \
  --out generate/strict-repeatability-v1
```

The bundle compares recipe fields after ignoring only `out_dir`, verifies each
manifest before comparing the library, top and score CSV hashes, and retains
runtime metadata. Runtime differences remain a limitation even when bytes match.

```bash
uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/selectivity-risk-v1 --list
```

If preflight passes, measure a bounded 1,024-sequence run on physical GPU 1. The
result is intentionally incomplete and returns status 3 after writing a
resumable cache. Do not interpret that as experiment completion.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/selectivity-risk-v1 --device cuda --batch-size 64 \
  --seeds 42 --max-new-sequences 1024
```

Resume seed 42 without a budget. The incomplete status is expected until all
seed-42 library sequences have valid cached predictions.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/selectivity-risk-v1 --device cuda --batch-size 64 --seeds 42
```

After reviewing seed-42 throughput and available GPU memory, complete all three
generation seeds. Keep the same device, batch size, source, protocol and output
directory so the cache identity matches.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/selectivity-risk-v1 --device cuda --batch-size 64 \
  --seeds 42 43 44
```

The comparison uses only cached scores and frozen arrays; it should run on CPU.
Run its preflight first, then produce a new report directory.

```bash
uv run --no-sync python scripts/compare_selectivity.py \
  --source sweep_results/epoch58-top100-v1 \
  --risk-cache sweep_results/selectivity-risk-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/selectivity-comparison-v1 --list

uv run --no-sync python scripts/compare_selectivity.py \
  --source sweep_results/epoch58-top100-v1 \
  --risk-cache sweep_results/selectivity-risk-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/selectivity-comparison-v1
```

Review `REPORT.md`, `criteria.json`, `paired_deltas.csv`, all per-seed summaries,
and `complete.json`. Keep the incumbent unless every predeclared criterion passes;
even then the result only supports independent evaluation. Never modify source
run manifests, weights, novelty, shortlist, risk label meaning or penalty after
seeing the results. On a source mismatch, corrupt marked chunk, parity failure,
OOM or underfilled selection, stop and preserve the cache for diagnosis. If a
runtime setting must change, create a new output directory and document the
changed identity.

To inventory the resulting run, use a new evidence output directory:

```bash
uv run --no-sync python scripts/collect_competition_evidence.py \
  --config experiments/competition_evidence_v2.json \
  --out sweep_results/competition-evidence-v2
```

The evidence config includes both selectivity outputs as optional sources; an
incomplete cache remains explicitly missing/unverified until its completion
marker and all cell hashes verify. Do not edit source lists or completion
markers to make a partial cache appear complete.
