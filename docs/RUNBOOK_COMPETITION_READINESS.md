# Competition readiness review on SSH

Run from the existing SSH checkout after this code is available there. Use the
locked environment with `uv run --no-sync`; do not run `uv sync` or upgrade
packages for this review. Use one writer per output directory. This runbook does
not train models, change ranking weights, promote checkpoints or upload to Kaggle.

## 1. Pull and inventory the remote evidence

```bash
cd /home/istke/Documents/Project/amp_challenge_2027
git fetch origin
git switch feat/nway-blend-multiaxis-conditioning
git pull --ff-only origin feat/nway-blend-multiaxis-conditioning
git status --short
uv run --no-sync python scripts/collect_competition_evidence.py \
  --repo-root . --config experiments/competition_evidence_v1.json \
  --out sweep_results/competition-evidence-v1
```

Use a new output directory for every collection. Exit 2 means a required
artifact is missing or an input/marker is invalid; the report is still written.
Optional absent sources are listed as missing and do not by themselves fail.
Open `STATUS.md`, `inventory.json` and `metrics.csv`. Marker verification checks
integrity, not scientific meaning. Read each producer report and protocol.

If family-held-out, label-audit, selection or top-100 experiment roots are
missing, consult their runbooks and `--list` preflights before any expensive
inference. Do not unlock or rerun the already-inspected test. Do not regenerate
source libraries because an audit summary is absent.

## 2. Run the strict incumbent release check

This samples the default 50,000-sequence incumbent with both shipped generator
checkpoints, seed 42 and all nonzero default scoring components. It may need
cached Hugging Face weights and the established GPU runtime. Stop on failure;
do not accept fallback generation or a dropped component as a pass.

```bash
uv run --no-sync generate --strict \
  --out-dir generate/competition-readiness-v1
```

Strict mode rejects changed recipe flags, pools, missing model files, fallback
generation, omitted scorers, nonfinite scores, undersized outputs, library
reference overlap and top-100 similarity above 0.8. A manifest is written only
after outputs pass. Review:

- `generate/competition-readiness-v1/generation_manifest.json`
- `generate/competition-readiness-v1/library.fasta`
- `generate/competition-readiness-v1/top.fasta`
- `generate/competition-readiness-v1/top_scores.csv`

The manifest records checkpoint/config/reference hashes, scorer names and
metadata, available pretrained-backbone revisions, runtime, source identity and
output hashes. A null backbone revision is an explicit limitation. Model scores
are ranking surrogates, not assay results. Keep the output directory intact.
Strict mode requires a fresh output directory. For byte comparison across
independent runs, use another fresh output directory with the same code, runtime,
seed and hardware.

## 3. Audit the whole top-100

This analyzes the FASTAs and 100 saved score rows; it does not rerun scorers.

```bash
uv run --no-sync python scripts/report_top100_readiness.py \
  --library generate/competition-readiness-v1/library.fasta \
  --top generate/competition-readiness-v1/top.fasta \
  --reference data/antibacterial.fasta \
  --scores generate/competition-readiness-v1/top_scores.csv \
  --out sweep_results/top100-readiness-v1
```

Review `REPORT.md`, `summary.json` and `candidates.csv`. The report verifies
sizes, membership, uniqueness, plausibility, maximum reference similarity,
near-identical top pairs/families, score coverage and score tails. Its optional
10,000 random subsets of 25 summarize variation in fixed surrogate scores and
family representation. They do not predict wet-lab outcomes. Synthesizability
remains `not_assessed` without a separate sourced assessment. Hemolysis is not
run by strict generation because its default ranking weight is zero.

## 4. Refresh the inventory

After Steps 2 and 3, collect again in a new directory:

```bash
uv run --no-sync python scripts/collect_competition_evidence.py \
  --repo-root . --config experiments/competition_evidence_v1.json \
  --out sweep_results/competition-evidence-v2
```

Strict-generation and top-100 rows should show verified producer markers and
output hashes. Missing experiments remain visible. Record a dated entry in
`docs/COMPETITION_WORKLOG.md`; update `docs/COMPETITION_STATUS.md` with source
paths, hashes, population, comparison and limitations. Do not call the result
scientifically validated just because an inventory marker passes.

## 5. Recheck model inference parity if missing or stale

Choose new output directories if these already exist.

```bash
uv run --no-sync python scripts/audit_inference_parity.py \
  --out sweep_results/inference-metadata-readiness-v1
```

If `sweep_results/reward-reconstruction-v1` exists and its markers verify,
replay the fixed validation probes:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/audit_inference_parity.py \
  --reconstruction sweep_results/reward-reconstruction-v1 \
  --device cuda --out sweep_results/inference-probes-readiness-v1
```

Do not recreate a missing reconstruction; report the probe unavailable and
inspect its existing runbook first.

## 6. Decide what to do next

Use `docs/NEXT_EXPERIMENT_DECISION.md` after reviewing results. Select one
bounded experiment only if existing evidence identifies a resolvable weakness.
Reuse verified frozen artifacts where producer contracts allow it. Preserve the
inspected-test status.

No external predictor benchmark is ready. First identify a legally reusable
source with matching organism, assay and chemistry; document its provenance and
audit sequence-family overlap against existing splits. See
`experiments/external_predictor_protocol_v1.json`. Another split of the current
DBAASP-derived records is not independent confirmation.

The indexed AMP Challenge overview describes advancement of up to 20 teams and
random selection of 25 peptides from each advancing team's top-100. Record an
actual submission receipt, score and rank if available. Local FBD, MMD and
classifier scores do not determine hidden competition rank.
