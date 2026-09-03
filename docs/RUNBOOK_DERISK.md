# Runbook — de-risking the current position (A4)

Four validation actions that must land BEFORE any further model improvement.
All are measurement/validation; the deployed submission stack only changes
through the explicit gate decisions at the end. Code: commits through
`clustered-split + hemolysis safety` on `main`.

Rough budget: ~2–2.5 GPU-hours total on `istke-compute-1`.

## Step 0 — sync

```bash
cd ~/Documents/Project/amp_challenge_2027
git checkout main && git pull origin main
uv sync --extra ml --extra seqme
uv run pytest -q        # expect all green incl. test_derisk.py
```

## A4.1 — Clustered-split classifier eval (the honest AUROC)

Retrains the panel head with identity-clustered train/val (Levenshtein ratio
≥ 0.70, no homolog straddles the split). **Separate out-dir is mandatory** —
the script refuses to write a clustered run into the deployed `checkpoint/reward/`
(promotion would overwrite `classifier_panel.pt`).

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run --extra ml python scripts/train_reward_classifier.py \
    --panel --split clustered --cluster-threshold 0.7 \
    --ensemble-size 3 --epochs 50 \
    --out-dir checkpoint/reward_clustered \
    > ~/panel_clustered.log 2>&1 &
# ~35 min (3 members × ~10 min); log line to watch:
#   [reward] CLUSTERED split: N clusters (threshold 0.7) → train: … val: …
```

**Gate on the best member's macro AUROC:**
- ≥ 0.75 → potency narrative holds; quote this number everywhere.
- 0.70–0.75 → temper claims in docs/abstract; stack unchanged.
- < 0.70 → temper claims AND switch deployment to mean-of-3-members inference
  before relying on ranker numbers again.

## A4.2 — 650M-embedder robustness check (zero code)

Re-scores both candidate libraries under the official-fidelity embedder. Run
after A4.1 finishes (same GPU).

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml --extra seqme python scripts/eval_official.py \
    --library generate/submission-e100_p10-seed44/library.fasta \
    --esm-model facebook/esm2_t33_650M_UR50D --device cuda \
    --out ~/eval_seed44_650m.csv
CUDA_VISIBLE_DEVICES=1 uv run --extra ml --extra seqme python scripts/eval_official.py \
    --library generate/submission-v2/library.fasta \
    --esm-model facebook/esm2_t33_650M_UR50D --device cuda \
    --out ~/eval_v2_650m.csv
```

**Gate:** seed44 keeps its FBD/MMD/Recall lead → instrument-bet de-risked.
Lead flips to v2/seed42 → the submission library switches to the 650M winner
(checkpoint copy + regenerate + byte-check, same procedure as the seed44
promotion).

## A4.3 — HC50 safety head

**3a. Refetch with the corrected schema, then inspect (writes no labels).**
The original fetch's `kind` column was empty (wrong key — v4 stores the
measure in `activityMeasureForLysisGroup`); the data is also (concentration,
percent-lysis-band) pairs, not HC50 values. `--hemo-refetch` rewrites only
`hemolysis_raw.csv` using the server-side hemolytic filter (~13.9k peptides,
~15–30 min):

```bash
nohup uv run python scripts/fetch_dbaasp_v4.py --hemo-refetch     > ~/hemo_refetch.log 2>&1 &
# …then:
uv run python scripts/build_hemolysis_labels.py --report | tee ~/hemo_report.txt
```

Review the kind×outcome table; adjust `--risk-band`/`--safe-band`/`--ceiling`
(defaults 40% / 30% / 128 µM) or `--targets` if the full-data distribution
disagrees with the 250-peptide sample that set them.

**3b. Build labels + train the head** (separate dir; ~10 min):

```bash
uv run python scripts/build_hemolysis_labels.py --out data/processed/hemolysis_labels.csv
CUDA_VISIBLE_DEVICES=1 uv run --extra ml python scripts/train_reward_classifier.py \
    --data data/processed/hemolysis_labels.csv \
    --out-dir checkpoint/reward_hemo --ensemble-size 3 --epochs 50 \
    2>&1 | tee ~/hemo_train.log
```

(Direction convention: label "active" = RISKY; `HemoScorer` exposes
`safety = 1 − p(risky)`.)

**3c. Safety audit + tradeoff measurement** on the adopted top-100:

```bash
uv run python - <<'EOF'
from amp_challenge_2027.score import HemoScorer
from amp_challenge_2027.data import iter_fasta
hemo = HemoScorer.load(device="cuda")
assert hemo is not None
p = hemo.p_risky([s for _, s in iter_fasta("generate/submission-final/top.fasta")])
import numpy as np
print(f"top-100 p(risky): mean={p.mean():.3f} p75={np.percentile(p,75):.3f} max={p.max():.3f}")
EOF
```

**Gate:** top quartile p_risky > ~0.6 → re-rank with
`--w-safety 0.25` (then 0.5 if needed) and check panel potency stays at the
adopted profile; adopt the smallest weight that defuses the risk. Already-low
risk → record and leave the recipe untouched (`--w-safety` default stays 0).

## A4.4 — Record (docs)

Paste the four outputs (clustered log tail, both 650M CSVs, hemo report +
audit) back to the dev session; PHASE1_RESULTS.md gets: clustered AUROC
alongside the random-split 0.863, the 650M comparison, the safety audit, and
the in-sample framing of distributional metrics. These paragraphs are the
abstract's skeleton.

## After the gates

Only once the position is verified (or the library switched per A4.2) do the
improvement tracks resume: L0 decode-parameter sweep → conditioning → RAFT.

## After the gates — L0 decode-parameter sweep (first improvement track)

Decode params (T / top-p / rep-penalty) were hand-set once; this sweeps a
3×3 grid around the default on the promoted seed44 checkpoint. Pure
inference, no code changes to generation itself:

```bash
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/sweep_decode.sh \
    > ~/decode_sweep.log 2>&1 &
# ~2 h (9 cells × ~14 min). Summary lands in sweep_results/decode/summary.tsv
```

Read-out: the anchor row (`t1.00-p0.90-r1.3`) re-baselines the current
default under identical eval; adopt a new decode config only if FBD/MMD
improve beyond seed-noise without sacrificing Conformity — then it becomes
the `generate.py` defaults (and regeneration + byte-check follow, same as
the seed44 promotion).
