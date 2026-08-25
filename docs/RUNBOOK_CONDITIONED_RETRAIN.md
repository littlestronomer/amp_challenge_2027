# Runbook — charge-conditioned generator retraining

Everything needed to train the conditioned variant of the confirmed v2 recipe
on `istke-compute-1` and fold it back into the selection sweep. The training
itself is **self-supervised** (causal LM over peptides); the only addition is a
computed conditioning input: each training sequence's modlamp (Bjellqvist) net
charge bin, derived from the sequence itself via
`src/amp_challenge_2027/conditioning.py`. No labels, no external annotation.

Why: the unconditioned model produces a too-narrow charge distribution (pool σ
≈ 2.4 vs reference σ ≈ 3.3), which caps ConformityScore. Conditioning lets us
*draw* charge bins from the reference histogram at generation time instead of
accepting the model's preferred marginal.

## Prereqs

```bash
git pull origin main
uv sync --extra ml --extra seqme
uv run pytest -q          # includes test_conditioned_training.py (CPU, ~seconds)
```

## Step 1 — train (v2 recipe + `--conditioning charge`)

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml python scripts/train_generator.py sft \
    --data data/processed/generative.csv \
    --conditioning charge \
    --residual block_attnres \
    --epochs 100 --patience 10 \
    --precision bf16 \
    --out-dir checkpoint/generator-e100_p10-cond \
    --log-dir runs/e100_p10-cond
```

Notes:
- Same budget/architecture as the confirmed v2 recipe (`--epochs 100 --patience
  10`, block_attnres 6L×384H dense defaults), so the comparison against v2/seed44
  isolates the conditioning effect.
- Resume-safe: if it crashes, rerun the identical command — per-epoch checkpoints
  under `checkpoint/generator-e100_p10-cond/checkpoints/` are picked up
  automatically (`--no-resume` to start fresh).

## Step 2 — generate the pool (bins drawn from the reference histogram)

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml python -m amp_challenge_2027.generate \
    --checkpoint checkpoint/generator-e100_p10-cond \
    --out-dir generate/pool-cond-seed42 \
    --temperature 1.0 --top-p 0.9 --repetition-penalty 1.3 \
    --seed 42
```

Conditioning auto-detects from the checkpoint's `config.json`
(`conditioning == "charge"`); per-sequence bins are then drawn from
`data/antibacterial.fasta`'s charge histogram (deterministic given the seed).
No flag needed; `--charge-conditioned` only forces it on/off explicitly.

Sanity check before spending eval time — pool charge spread should approach the
reference's (≈ +2.85 ± 3.28 on the modlamp scale):

```bash
uv run python - <<'EOF'
import statistics
from amp_challenge_2027.conditioning import charge_modlamp
from amp_challenge_2027.data import iter_fasta

seqs = [s for _, s in iter_fasta("generate/pool-cond-seed42/library.fasta")]
charges = [charge_modlamp(s) for s in seqs]
print(f"pool charge: {statistics.mean(charges):+.2f} ± {statistics.stdev(charges):.2f}"
      "  (reference ≈ +2.85 ± 3.28)")
EOF
```

## Step 3 — fold into the selection sweep

Add the new pool to the existing sweep command (see
docs/RUNBOOK_SELECTION_SWEEP.md step 2):

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml --extra seqme python scripts/sweep_selection.py \
    --pools generate/submission-e100_p10-seed44/library.fasta \
            generate/submission-v2/library.fasta \
            generate/pool-seed51/library.fasta \
            generate/pool-cond-seed42/library.fasta \
    --device cuda --mixes '0;0,1,2;0,1,2,3' \
    --out sweep_results/selection-cond
```

Success criterion: a cond-pool cell matches seed44's coverage (FBD ≤ ~0.35,
Recall ≥ ~0.85) while lifting Precision toward ≥ ~0.89 and Conformity ≥ ~0.59 —
i.e., recovers what seed44 loses without giving back what it wins.

## Caveats

- One seed first (seed 42). If it clears the bar, train seeds 43/44 with
  `--seed` for blend material; if not, inspect the charge-bin histogram of its
  raw pool before blaming conditioning.
- Do NOT point `--out-dir` at an existing unconditional checkpoint dir;
  checkpoints/configs would mix.
