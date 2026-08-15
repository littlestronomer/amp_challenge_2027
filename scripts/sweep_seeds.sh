#!/usr/bin/env bash
# sweep_seeds.sh — train + generate + evaluate the v2 AttnRes recipe across seeds.
#
# Purpose: measure the seed-to-seed variance of the full Phase-1 pipeline so we
# can tell whether the v2 submission's metric profile is typical or fortunate.
#
# Per seed:
#   1. SFT train  (v2 recipe: block_attnres, 6L/384H/6H, 10 epochs, defaults)
#   2. Generate   (50k library + top-100, repetition-penalty 1.3, seed fixed at 42
#                  so differences across runs come from the model, not sampling)
#   3. Evaluate   (full 11-metric protocol via eval_official.py, CSV per seed)
# At the end: aggregated per-seed table + mean/std/min/max across seeds.
#
# Usage (from the repo root, on the SSH machine):
#   bash scripts/sweep_seeds.sh                          # defaults: GPU 1, seeds 43-52
#   GPU=0 SEEDS="43 44 45 46 47" bash scripts/sweep_seeds.sh &
#   GPU=1 SEEDS="48 49 50 51 52" bash scripts/sweep_seeds.sh &   # split across both GPUs
#   EPOCHS=100 PATIENCE=10 bash scripts/sweep_seeds.sh   # converged recipe instead
#
# Resumable: a seed is skipped entirely if its metrics CSV already exists;
# training is skipped if the checkpoint already exists.

set -u

SEEDS="${SEEDS:-43 44 45 46 47 48 49 50 51 52}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-}"          # empty = fixed-epoch budget (no early stopping)
GPU="${GPU:-1}"
ESM_MODEL="${ESM_MODEL:-facebook/esm2_t6_8M_UR50D}"
RESULTS_DIR="${RESULTS_DIR:-sweep_results}"
PY="${PY:-.venv/bin/python}"

mkdir -p "$RESULTS_DIR"

# Optional --patience flag (only when PATIENCE is set).
PATIENCE_FLAGS=()
if [ -n "$PATIENCE" ]; then
  PATIENCE_FLAGS=(--patience "$PATIENCE")
fi

for SEED in $SEEDS; do
  CKPT="checkpoint/generator-seed${SEED}"
  OUTDIR="generate/submission-seed${SEED}"
  LOG="$RESULTS_DIR/seed${SEED}.log"
  CSV="$RESULTS_DIR/seed${SEED}_metrics.csv"

  echo "=== seed ${SEED} ==="

  # Skip seeds that are already fully evaluated (makes reruns after interruption cheap).
  if [ -f "$CSV" ]; then
    echo "seed ${SEED}: $CSV exists, skipping"
    continue
  fi

  # --- 1. Train -------------------------------------------------------------
  if [ ! -f "$CKPT/model.pt" ]; then
    echo "seed ${SEED}: training..."
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/train_generator.py sft \
      --data data/processed/generative.csv \
      --epochs "$EPOCHS" --batch-size 128 --lr 3e-4 \
      --num-layers 6 --hidden-size 384 --num-heads 6 \
      --residual block_attnres --ffn dense --attnres-blocks 4 \
      --precision bf16 \
      --eval-every 500 --save-every 2000 \
      "${PATIENCE_FLAGS[@]+"${PATIENCE_FLAGS[@]}"}" \
      --num-workers 2 \
      --seed "$SEED" \
      --out-dir "$CKPT" \
      --log-dir "runs/seed${SEED}" \
      --no-resume >> "$LOG" 2>&1 \
      || { echo "seed ${SEED}: TRAIN FAILED (see $LOG)"; continue; }
  else
    echo "seed ${SEED}: checkpoint exists, skipping training"
  fi

  # --- 2. Generate ----------------------------------------------------------
  echo "seed ${SEED}: generating 50k library..."
  CUDA_VISIBLE_DEVICES=$GPU $PY -u -m amp_challenge_2027.generate \
    --checkpoint "$CKPT" \
    --n-sequences 50000 --top-k 100 \
    --repetition-penalty 1.3 \
    --out-dir "$OUTDIR" >> "$LOG" 2>&1 \
    || { echo "seed ${SEED}: GENERATE FAILED (see $LOG)"; continue; }

  # --- 3. Evaluate ----------------------------------------------------------
  echo "seed ${SEED}: evaluating (~8-9 min)..."
  CUDA_VISIBLE_DEVICES=$GPU $PY -u -c "
import importlib.util
spec = importlib.util.spec_from_file_location('eval_official', 'scripts/eval_official.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main(['--library', '$OUTDIR/library.fasta', '--esm-model', '$ESM_MODEL', '--device', 'cuda', '--out', '$CSV'])
" >> "$LOG" 2>&1 \
    || { echo "seed ${SEED}: EVAL FAILED (see $LOG)"; continue; }

  echo "seed ${SEED}: done -> $CSV"
done

# --- Aggregate --------------------------------------------------------------
echo ""
$PY -u - <<'PYEOF'
import glob
import re
import sys

import pandas as pd

paths = sorted(glob.glob("sweep_results/seed*_metrics.csv"))
if not paths:
    sys.exit("no metric CSVs found under sweep_results/")

rows = {}
for p in paths:
    m = re.search(r"seed(\d+)_metrics\.csv$", p)
    if not m:
        continue
    seed = int(m.group(1))
    try:
        df = pd.read_csv(p, header=[0, 1], index_col=0)
        vals = df.xs("value", axis=1, level=1).iloc[0]
    except Exception:
        try:
            df = pd.read_csv(p, index_col=0)
            vals = df.iloc[0]
        except Exception as e:
            print(f"could not parse {p}: {e}")
            continue
    rows[f"seed{seed}"] = vals

table = pd.DataFrame(rows).T
print("=== per-seed Phase-1 metrics ===")
print(table.to_string())
print()
print("=== mean / std / min / max across seeds ===")
print(table.agg(["mean", "std", "min", "max"]).T.to_string())
PYEOF

echo ""
echo "sweep complete. Reference (v2, seed 42): Conformity 0.5909  FBD 0.5130  MMD 1.1132  Precision 0.9091  Recall 0.8342  Diversity 0.8452"
