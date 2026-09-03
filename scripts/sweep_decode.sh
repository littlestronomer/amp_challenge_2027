#!/usr/bin/env bash
# L0 decode-parameter sweep on the adopted generator (seed44 checkpoint).
#
# Temperature / top-p / repetition-penalty were hand-set once (1.0 / 0.9 / 1.3)
# and never swept, while the seed sweep proved distribution sharpness moves
# FBD/Recall/Conformity strongly. This driver generates a full submission pair
# per decode config and scores the LIBRARY under the official protocol
# (8M embedder — A4.2 showed library rankings are stable across embedders).
#
# Usage (GPU box):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/sweep_decode.sh [checkpoint_dir]
# Defaults to checkpoint/generator (the promoted seed44). Output:
#   sweep_results/decode/<tag>/library.fasta + metrics.csv per cell
#   sweep_results/decode/summary.tsv  (all cells, sorted by FBD)
#
# ~14 min per cell (≈5 min generate + ≈9 min eval) × 9 cells ≈ 2 h.

set -euo pipefail

CKPT="${1:-checkpoint/generator}"
OUT="sweep_results/decode"
ESM="${ESM:-facebook/esm2_t6_8M_UR50D}"
DEV="${DEV:-cuda}"

mkdir -p "$OUT"
: > "$OUT/summary.tsv"
printf 'cell\ttemperature\ttop_p\trep_pen\tFBD\tMMD\tRecall\tPrecision\tConformity\tAuthenticity\tDiversity\tLength\n' >> "$OUT/summary.tsv"

run_cell () {
  local tag="$1" T="$2" P="$3" R="$4"
  echo "=== [decode-sweep] $tag : T=$T top-p=$P rep-pen=$R ==="
  uv run python -m amp_challenge_2027.generate \
    --checkpoint "$CKPT" \
    --temperature "$T" --top-p "$P" --repetition-penalty "$R" \
    --sample-top-k 50 \
    --seed 42 \
    --out-dir "$OUT/$tag" > "$OUT/$tag.generate.log" 2>&1

  uv run --extra ml --extra seqme python scripts/eval_official.py \
    --library "$OUT/$tag/library.fasta" \
    --esm-model "$ESM" --device "$DEV" \
    --out "$OUT/$tag/metrics.csv" > "$OUT/$tag.eval.log" 2>&1

# CSV layout: row0 = duplicated metric names, row1 = value/deviation labels,
# row2 = "library" + the values
python3 - "$OUT/$tag/metrics.csv" "$tag" "$T" "$P" "$R" "$OUT/summary.tsv" <<'PYEOF'
import csv, sys
path, tag, T, P, R, summary = sys.argv[1:8]
with open(path, newline="") as f:
    rows = list(csv.reader(f))
values = dict(zip(rows[0][1::2], rows[2][1::2]))
cols = ["FBD", "MMD", "Recall", "Precision", "Conformity score", "Authenticity", "Diversity", "Length"]
line = [tag, T, P, R] + [values.get(c, "") for c in cols]
with open(summary, "a") as f:
    f.write("\t".join(line) + "\n")
print("  ->", line)
PYEOF
}

# 3×3 core grid around the hand-set default (T=1.0, top-p=0.9, rep-pen=1.3)
run_cell t0.90-p0.90-r1.3  0.90 0.90 1.3
run_cell t1.00-p0.90-r1.3  1.00 0.90 1.3   # the current default (anchor)
run_cell t1.15-p0.90-r1.3  1.15 0.90 1.3
run_cell t1.00-p0.85-r1.3  1.00 0.85 1.3
run_cell t1.00-p0.95-r1.3  1.00 0.95 1.3
run_cell t1.00-p0.90-r1.2  1.00 0.90 1.2
run_cell t1.00-p0.90-r1.4  1.00 0.90 1.4
run_cell t1.15-p0.95-r1.2  1.15 0.95 1.2   # cooler+broader corner
run_cell t0.90-p0.85-r1.4  0.90 0.85 1.4   # sharper+stricter corner

echo ""
echo "=== [decode-sweep] summary (input order) ==="
column -t -s $'\t' "$OUT/summary.tsv" || cat "$OUT/summary.tsv"
