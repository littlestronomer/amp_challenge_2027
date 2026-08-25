"""Binarize DBAASP MIC data into activity labels for the reward classifier.

Policy (defaults configurable):
  ACTIVE    — measured MIC ≤ ``--active-threshold`` (default 4 µM) against any
              panel genus (min across measurements).
  INACTIVE  — every measured MIC > ``--inactive-threshold`` (default 32 µM)
              against panel genera (min across measurements still > threshold).
  otherwise — AMBIGUOUS, dropped (weak evidence either way).

The min-across-genera rule matters: a peptide potent against E. coli but weak
against P. aeruginosa is still an active AMP; using the mean would dilute it.

Writes ``data/processed/activity_labels.csv`` with columns
``sequence,label,organism`` (best = most-potent genus), prints the class
balance, and refuses to write a degenerate (<10% minority class) dataset
without --allow-imbalanced.

Run:
    uv run python scripts/build_activity_labels.py \
        --mic data/processed/mic.csv [--active-threshold 4] [--inactive-threshold 32]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import PROCESSED_DATA_DIR
from amp_challenge_2027.data import SPECIES_TO_PANEL


def load_mic_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def binarize(
    rows: list[dict],
    *,
    active_threshold: float = 4.0,
    inactive_threshold: float = 32.0,
) -> list[dict]:
    """Apply the min-across-panel-genera labeling policy. Pure function."""
    best_mic: dict[str, dict[str, float]] = defaultdict(dict)  # seq -> genus -> min MIC
    for r in rows:
        seq = r["sequence"].strip().upper()
        if not tok.is_valid_sequence(seq):
            continue
        try:
            mic = float(r["mic_value_um"])
        except (ValueError, KeyError):
            continue
        if mic <= 0:
            continue
        organism = r["target_organism"].strip()
        genus = next((g for k, g in SPECIES_TO_PANEL.items() if k.lower() in organism.lower()), "")
        if not genus:
            continue
        prev = best_mic[seq].get(genus)
        if prev is None or mic < prev:
            best_mic[seq][genus] = mic

    labels: list[dict] = []
    for seq, by_genus in sorted(best_mic.items()):
        genus, mic = min(by_genus.items(), key=lambda kv: kv[1])  # most potent genus
        if mic <= active_threshold:
            labels.append({"sequence": seq, "label": "active", "organism": genus})
        elif mic > inactive_threshold:
            labels.append({"sequence": seq, "label": "inactive", "organism": genus})
        # else: ambiguous band — no label emitted
    return labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build activity labels from DBAASP MICs.")
    parser.add_argument("--mic", type=Path, default=PROCESSED_DATA_DIR / "mic.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DATA_DIR / "activity_labels.csv")
    parser.add_argument("--active-threshold", type=float, default=4.0)
    parser.add_argument("--inactive-threshold", type=float, default=32.0)
    parser.add_argument("--allow-imbalanced", action="store_true")
    args = parser.parse_args(argv)

    if not args.mic.exists():
        print(
            f"[labels] MIC data missing: {args.mic}; run fetch_data + build_datasets first",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = load_mic_rows(args.mic)
    labels = binarize(
        rows,
        active_threshold=args.active_threshold,
        inactive_threshold=args.inactive_threshold,
    )
    n_active = sum(1 for r in labels if r["label"] == "active")
    n_inactive = len(labels) - n_active
    total_raw = len({r["sequence"].strip().upper() for r in rows})
    print(f"[labels] {total_raw} unique sequences in mic.csv")
    print(
        f"[labels] labeled: {len(labels)} (active={n_active}, inactive={n_inactive}, "
        f"ambiguous dropped={total_raw - len(labels)})"
    )

    minority = min(n_active, n_inactive) / max(len(labels), 1)
    if len(labels) == 0 or (minority < 0.10 and not args.allow_imbalanced):
        print(
            f"[labels] refusing to write: minority class {minority:.1%} < 10% "
            "(adjust thresholds or pass --allow-imbalanced)",
            file=sys.stderr,
        )
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "label", "organism"])
        w.writeheader()
        w.writerows(labels)
    print(f"[labels] wrote {args.out}")


if __name__ == "__main__":
    main()
