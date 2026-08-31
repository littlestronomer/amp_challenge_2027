"""Build hemolysis labels from the fetched DBAASP hemoliticCytotoxicActivities.

Bridges ``data/raw/dbaasp/hemolysis_raw.csv`` (peptide_id, kind, target,
value, unit — raw DBAASP activity strings) into a simple
``(sequence, hc50_um, label)`` table for training a hemolysis-risk head.

Label rule (defaults, both configurable after inspecting the real data):
  RISKY    — a hemolysis-kind reading with normalized HC50 ≤ ``--ceiling``
             (low HC50 = lyses red cells at low concentration).
  SAFE     — every hemolysis-kind reading for that sequence is > ``--ceiling``.
  dropped  — unparseable values / unsupported units / non-hemolysis kinds.

The label vocabulary is "active"/"inactive" (active = RISKY) so the generic
binary trainer (``train_reward_classifier.py --data ... --out-dir
checkpoint/reward_hemo``) consumes it unchanged; the direction convention is
documented in ``score.HemoScorer``.

``--report`` prints kind/value/unit distributions WITHOUT writing anything —
run it first and adjust ``--kinds``/``--ceiling`` from what the data actually
contains (never guess label rules).

Run:
    uv run python scripts/build_hemolysis_labels.py --report
    uv run python scripts/build_hemolysis_labels.py \
        --kinds 'hc50|hemolys' --ceiling 128 --out data/processed/hemolysis_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from build_ranking_labels import peptide_mw
from fetch_dbaasp_v4 import concentration_to_um

DEFAULT_RAW = Path("data/raw/dbaasp/hemolysis_raw.csv")
DEFAULT_PEPTIDES = Path("data/raw/dbaasp/peptides.csv")


def load_peptide_sequences(peptides_csv: Path) -> dict[str, str]:
    """id (numeric or string) → sequence."""
    with open(peptides_csv, newline="") as f:
        return {
            str(row["id"]): row["sequence"].strip().upper()
            for row in csv.DictReader(f)
            if row.get("sequence")
        }


def parse_hemolysis_rows(
    raw_csv: Path, peptides_csv: Path, *, kinds: re.Pattern
) -> tuple[dict[str, list[float]], dict[str, Counter], int]:
    """Aggregate per-sequence HC50 µM values for hemolysis-kind rows.

    Returns (seq → [hc50_um...], kind_stats, n_unmatched_ids). ``kind_stats``
    maps kind → Counter of outcomes (converted / unparseable / bad_unit /
    no_sequence) for the ``--report`` view.
    """
    seqs = load_peptide_sequences(peptides_csv)
    per_seq: dict[str, list[float]] = defaultdict(list)
    kind_stats: dict[str, Counter] = defaultdict(Counter)
    n_unmatched = 0
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            kind = (row.get("kind") or "").strip()
            kind_stats[kind]["rows"] += 1
            if not kinds.search(kind):
                continue
            seq = seqs.get(str(row.get("peptide_id", "")).strip())
            if not seq:
                n_unmatched += 1
                kind_stats[kind]["no_sequence"] += 1
                continue
            unit = (row.get("unit") or "").strip()
            value = (row.get("value") or "").strip()
            val_um, _ = concentration_to_um(value, unit, peptide_mw(seq))
            if val_um is None:
                reason = "bad_unit" if value else "unparseable"
                kind_stats[kind][reason] += 1
                continue
            per_seq[seq].append(val_um)
            kind_stats[kind]["converted"] += 1
    return dict(per_seq), dict(kind_stats), n_unmatched


def binarize(per_seq: dict[str, list[float]], *, ceiling: float) -> list[dict]:
    """RISKY if min HC50 ≤ ceiling (label 'active'), else SAFE ('inactive')."""
    rows = []
    for seq in sorted(per_seq):
        hc50 = min(per_seq[seq])
        label = "active" if hc50 <= ceiling else "inactive"
        rows.append({"sequence": seq, "hc50_um": f"{hc50:.4f}", "label": label})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DBAASP hemolysis_raw → hemolysis labels.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--peptides", type=Path, default=DEFAULT_PEPTIDES)
    parser.add_argument(
        "--kinds",
        type=str,
        default=r"hc50|hemolys",
        help="regex over the `kind` column selecting hemolysis evidence",
    )
    parser.add_argument(
        "--ceiling", type=float, default=128.0, help="HC50 µM ceiling; min HC50 ≤ ceiling → risky"
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/hemolysis_labels.csv"))
    parser.add_argument(
        "--report",
        action="store_true",
        help="print kind/unit/value stats and exit (writes nothing)",
    )
    args = parser.parse_args(argv)

    for p in (args.raw, args.peptides):
        if not p.exists():
            print(
                f"[hemo] missing {p}; fetch DBAASP first (scripts/fetch_dbaasp_v4.py)",
                file=sys.stderr,
            )
            sys.exit(1)

    kinds = re.compile(args.kinds, re.I)
    per_seq, kind_stats, n_unmatched = parse_hemolysis_rows(args.raw, args.peptides, kinds=kinds)

    print("=== kind × outcome (ALL kinds; only matched ones feed labels) ===")
    for kind, stats in sorted(kind_stats.items(), key=lambda kv: -kv[1]["rows"]):
        flag = "← matched" if kinds.search(kind) else ""
        print(f"  {kind or '<empty>'!r}: {dict(stats)} {flag}")
    if n_unmatched:
        print(f"  (rows with no resolvable sequence: {n_unmatched})")

    if args.report:
        converted = sum(s.get("converted", 0) for s in kind_stats.values())
        print(
            f"\n[hemo] report only; would binarize {len(per_seq)} sequences "
            f"({converted} converted readings) at ceiling {args.ceiling:g} µM"
        )
        return

    rows = binarize(per_seq, ceiling=args.ceiling)
    n_risky = sum(1 for r in rows if r["label"] == "active")
    minority = min(n_risky, len(rows) - n_risky) / max(len(rows), 1)
    print(
        f"\n[hemo] {len(rows)} labeled sequences: risky={n_risky}, safe={len(rows) - n_risky} "
        f"(minority {minority:.1%})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "hc50_um", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"[hemo] wrote {args.out}")
    if minority < 0.10:
        print(
            "[hemo] WARNING: extreme class imbalance — adjust --kinds/--ceiling before training",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
