"""Build hemolysis labels from DBAASP hemolysis rows (band-based rule).

DBAASP v4 hemolysis data is (concentration, percent-lysis-band) pairs — e.g.
"50-60% Hemolysis at 6 µM" — not HC50 values (verified 2026-08-29 over a
250-peptide sample; kind histogram: 50-60% 49, 0-10% 48, 50% Cell death 22,
IC50 6, …). ``scripts/fetch_dbaasp_v4.py --hemo-refetch`` writes the
corrected schema (kind/target/value/unit/note/raw).

Label rule (defaults; all knobs configurable after inspecting --report):
  Hemolysis rows only: ``target`` matches ``--targets`` regex (default
  erythrocytes/RBC — "Cell death" rows against other cells are cytotoxicity,
  a different axis, excluded by default).
  RISKY ("active")  — any row with lysis band ≥ ``--risk-band`` (default 40%,
      midpoint parsed from the band in `kind`; IC50/HC50-style measures count
      as 50%) at concentration ≤ ``--ceiling`` µM.
  SAFE ("inactive") — a row with concentration ≥ ``--ceiling`` whose band is
      ≤ ``--safe-band`` (default 30%) — tested high and still benign.
  Sequences with neither kind of evidence are dropped (ambiguous).

Label vocabulary "active"/"inactive" (active = RISKY) keeps the generic
binary trainer compatible: ``train_reward_classifier.py --data ...
--out-dir checkpoint/reward_hemo``; direction is documented in
``score.HemoScorer``.

``--report`` prints kind/target/unit distributions WITHOUT writing anything —
run it first and adjust the knobs from the real data, never by guessing.

Run:
    uv run python scripts/fetch_dbaasp_v4.py --hemo-refetch     # corrected raw
    uv run python scripts/build_hemolysis_labels.py --report
    uv run python scripts/build_hemolysis_labels.py \
        --out data/processed/hemolysis_labels.csv
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


def band_midpoint(kind: str) -> float | None:
    """Lysis-band midpoint from a measure name: '50-60% Hemolysis' → 55,
    '0-10%' → 5, '50% Cell death' → 50, 'IC50'/'HC50' → 50 (half-lysis)."""
    lowered = kind.lower()
    if "ic50" in lowered or "hc50" in lowered or lowered.startswith("mhc"):
        return 50.0
    nums = re.findall(r"(\d+)\s*[-–~]?\s*(\d+)?\s*%", kind)
    if not nums:
        return None
    lo, hi = nums[0]
    return (int(lo) + int(hi or lo)) / 2.0


def load_peptide_sequences(peptides_csv: Path) -> dict[str, str]:
    with open(peptides_csv, newline="") as f:
        return {
            str(row["id"]): row["sequence"].strip().upper()
            for row in csv.DictReader(f)
            if row.get("sequence")
        }


def collect_observations(
    raw_csv: Path,
    peptides_csv: Path,
    *,
    targets: re.Pattern,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Counter], int]:
    """Per-sequence hemolysis observations (band_midpoint, conc_um).

    Returns (seq → [(band, conc_um)…], kind_stats for the report,
    n_unmatched_ids). Rows failing target/unit/conversion filters are only
    counted, never guessed into labels.
    """
    seqs = load_peptide_sequences(peptides_csv)
    per_seq: dict[str, list[tuple[float, float]]] = defaultdict(list)
    kind_stats: dict[str, Counter] = defaultdict(Counter)
    n_unmatched = 0
    with open(raw_csv, newline="") as f:
        for row in csv.DictReader(f):
            kind = (row.get("kind") or "").strip()
            kind_key = kind or "<empty>"
            kind_stats[kind_key]["rows"] += 1
            band = band_midpoint(kind)
            if band is None:
                kind_stats[kind_key]["no_band"] += 1
                continue
            if not targets.search((row.get("target") or "").strip()):
                kind_stats[kind_key]["other_target"] += 1
                continue
            seq = seqs.get(str(row.get("peptide_id", "")).strip())
            if not seq:
                n_unmatched += 1
                kind_stats[kind_key]["no_sequence"] += 1
                continue
            val_um, _ = concentration_to_um(
                row.get("value") or "", (row.get("unit") or "").strip(), peptide_mw(seq)
            )
            if val_um is None:
                kind_stats[kind_key]["unconvertible"] += 1
                continue
            per_seq[seq].append((band, val_um))
            kind_stats[kind_key]["converted"] += 1
    return dict(per_seq), dict(kind_stats), n_unmatched


def binarize(
    per_seq: dict[str, list[tuple[float, float]]],
    *,
    ceiling: float,
    risk_band: float,
    safe_band: float,
) -> tuple[list[dict], dict[str, int]]:
    """RISKY if any band ≥ risk_band at conc ≤ ceiling; SAFE if any row at
    conc ≥ ceiling with band ≤ safe_band; else dropped (ambiguous)."""
    rows: list[dict] = []
    stats = {"risky": 0, "safe": 0, "ambiguous_dropped": 0}
    for seq in sorted(per_seq):
        obs = per_seq[seq]
        risky = any(b >= risk_band and c <= ceiling for b, c in obs)
        safe = any(b <= safe_band and c >= ceiling for b, c in obs)
        if risky:
            worst = min(c for b, c in obs if b >= risk_band and c <= ceiling)
            rows.append({"sequence": seq, "hc50_um": f"{worst:.4f}", "label": "active"})
            stats["risky"] += 1
        elif safe:
            best = max(c for b, c in obs if b <= safe_band and c >= ceiling)
            rows.append({"sequence": seq, "hc50_um": f"{best:.4f}", "label": "inactive"})
            stats["safe"] += 1
        else:
            stats["ambiguous_dropped"] += 1
    return rows, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DBAASP hemolysis_raw (band schema) → hemolysis labels."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--peptides", type=Path, default=DEFAULT_PEPTIDES)
    parser.add_argument(
        "--targets",
        type=str,
        default=r"erythrocyte|\brbc\b|hemol",
        help="regex over the `target` column selecting HEMOLYSIS rows "
        "(default excludes non-RBC cytotoxicity)",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=128.0,
        help="µM ceiling separating 'low' from 'high' concentration",
    )
    parser.add_argument(
        "--risk-band",
        type=float,
        default=40.0,
        help="lysis %% midpoint at/below ceiling that marks RISKY",
    )
    parser.add_argument(
        "--safe-band",
        type=float,
        default=30.0,
        help="lysis %% midpoint at/above ceiling that marks SAFE",
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/hemolysis_labels.csv"))
    parser.add_argument(
        "--report", action="store_true", help="print kind/target stats and exit (writes nothing)"
    )
    args = parser.parse_args(argv)

    for p in (args.raw, args.peptides):
        if not p.exists():
            print(
                f"[hemo] missing {p}; run scripts/fetch_dbaasp_v4.py --hemo-refetch first",
                file=sys.stderr,
            )
            sys.exit(1)

    per_seq, kind_stats, n_unmatched = collect_observations(
        args.raw, args.peptides, targets=re.compile(args.targets, re.I)
    )

    print("=== kind × outcome (band parsed / target matched / converted) ===")
    for kind, stats in sorted(kind_stats.items(), key=lambda kv: -kv[1]["rows"]):
        print(f"  {kind!r}: {dict(stats)}")
    if n_unmatched:
        print(f"  (rows with no resolvable sequence: {n_unmatched})")

    if args.report:
        print(
            f"\n[hemo] report only; {len(per_seq)} sequences carry usable "
            f"hemolysis observations at current filters"
        )
        return

    rows, stats = binarize(
        per_seq, ceiling=args.ceiling, risk_band=args.risk_band, safe_band=args.safe_band
    )
    minority = min(stats["risky"], stats["safe"]) / max(len(rows), 1)
    print(
        f"\n[hemo] {len(rows)} labeled: risky={stats['risky']} safe={stats['safe']} "
        f"ambiguous_dropped={stats['ambiguous_dropped']} (minority {minority:.1%})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "hc50_um", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"[hemo] wrote {args.out}")
    if minority < 0.10:
        print(
            "[hemo] WARNING: extreme class imbalance — adjust --ceiling/--risk-band/"
            "--safe-band before training",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
