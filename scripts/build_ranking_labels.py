"""Build the EXPANDED ranking-label datasets from DBAASP (+ DRAMP memberships).

The legacy ``build_activity_labels.py`` collapses everything to ~2k rows: one
row per sequence (min across panel genera) with a narrow 4–32 µM decision band
that discards the middle. This builder exploits ``mic.csv`` fully WITHOUT
touching any existing artifact:

  1. ``activity_labels_full.csv`` — one row per (sequence × measured panel
     genus): min MIC in µM, a banded classification, and a binary label at
     ``MIC_SUCCESS_THRESHOLD_UM`` (blank = ambiguous → mask at training time).
  2. ``activity_labels_gram.csv`` — per-sequence Gram-type votes derived from
     those rows, unioned with DRAMP antibacterial/Gram-split FASTA membership
     as source-tagged weak positives; numeric DBAASP evidence always wins on
     conflict.

Unit honesty: rows parsed as µg/mL are converted via deterministic average-
residue peptide MW instead of being misread as µM; other unknown units are
dropped and counted.

Outputs land beside (never overwrite) existing processed files. Fully
deterministic given inputs.

Run:
    uv run python scripts/build_ranking_labels.py \
        [--mic data/processed/mic.csv] [--dramp-dir data/raw/dramp] \
        [--out-dir data/processed] [--potent 4 --success 16 --inactive 32]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from build_activity_labels import load_mic_rows  # sibling reuse via conftest/sys.path

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import (
    MIC_SUCCESS_THRESHOLD_UM,
    PANEL_GENUS_TO_GRAM,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from amp_challenge_2027.data import SPECIES_TO_PANEL, iter_fasta

# Average residue masses (Da) + water, for µg/mL → µM conversion.
RESIDUE_AVG_MW: dict[str, float] = {
    "A": 71.0788,
    "C": 103.1388,
    "D": 115.0886,
    "E": 129.1155,
    "F": 147.1766,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "K": 128.1741,
    "L": 113.1594,
    "M": 101.1051,
    "N": 114.1038,
    "P": 97.1167,
    "Q": 128.1307,
    "R": 156.1875,
    "S": 87.0782,
    "T": 101.1051,
    "V": 99.1326,
    "W": 186.2132,
    "Y": 163.1760,
}
_WATER_MW = 18.01528


def peptide_mw(seq: str) -> float:
    # Real DRAMP/DBAASP sequences sometimes carry nonstandard residues
    # (Z, X, B…); MW is computed over the standard subset — an approximation,
    # but these are fuzzy measurements anyway.
    standard = "".join(ch for ch in seq if ch in RESIDUE_AVG_MW) or "A"
    return sum(RESIDUE_AVG_MW[aa] for aa in standard) + _WATER_MW


def mic_to_um(value: float, unit: str, seq: str) -> tuple[float | None, str]:
    """Normalize a parsed mic row to µM. Returns (value_or_None, note)."""
    unit = unit.strip().lower()
    if unit in ("um", ""):
        return value, ("assumed-uM" if unit == "" else "")
    if unit == "ug_ml":
        return value * 1000.0 / peptide_mw(seq), ""
    return None, f"unsupported unit '{unit}'"


def classify_band(mic_um: float, *, potent: float, success: float, inactive: float) -> str:
    """Banded classification; boundaries inclusive."""
    if mic_um <= potent:
        return "potent"
    if mic_um <= success:
        return "active-band"
    if mic_um <= inactive:
        return "weak"
    return "inactive"


# ---------------------------------------------------------------------------
# Aggregation over mic.csv
# ---------------------------------------------------------------------------


def aggregate_genus_mics(rows: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """min MIC per (sequence, panel genus), with honest skip accounting."""
    best: dict[str, dict[str, float]] = defaultdict(dict)
    stats = {
        "rows": len(rows),
        "invalid_sequence": 0,
        "bad_value": 0,
        "unmapped_organism": 0,
        "unsupported_unit": 0,
        "unit_assumed_uM": 0,
        "ug_ml_converted": 0,
    }
    for r in rows:
        seq = r["sequence"].strip().upper()
        if not tok.is_valid_sequence(seq):
            stats["invalid_sequence"] += 1
            continue
        try:
            raw_val = float(r["mic_value_um"])
        except (ValueError, KeyError):
            stats["bad_value"] += 1
            continue
        if raw_val <= 0:
            stats["bad_value"] += 1
            continue
        organism = r["target_organism"].strip()
        genus = next((g for k, g in SPECIES_TO_PANEL.items() if k.lower() in organism.lower()), "")
        if not genus:
            stats["unmapped_organism"] += 1
            continue
        val_um, note = mic_to_um(raw_val, r.get("unit", ""), seq)
        if val_um is None:
            stats["unsupported_unit"] += 1
            continue
        if note == "assumed-uM":
            stats["unit_assumed_uM"] += 1
        elif note == "":
            pass
        else:
            stats["ug_ml_converted"] += 1
        prev = best[seq].get(genus)
        if prev is None or val_um < prev:
            best[seq][genus] = val_um
    return dict(best), stats


def build_full_rows(
    genus_mics: dict[str, dict[str, float]], *, potent: float, success: float, inactive: float
) -> list[dict]:
    rows: list[dict] = []
    for seq in sorted(genus_mics):
        for genus, mic in sorted(genus_mics[seq].items()):
            band = classify_band(mic, potent=potent, success=success, inactive=inactive)
            label = ""
            if band in ("potent", "active-band"):
                label = "active"
            elif band == "inactive":
                label = "inactive"
            rows.append(
                {
                    "sequence": seq,
                    "organism": genus,
                    "mic_um": f"{mic:.4f}",
                    "band": band,
                    "label": label,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Gram-type votes (DBAASP numeric evidence first)
# ---------------------------------------------------------------------------


def gram_votes_from_full(full_rows: list[dict], *, inactive: float) -> list[tuple[str, str, str]]:
    """Per-(sequence, Gram type) votes: any active ⇒ 'active'; all genera
    measured for that side > ``inactive`` threshold ⇒ 'inactive'; else no vote.
    Returns (sequence, gram, label) sorted deterministically."""
    by_seq_side: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in full_rows:
        side = PANEL_GENUS_TO_GRAM.get(r["organism"])
        if not side:
            continue
        by_seq_side[(r["sequence"], side)].append(float(r["mic_um"]))
    votes: list[tuple[str, str, str]] = []
    for seq, side in sorted(by_seq_side):
        mics = by_seq_side[(seq, side)]
        if min(mics) <= MIC_SUCCESS_THRESHOLD_UM:
            votes.append((seq, side, "active"))
        elif min(mics) > inactive:
            votes.append((seq, side, "inactive"))
    return votes


def load_dramp_memberships(dramp_dir: Path) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Weak-positive memberships from DRAMP split FASTAs.

    Only sides (anti-Gram-positive / anti-Gram-negative) contribute labels;
    general / antibacterial files stay generative-corpus material only and are
    just counted here. Missing files are skipped with a note.
    """
    file_to_side = {
        "anti_gram_positive.fasta": ("positive", "anti_gram_positive.fasta"),
        "anti_gram_negative.fasta": ("negative", "anti_gram_negative.fasta"),
    }
    counts = {"memberships": 0}
    out: list[tuple[str, str, str]] = []
    for fname, (side, display) in file_to_side.items():
        path = dramp_dir / fname
        if not path.exists():
            print(f"[labels] DRAMP {display}: not present under {dramp_dir}; skipped")
            continue
        n_valid = n_total = 0
        seen_this_file: set[str] = set()
        for _header, seq_raw in iter_fasta(path):
            n_total += 1
            seq = seq_raw.strip().upper()
            if not tok.is_valid_sequence(seq) or seq in seen_this_file:
                continue
            seen_this_file.add(seq)
            out.append((seq, side, "active"))
            n_valid += 1
        print(f"[labels] DRAMP {display}: {n_valid} valid members ({n_total} records)")
        counts["memberships"] += n_valid
    return out, counts


def merge_gram_labels(
    dbaasp_votes: list[tuple[str, str, str]],
    dramp_members: list[tuple[str, str, str]],
) -> tuple[list[dict], int]:
    """Union votes; DBAASP numeric evidence wins; conflicts counted."""
    merged: dict[tuple[str, str], str] = {}
    sources: dict[tuple[str, str], str] = {}
    for seq, side, label in dramp_members:
        key = (seq, side)
        merged[key] = label
        sources[key] = "dramp"
    conflicts = 0
    for seq, side, label in dbaasp_votes:
        key = (seq, side)
        prev = merged.get(key)
        if prev is not None and prev != label:
            conflicts += 1
        merged[key] = label
        sources[key] = "dbaasp"
    rows = [
        {"sequence": key[0], "gram": key[1], "label": merged[key], "source": sources[key]}
        for key in sorted(merged)
    ]
    return rows, conflicts


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------


def print_coverage(full_rows: list[dict], mdr_note: str = "") -> None:
    per_genus: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seqs_by_label: dict[str, set] = defaultdict(set)
    for r in full_rows:
        per_genus[r["organism"]][r["band"]] += 1
        if r["label"]:
            seqs_by_label[r["label"]].add(r["sequence"])
    bands = ["potent", "active-band", "weak", "inactive"]
    print("\n=== per-genus coverage (rows by band) ===")
    header = "  genus".ljust(18) + "".join(b.ljust(14) for b in bands)
    print(header + "total")
    total_rows = 0
    for genus in sorted(per_genus):
        cells = [per_genus[genus][b] for b in bands]
        total_rows += sum(cells)
        print(("  " + genus).ljust(18) + "".join(str(c).ljust(14) for c in cells) + str(sum(cells)))
    print(f"  TOTAL rows: {total_rows}\n")
    print("=== unique labeled sequences ===")
    for lbl in ("active", "inactive"):
        print(f"  {lbl}: {len(seqs_by_label[lbl])}")
    masked = {r["sequence"] for r in full_rows} - (
        seqs_by_label["active"] | seqs_by_label["inactive"]
    )
    print(f"  masked/ambiguous (per-genus blank labels): {len(masked)}")
    if mdr_note:
        print(f"  MDR note: {mdr_note}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build expanded ranking-label datasets.")
    parser.add_argument(
        "--mic",
        type=Path,
        action="append",
        default=None,
        help="mic.csv to aggregate (repeatable, e.g. DBAASP + mic_dramp.csv); "
        "default: data/processed/mic.csv",
    )
    parser.add_argument("--dramp-dir", type=Path, default=RAW_DATA_DIR / "dramp")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--potent", type=float, default=4.0)
    parser.add_argument("--success", type=float, default=MIC_SUCCESS_THRESHOLD_UM)
    parser.add_argument("--inactive", type=float, default=32.0)
    args = parser.parse_args(argv)

    mic_paths = args.mic or [PROCESSED_DATA_DIR / "mic.csv"]
    for p in mic_paths:
        if not p.exists():
            print(
                f"[labels] MIC data missing: {p}; run fetch_data + build_datasets first",
                file=sys.stderr,
            )
            sys.exit(1)

    rows: list[dict] = []
    for p in mic_paths:
        part = load_mic_rows(p)
        print(f"[labels] {p}: {len(part)} rows")
        rows.extend(part)
    genus_mics, agg_stats = aggregate_genus_mics(rows)
    full_rows = build_full_rows(
        genus_mics, potent=args.potent, success=args.success, inactive=args.inactive
    )

    votes = gram_votes_from_full(full_rows, inactive=args.inactive)
    dramp_members, member_counts = load_dramp_memberships(args.dramp_dir)
    gram_rows, conflicts = merge_gram_labels(votes, dramp_members)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.out_dir / "activity_labels_full.csv"
    with open(full_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "organism", "mic_um", "band", "label"])
        w.writeheader()
        w.writerows(full_rows)
    gram_path = args.out_dir / "activity_labels_gram.csv"
    with open(gram_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "gram", "label", "source"])
        w.writeheader()
        w.writerows(gram_rows)

    print("\n[labels] aggregation stats: " + ", ".join(f"{k}={v}" for k, v in agg_stats.items()))
    print_coverage(full_rows)
    n_dbaasp = sum(1 for r in gram_rows if r["source"] == "dbaasp")
    n_dramp = len(gram_rows) - n_dbaasp
    print(
        f"\n[labels] gram rows: {len(gram_rows)} (dbaasp={n_dbaasp}, dramp={n_dramp}, "
        f"conflicts resolved toward dbaasp={conflicts}); "
        f"membership additions={member_counts['memberships']}"
    )

    legacy = PROCESSED_DATA_DIR / "activity_labels.csv"
    if legacy.exists():
        with open(legacy, newline="") as f:
            legacy_seqs = {r["sequence"].strip().upper() for r in csv.DictReader(f)}
        new_seqs = {r["sequence"] for r in full_rows if r["label"]}
        gained = len(new_seqs - legacy_seqs)
        print(
            f"[labels] vs legacy activity_labels.csv ({len(legacy_seqs)} rows): "
            f"+{gained} newly covered sequences"
        )
    print(f"[labels] wrote {full_path}\n[labels] wrote {gram_path}")


if __name__ == "__main__":
    main()
