"""Build an expanded generative corpus from every raw source available.

Merges, deduplicates, and curates peptide sequences for generator pretraining:

  1. The existing curated MarLys corpus (``data/processed/generative.csv``).
  2. Every FASTA under ``data/raw/<source>/`` (DRAMP/APD/LAMP exports you
     dropped in manually — their download pages are click-mediated, so the
     fetcher does not guess direct URLs).
  3. DBAASP sequences already on disk (from ``peptides.csv`` / ``activity.csv``
     via ``mic.csv``) — thousands of unique activity-annotated peptides at zero
     extra download cost.

Curation: standard amino acids only, length 8–50, global exact-dedup (first
source wins in a fixed priority order), and — by default — removal of anything
exactly matching ``data/antibacterial.fasta`` so the model never memorizes a
sequence the submission is forbidden to emit.

Writes ``data/processed/generative_expanded.csv`` with provenance columns and
prints per-source counts plus charge/hydrophobicity summaries before/after.

Run:
    uv run --extra ml python scripts/build_expanded_generative.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import (
    ANTIBACTERIAL_FASTA,
    MAX_LENGTH,
    MIN_LENGTH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from amp_challenge_2027.data import iter_fasta, read_reference_set

# Priority order when several sources contain the same sequence; earlier wins.
SOURCE_PRIORITY = ["marlys", "dramp", "apd", "lamp", "dbamp", "dbaasp"]


def _load_curated_generative(path: Path) -> list[tuple[str, str]]:
    """(sequence, source) pairs from the existing processed corpus."""
    out: list[tuple[str, str]] = []
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq = row["sequence"].strip().upper()
            if tok.is_valid_sequence(seq):
                out.append((seq, "marlys"))
    return out


def _load_raw_fastas(raw_dir: Path) -> list[tuple[str, str]]:
    """Every FASTA under data/raw/<source>/**, tagged by its directory name."""
    out: list[tuple[str, str]] = []
    if not raw_dir.exists():
        return out
    for fasta in sorted(raw_dir.rglob("*.fasta")) + sorted(raw_dir.rglob("*.fa")):
        source = fasta.parent.name.lower()
        n0 = len(out)
        for _header, seq in iter_fasta(fasta):
            if tok.is_valid_sequence(seq):
                out.append((seq, source))
        print(f"[expand] {fasta}: {len(out) - n0} valid sequences")
    return out


def _load_dbaasp_sequences(processed_dir: Path, raw_dir: Path) -> list[tuple[str, str]]:
    """Unique DBAASP sequences from mic.csv (already built) or peptides.csv."""
    seqs: set[str] = set()
    mic_csv = processed_dir / "mic.csv"
    if mic_csv.exists():
        with open(mic_csv, newline="") as f:
            for row in csv.DictReader(f):
                s = row["sequence"].strip().upper()
                if tok.is_valid_sequence(s):
                    seqs.add(s)
    pep_csv = raw_dir / "dbaasp" / "peptides.csv"
    if pep_csv.exists():
        with open(pep_csv, newline="") as f:
            for row in csv.DictReader(f):
                s = (row.get("sequence") or row.get("SEQUENCE") or "").strip().upper()
                if tok.is_valid_sequence(s):
                    seqs.add(s)
    return [(s, "dbaasp") for s in sorted(seqs)]


def curate_and_merge(
    pools: list[tuple[str, list[tuple[str, str]]]],
    *,
    exclude_reference: set[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Priority-order merge with curation. Returns ((seq, source), stats)."""
    stats = {"raw": 0, "invalid": 0, "duplicate": 0, "reference_overlap": 0, "kept": 0}
    seen: set[str] = set()
    merged: list[tuple[str, str]] = []
    for _name, pool in pools:
        for seq, source in pool:
            stats["raw"] += 1
            if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH) or not tok.is_valid_sequence(seq):
                stats["invalid"] += 1
                continue
            if seq in seen:
                stats["duplicate"] += 1
                continue
            if exclude_reference and seq in exclude_reference:
                stats["reference_overlap"] += 1
                continue
            seen.add(seq)
            merged.append((seq, source))
            stats["kept"] += 1
    return merged, stats


def _property_summary(seqs: list[str]) -> str:
    from amp_challenge_2027.conditioning import charge_modlamp
    from amp_challenge_2027.props import mean_hydrophobicity

    charges = [charge_modlamp(s) for s in seqs]
    hydros = [mean_hydrophobicity(s) for s in seqs]
    lengths = [len(s) for s in seqs]
    return (
        f"n={len(seqs)} length μ={np.mean(lengths):.1f}±{np.std(lengths):.1f} "
        f"charge μ={np.mean(charges):+.2f}±{np.std(charges):.2f} "
        f"KD μ={np.mean(hydros):+.2f}±{np.std(hydros):.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the expanded generative corpus.")
    parser.add_argument(
        "--generative",
        type=Path,
        default=PROCESSED_DATA_DIR / "generative.csv",
        help="existing curated corpus (MarLys)",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DATA_DIR,
        help="directory holding mic.csv (DBAASP sequences)",
    )
    parser.add_argument("--out", type=Path, default=PROCESSED_DATA_DIR / "generative_expanded.csv")
    parser.add_argument(
        "--keep-reference-overlap",
        action="store_true",
        help="do NOT drop sequences identical to data/antibacterial.fasta (not recommended)",
    )
    args = parser.parse_args(argv)

    reference: set[str] = set()
    if ANTIBACTERIAL_FASTA.exists() and not args.keep_reference_overlap:
        reference = read_reference_set(ANTIBACTERIAL_FASTA)

    pools: list[tuple[str, list[tuple[str, str]]]] = [
        ("marlys-curated", _load_curated_generative(args.generative)),
        ("raw-fastas", _load_raw_fastas(args.raw_dir)),
        ("dbaasp", _load_dbaasp_sequences(args.processed_dir, args.raw_dir)),
    ]
    # Stable priority: sort sources within each pool by SOURCE_PRIORITY index.
    prio = {name: i for i, name in enumerate(SOURCE_PRIORITY)}
    flat: list[tuple[str, str]] = [pair for _n, pool in pools for pair in pool]
    flat.sort(key=lambda p: prio.get(p[1], len(prio)))  # stable: keeps intra-source order

    merged, stats = curate_and_merge([("all", flat)], exclude_reference=reference)

    print("\n=== expansion summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    per_source: dict[str, int] = {}
    for _seq, src in merged:
        per_source[src] = per_source.get(src, 0) + 1
    print("  per-source kept:")
    for src, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"    {src}: {n}")

    seqs_only = [s for s, _ in merged]
    print(f"\n  corpus properties: {_property_summary(seqs_only)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "source_id", "charge", "activity", "source_dbs"])
        for i, (seq, src) in enumerate(merged):
            w.writerow([seq, f"{src}_{i:06d}", "", "", src])
    print(f"\n[expand] wrote {len(merged)} sequences → {args.out}")

    base = sum(1 for _s, src in merged if src == "marlys")
    extra = len(merged) - base
    print(
        f"[expand] MarLys baseline {base} + {extra} new → {len(merged)} total "
        f"(×{len(merged) / max(base, 1):.2f})"
    )
    if extra == 0:
        print(
            "[expand] NOTE: nothing new was added. Drop DRAMP/APD/LAMP FASTA exports "
            f"under {args.raw_dir}/<source>/ (e.g. data/raw/dramp/general.fasta) "
            "and re-run; see docs/RUNBOOK_DATA_EXPANSION.md.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
