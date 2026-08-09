"""Build the curated processed datasets from raw downloads.

Thin CLI wrapper around ``amp_challenge_2027.data.build_datasets``.

Run:
    uv run --extra ml python scripts/build_datasets.py

Reads from ``data/raw/`` and writes ``data/processed/generative.csv``,
``data/processed/mic.csv``, ``data/processed/hemolysis.csv``. Missing sources
are skipped (with a note), so this can run incrementally as data arrives.
"""

from __future__ import annotations

import argparse

from amp_challenge_2027.config import RAW_DATA_DIR
from amp_challenge_2027.data import build_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated processed datasets.")
    parser.add_argument(
        "--marlys", type=str, default=str(RAW_DATA_DIR / "marlys" / "marlys.fasta"),
        help="path to MarLys FASTA",
    )
    parser.add_argument(
        "--dbaasp-peptides", type=str,
        default=str(RAW_DATA_DIR / "dbaasp" / "peptides.csv"),
    )
    parser.add_argument(
        "--dbaasp-mic", type=str,
        default=str(RAW_DATA_DIR / "dbaasp" / "activity.csv"),
    )
    args = parser.parse_args()

    counts = build_datasets(
        marlys_fasta=args.marlys,
        dbaasp_peptides_csv=args.dbaasp_peptides,
        dbaasp_mic_csv=args.dbaasp_mic,
    )
    print("\n=== built datasets ===")
    for name, n in counts.items():
        print(f"  {name}: {n} records")
    if not counts:
        print("  (no datasets built — check that data/raw/ is populated)")


if __name__ == "__main__":
    main()
