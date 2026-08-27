"""Rebuild the generative corpus from raw sources into a SEPARATE output path.

The seed-44 checkpoint lineage (the confirmed v2 recipe) trains on exactly
``data/processed/generative.csv``, and ``data.build_datasets()`` overwrites
that file unconditionally whenever the MarLys FASTA is present. One careless
re-run of the dataset build silently replaces the canonical corpus and seed-44
reproduction can no longer be checked against byte-identical training data.

This wrapper makes a rebuild safe by construction:

  1. Snapshot SHA-256 of every file under ``--watch-dir``
     (default: the directory holding the canonical corpus).
  2. Rebuild the base curated sets from raw inputs with the SAME parse+curate
     code that produced the canonical corpus, but into ``--out-dir``
     (default: ``data/processed/rebuild/``). When both exist, prints a
     byte-equality verdict between the rebuilt and canonical ``generative.csv``
     — a free check for upstream-data or curation drift.
  3. Merge base + extra raw FASTAs + DBAASP sequences into
     ``{out-dir}/generative_expanded.csv``, reusing the merge rules, loaders,
     and column format of ``scripts/build_expanded_generative.py`` so the two
     builders cannot drift apart.
  4. Re-snapshot. Every file that existed at step 1 must still be
     byte-identical; any change aborts with exit code 3.

No RNG is used anywhere — repeated runs over identical inputs are
byte-identical. Heavy training deps stay optional (numpy only).

Run:
    uv run python scripts/rebuild_corpus.py
    # canonical corpus untouched: data/processed/generative.csv frozen;
    # outputs land in data/processed/rebuild/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from amp_challenge_2027.config import (
    ANTIBACTERIAL_FASTA,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from amp_challenge_2027.data import read_reference_set

INTEGRITY_EXIT_CODE = 3


# ---------------------------------------------------------------------------
# Sibling-builder reuse (merge rules live in ONE place)
# ---------------------------------------------------------------------------


def _load_builder():
    """Import ``scripts/build_expanded_generative.py`` regardless of how we run."""
    try:
        import build_expanded_generative as mod

        return mod
    except ImportError:
        pass
    import importlib.util

    sibling = Path(__file__).resolve().parent / "build_expanded_generative.py"
    spec = importlib.util.spec_from_file_location("build_expanded_generative", sibling)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Integrity guard
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_snapshot(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root``, keyed by relative path."""
    snap: dict[str, str] = {}
    if not root.exists():
        return snap
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[str(p.relative_to(root))] = _sha256(p)
    return snap


def integrity_violations(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Pre-existing files whose hash changed (new files are not violations)."""
    return sorted(name for name, digest in before.items() if after.get(name) != digest)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _rebuild_base(args) -> dict[str, int]:
    """Curated base sets from raw via the SAME builder code as the original."""
    from amp_challenge_2027.data import build_datasets

    return build_datasets(
        marlys_fasta=args.marlys,
        dbaasp_peptides_csv=args.dbaasp_peptides,
        dbaasp_mic_csv=args.dbaasp_mic,
        out_dir=args.out_dir,
    )


def _base_verdict(args) -> str:
    rebuilt = args.out_dir / "generative.csv"
    if not args.canonical_generative.exists() or not rebuilt.exists():
        return (
            f"skipped ({'canonical' if not args.canonical_generative.exists() else 'rebuilt'}"
            " missing)"
        )
    verdict = (
        "IDENTICAL — no upstream/curation drift"
        if _sha256(rebuilt) == _sha256(args.canonical_generative)
        else "DIFFERS — canonical corpus was built from other inputs or rules "
        "(informational only; nothing was overwritten)"
    )
    return verdict


def _merge_expanded(args):
    """Merged corpus into {out_dir}/generative_expanded.csv (sibling semantics)."""
    mod = _load_builder()

    reference: set[str] = set()
    if args.antibacterial.exists() and not args.keep_reference_overlap:
        reference = read_reference_set(args.antibacterial)

    # MarLys pool: prefer the canonical corpus when present so provenance stays
    # identical to past runs; fall back to the rebuild-sandbox base otherwise.
    if args.canonical_generative.exists():
        marlys_csv = args.canonical_generative
        print(f"[rebuild] MarLys pool source: {marlys_csv} (canonical)")
    else:
        marlys_csv = args.out_dir / "generative.csv"
        print(f"[rebuild] MarLys pool source: {marlys_csv} (rebuilt fallback)")
    pools = [
        ("marlys-curated", mod._load_curated_generative(marlys_csv)),
        ("raw-fastas", mod._load_raw_fastas(args.raw_dir)),
    ]

    mic_dirs = [Path(args.processed_dir), args.out_dir]
    dbaasp_done = False
    for d in mic_dirs:
        if (d / "mic.csv").exists():
            pools.append(("dbaasp", mod._load_dbaasp_sequences(d, args.raw_dir)))
            dbaasp_done = True
            break
    if not dbaasp_done:
        print("[rebuild] note: no mic.csv found anywhere; DBAASP pool skipped")

    prio = {name: i for i, name in enumerate(mod.SOURCE_PRIORITY)}
    flat = [pair for _name, pool in pools for pair in pool]
    flat.sort(key=lambda p: prio.get(p[1], len(prio)))  # stable: intra-source order kept

    merged, stats = mod.curate_and_merge([("all", flat)], exclude_reference=reference)

    out_path = args.out_dir / "generative_expanded.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "source_id", "charge", "activity", "source_dbs"])
        for i, (seq, src) in enumerate(merged):
            w.writerow([seq, f"{src}_{i:06d}", "", "", src])

    print("\n=== merge summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    per_source: dict[str, int] = {}
    for _seq, src in merged:
        per_source[src] = per_source.get(src, 0) + 1
    print("  per-source kept:")
    for src, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"    {src}: {n}")
    seqs_only = [seq for seq, _ in merged]
    if seqs_only:
        print(f"\n  corpus properties: {mod._property_summary(seqs_only)}")
    return merged, stats, out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Rebuild corpora into a separate sandbox path.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    # Source paths default to None and resolve under --raw-dir after parsing,
    # so repointing --raw-dir moves all sources at once.
    parser.add_argument("--marlys", type=Path, default=None)
    parser.add_argument("--dbaasp-peptides", type=Path, default=None)
    parser.add_argument("--dbaasp-mic", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "rebuild",
        help="sandbox output directory (never a pre-existing processed file lives here)",
    )
    parser.add_argument(
        "--canonical-generative",
        type=Path,
        default=PROCESSED_DATA_DIR / "generative.csv",
        help="frozen reference corpus (seed-44 lineage); only ever READ",
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=None,
        help="directory whose existing files must stay byte-identical "
        "(default: the canonical corpus's parent)",
    )
    parser.add_argument(
        "--keep-reference-overlap",
        action="store_true",
        help="do NOT drop sequences identical to the antibacterial reference (not recommended)",
    )
    parser.add_argument(
        "--antibacterial",
        type=Path,
        default=ANTIBACTERIAL_FASTA,
        help="reference FASTA whose exact members are excluded from the merged corpus",
    )
    parser.add_argument("--skip-base", action="store_true", help="skip the base-corpus rebuild")
    args = parser.parse_args(argv)

    # Resolve per-source paths under --raw-dir when not given explicitly.
    if args.marlys is None:
        args.marlys = args.raw_dir / "marlys" / "marlys.fasta"
    if args.dbaasp_peptides is None:
        args.dbaasp_peptides = args.raw_dir / "dbaasp" / "peptides.csv"
    if args.dbaasp_mic is None:
        args.dbaasp_mic = args.raw_dir / "dbaasp" / "activity.csv"

    watch = args.watch_dir or args.canonical_generative.parent
    before = dir_snapshot(watch)
    print(f"[rebuild] integrity snapshot: {len(before)} file(s) under {watch}")

    base_counts: dict[str, int] = {}
    if not args.skip_base:
        print("\n--- step 1: rebuild base sets (sandbox path) ---")
        base_counts = _rebuild_base(args)
        for name, n in base_counts.items():
            print(f"  {name}: {n} records")
        verdict = _base_verdict(args)
        print(f"  canonical-vs-rebuilt generative.csv: {verdict}")
    else:
        print("\n--- step 1 skipped (--skip-base) ---")

    print("\n--- step 2: merged expanded corpus ---")
    merged, stats, out_path = _merge_expanded(args)

    violations = integrity_violations(before, dir_snapshot(watch))
    status = "OK — pre-existing files byte-identical"
    for v in violations:
        status = "VIOLATION — pre-existing files were modified:"
        print(f"[rebuild] INTEGRITY {status} {v}")

    print("\n=== result ===")
    print(f"  outputs      : {args.out_dir}/")
    if not args.skip_base:
        print("                 generative.csv (+ mic.csv / hemolysis.csv when sources exist)")
    print(f"                 generative_expanded.csv ({len(merged)} sequences)")
    print(f"  integrity    : {status}")
    print(f"\n[rebuild] canonical corpus untouched: {args.canonical_generative} (read-only use)")

    summary = {
        "base_counts": base_counts,
        "merge_stats": stats,
        "expanded_size": len(merged),
        "expanded_path": str(out_path),
        "violations": violations,
    }
    if violations:
        sys.exit(INTEGRITY_EXIT_CODE)
    return summary


if __name__ == "__main__":
    main()
