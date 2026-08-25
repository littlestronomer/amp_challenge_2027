"""Blend and rerank pre-generated candidate pools into a submission pair.

Reads one FASTA per generator checkpoint (e.g. seed44 + v2 + seed51 raw
candidates), applies the cheap validity/dedup/no-overlap filters, scores the
union with the composite scorer (activity classifier + property-conformity
density + embedding kNN precision proxy), and writes ``library.fasta`` +
``top.fasta``.

This is the offline counterpart of ``generate.py --pool``: same pipeline code,
but as a standalone CLI you can point at pools generated earlier on the GPU box
without re-running any model inference.

Run (GPU box, after exporting pool FASTAs):
    uv run --extra ml python scripts/blend_libraries.py \\
        --pools generate/pool-seed44.fasta generate/pool-v2.fasta generate/pool-seed51.fasta \\
        --out-dir generate/submission-blend3

Run (minimal env; conformity-only scoring):
    uv run python scripts/blend_libraries.py --pools generate/library.fasta --out-dir generate/rerank
"""

from __future__ import annotations

import argparse
from pathlib import Path

from amp_challenge_2027.config import (
    ANTIBACTERIAL_FASTA,
    DEFAULT_SEED,
    LIBRARY_SIZE,
    TOP_K,
)
from amp_challenge_2027.data import read_reference_set, write_fasta
from amp_challenge_2027.pipeline import (
    DEFAULT_WEIGHTS,
    build_composite_scorer,
    clean_candidates,
    load_pool_fastas,
    score_candidates,
)
from amp_challenge_2027.select import select_library_and_top


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Blend + rerank candidate pools.")
    parser.add_argument(
        "--pools",
        type=Path,
        nargs="+",
        required=True,
        help="candidate FASTAs (one per checkpoint), in blend order",
    )
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--library-size", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for activity/precision components (default cpu)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pool-cap-per-source",
        type=int,
        default=0,
        help="cap candidates per source after a seeded shuffle (0 = unlimited)",
    )
    parser.add_argument("--w-activity", type=float, default=DEFAULT_WEIGHTS["activity"])
    parser.add_argument("--w-conformity", type=float, default=DEFAULT_WEIGHTS["conformity"])
    parser.add_argument("--w-precision", type=float, default=DEFAULT_WEIGHTS["precision"])
    parser.add_argument(
        "--conformity-sample",
        type=int,
        default=12000,
        help="reference subsample for the KDE density (0 = all)",
    )
    parser.add_argument("--precision-esm", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument(
        "--novelty-candidates",
        type=int,
        default=2000,
        help="score-ranked shortlist before novelty/FPS screens (0 = unlimited)",
    )
    args = parser.parse_args(argv)

    reference_set: set[str] = set()
    if args.reference.exists():
        reference_set = read_reference_set(args.reference)
        print(f"[blend] loaded {len(reference_set)} reference sequences")
    else:
        print(f"[blend] WARNING: {args.reference} missing; skipping overlap/novelty checks")

    raw = load_pool_fastas(args.pools, cap_per_source=args.pool_cap_per_source, seed=args.seed)
    clean = clean_candidates(raw, reference_set)
    print(f"[blend] {len(clean)} clean candidates from {len(raw)} raw")

    scorer = build_composite_scorer(
        sorted(reference_set),
        w_activity=args.w_activity,
        w_conformity=args.w_conformity,
        w_precision=args.w_precision,
        device=args.device,
        conformity_sample=args.conformity_sample,
        precision_esm_model=args.precision_esm,
        seed=args.seed,
    )
    combined, _parts = score_candidates(scorer, clean)

    result = select_library_and_top(
        clean,
        reference_set=reference_set,
        scores=combined,
        top_k=args.top_k,
        library_size=args.library_size,
        seed=args.seed,
        max_novelty_candidates=args.novelty_candidates,
    )
    print(f"[blend] selection: {len(result.library)} in library, {len(result.top)} in top")

    write_fasta(result.library, args.out_dir / "library.fasta")
    write_fasta(result.top, args.out_dir / "top.fasta")
    print(f"[blend] wrote {args.out_dir / 'library.fasta'} and top.fasta")


if __name__ == "__main__":
    main()
