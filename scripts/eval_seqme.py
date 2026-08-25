"""Score a generated library with seqme — mirrors the Phase-1 evaluation protocol.

Run:
    uv run --extra seqme python scripts/eval_seqme.py --library generate/library.fasta

seqme computes the four Phase-1 metric families (sequence-level, embedding
distributional, property conformity, surrogate hit-rate) using ESM-2 embeddings.
This script wraps a focused subset so you can iterate locally during development
without running the full organizer pipeline.

Requires the ``seqme[esm2]`` extra: ``uv sync --extra seqme``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA
from amp_challenge_2027.data import iter_fasta


def load_fasta(path: Path) -> list[str]:
    return [seq for _, seq in iter_fasta(path)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a peptide library with seqme.")
    parser.add_argument("--library", type=Path, required=True, help="FASTA to evaluate")
    parser.add_argument(
        "--reference", type=Path, default=ANTIBACTERIAL_FASTA,
        help="reference FASTA of known AMPs (default: data/antibacterial.fasta)",
    )
    parser.add_argument(
        "--esm-model", type=str, default="facebook/esm2_t33_650M_UR50D",
        help="ESM-2 model id for embedding metrics",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=None, help="optional CSV to write metrics")
    args = parser.parse_args(argv)

    try:
        import seqme as sm
    except ImportError:
        print(
            "seqme is not installed. Install with: uv sync --extra seqme",
            file=sys.stderr,
        )
        sys.exit(1)

    generated_seqs = load_fasta(args.library)
    reference = load_fasta(args.reference)
    print(f"[eval] generated: {len(generated_seqs)} sequences")
    print(f"[eval] reference: {len(reference)} sequences")

    # Embedder for distributional metrics (FBD, MMD, precision/recall).
    print(f"[eval] loading ESM-2 embedder: {args.esm_model}")
    embedder = sm.models.ESM2(model_name=args.esm_model, device=args.device)

    metrics = [
        sm.metrics.Uniqueness(),
        sm.metrics.Novelty(reference=reference),
        sm.metrics.Length(),
        sm.metrics.Diversity(),
    ]
    # Distributional metrics — add if the seqme version exposes them.
    for name in ("FBD", "MMD", "PrecisionRecall", "ConformityScore"):
        cls = getattr(sm.metrics, name, None)
        try:
            if name == "FBD":
                metrics.append(cls(reference=reference, embedder=embedder))
            elif name == "MMD":
                metrics.append(cls(reference=reference, embedder=embedder))
            elif name == "PrecisionRecall":
                metrics.append(cls(reference=reference, embedder=embedder))
            elif name == "ConformityScore":
                metrics.append(cls(reference=reference))
        except Exception as e:
            print(f"[eval] skip {name}: {e}")

    print(f"[eval] computing {len(metrics)} metrics...")
    # seqme expects {group_name: [sequences]}; we score the whole library as one group.
    groups = {"library": generated_seqs}
    df = sm.evaluate(groups, metrics)
    print("\n=== seqme results ===")
    print(df.to_string())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
