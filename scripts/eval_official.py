"""Score a peptide library under the full Phase-1 protocol (not just a subset).

Replicates the competition's four metric families as described in the proposal
(§1.5), using the same seqme library the organizers use. This gives you the
most accurate local estimate of your Phase-1 standing.

Metric families (per the competition PDF):
  1. Sequence-level:      Uniqueness, Novelty, Diversity, Length, NGramJaccard
  2. Distributional:      FBD, MMD, Precision, Recall (via ESM-2 embeddings)
  3. Property conformity: ConformityScore (charge, hydrophobicity, amphipathicity)
  4. Surrogate activity:  HitRate (AMP classifier surrogate)

The exact aggregation weights and reference-set composition are HELD BACK by
the organizers until Phase-1 closes, so this can't reproduce the official
ranking — but it gives you all the component metrics under matched conditions.

Usage:
    .venv/bin/python -u -c "..." --library generate/submission/library.fasta \\
        --esm-model facebook/esm2_t33_650M_UR50D --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA
from amp_challenge_2027.data import iter_fasta


def load_fasta(path: Path) -> list[str]:
    return [seq for _, seq in iter_fasta(path)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full Phase-1 protocol scoring.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument(
        "--esm-model", type=str, default="facebook/esm2_t6_8M_UR50D",
        help="ESM-2 model (8M fast, 650M for official-fidelity)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import seqme as sm

    generated = load_fasta(args.library)
    reference = load_fasta(args.reference)
    print(f"[eval] generated: {len(generated)} sequences")
    print(f"[eval] reference: {len(reference)} sequences")
    print(f"[eval] embedder: {args.esm_model} on {args.device}")

    # --- Embedder (shared across distributional metrics) ---
    print("[eval] loading ESM-2 embedder...")
    embedder = sm.models.ESM2(model_name=args.esm_model, device=args.device)

    # --- Property predictors (for ConformityScore) ---
    print("[eval] loading property predictors...")
    predictors = {
        "charge": sm.models.Charge(),
        "hydrophobicity": sm.models.Hydrophobicity(),
        "hydrophobic_moment": sm.models.HydrophobicMoment(),
    }

    # --- Build the full metric set ---
    metrics = []

    # Family 1: Sequence-level
    metrics.append(("Uniqueness", sm.metrics.Uniqueness()))
    metrics.append(("Novelty", sm.metrics.Novelty(reference=reference)))
    metrics.append(("Diversity", sm.metrics.Diversity()))
    metrics.append(("Length", sm.metrics.Length()))
    try:
        metrics.append(("NGramJaccard", sm.metrics.NGramJaccardSimilarity(reference=reference)))
    except Exception as e:
        print(f"[eval] skip NGramJaccard: {e}")

    # Family 2: Distributional (embedding-space)
    try:
        metrics.append(("FBD", sm.metrics.FBD(reference=reference, embedder=embedder)))
    except Exception as e:
        print(f"[eval] skip FBD: {e}")
    try:
        metrics.append(("MMD", sm.metrics.MMD(reference=reference, embedder=embedder)))
    except Exception as e:
        print(f"[eval] skip MMD: {e}")
    try:
        pr = sm.metrics.PrecisionRecall(reference=reference, embedder=embedder)
        metrics.append(("PrecisionRecall", pr))
    except Exception as e:
        print(f"[eval] skip PrecisionRecall: {e}")

    # Family 3: Property conformity
    try:
        metrics.append((
            "ConformityScore",
            sm.metrics.ConformityScore(reference=reference, predictors=predictors),
        ))
    except Exception as e:
        print(f"[eval] skip ConformityScore: {e}")

    # Family 4: Authenticity (fraction passing all property filters)
    try:
        metrics.append(("AuthPct", sm.metrics.AuthPct(reference=reference)))
    except Exception as e:
        print(f"[eval] skip AuthPct: {e}")

    names = [n for n, _ in metrics]
    metric_objs = [m for _, m in metrics]
    print(f"[eval] computing {len(metric_objs)} metrics: {names}")

    groups = {"library": generated}
    df = sm.evaluate(groups, metric_objs)
    print("\n" + "=" * 80)
    print("FULL PHASE-1 PROTOCOL RESULTS")
    print("=" * 80)
    print(df.to_string())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
