"""Score a peptide library under the full Phase-1 protocol (not just a subset).

Replicates the competition's four metric families as described in the proposal
(§1.5), using the same seqme library the organizers use. This gives you the
most accurate local estimate of your Phase-1 standing.

Metric construction lives in ``amp_challenge_2027.metrics_official`` so this
CLI and ``scripts/sweep_selection.py`` share one implementation.

The exact aggregation weights and reference-set composition are HELD BACK by
the organizers until Phase-1 closes, so this can't reproduce the official
ranking — but it gives you all the component metrics under matched conditions.

Usage:
    uv run --extra ml --extra seqme python scripts/eval_official.py \\
        --library generate/library.fasta --esm-model facebook/esm2_t6_8M_UR50D --device cuda
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
    parser.add_argument("--json-out", type=Path, default=None, help="Flat metrics for experiment summaries")
    parser.add_argument("--strict", action="store_true", help="Require all 12 core metrics; amp+charge conformity is optional")
    args = parser.parse_args(argv)

    import seqme as sm

    from amp_challenge_2027.metrics_official import build_metric_list

    generated = load_fasta(args.library)
    reference = load_fasta(args.reference)
    print(f"[eval] generated: {len(generated)} sequences")
    print(f"[eval] reference: {len(reference)} sequences")
    print(f"[eval] embedder: {args.esm_model} on {args.device}")

    print("[eval] loading ESM-2 embedder...")
    embedder = sm.models.ESM2(model_name=args.esm_model, device=args.device)

    metrics = build_metric_list(reference, embedder, strict=args.strict)
    names = [n for n, _ in metrics]
    metric_objs = [m for _, m in metrics]
    print(f"[eval] computing {len(metric_objs)} metrics: {names}")

    groups = {"library": generated}
    df = sm.evaluate(groups, metric_objs)
    print("\n" + "=" * 80)
    print("FULL PHASE-1 PROTOCOL RESULTS")
    print("=" * 80)
    print(df.to_string())

    if args.strict or args.json_out:
        from experiment_utils import metrics_for_json

        row = metrics_for_json(
            df, expected_names=[metric.name for metric in metric_objs] if args.strict else None,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        print(f"\n[eval] wrote {args.out}")
    if args.json_out:
        from experiment_utils import write_json

        write_json(args.json_out, row)


if __name__ == "__main__":
    main()
