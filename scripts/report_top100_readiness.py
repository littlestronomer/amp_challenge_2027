"""Audit a frozen top-100 and its available per-peptide surrogate scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from amp_challenge_2027.config import (
    ANTIBACTERIAL_FASTA,
    LIBRARY_SIZE,
    TOP_K,
    TOP_SIMILARITY_THRESHOLD,
)
from amp_challenge_2027.data import iter_fasta
from amp_challenge_2027.props import compute_properties, is_plausible
from amp_challenge_2027.select import is_valid_sequence


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_keyed_csv(path: Path | None, prefix: str) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sequence" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a sequence column")
        for line, row in enumerate(reader, start=2):
            sequence = row["sequence"].strip().upper()
            if not sequence:
                raise ValueError(f"Empty sequence key in {path}, row {line}")
            if sequence in records:
                raise ValueError(f"Duplicate sequence key in {path}: {sequence}")
            records[sequence] = {
                f"{prefix}{key}": value for key, value in row.items()
                if key and key != "sequence" and value is not None
            }
    return records


def audit(library_path: Path, top_path: Path, reference_path: Path,
          scores_path: Path | None, synthesis_path: Path | None,
          draws: int, seed: int) -> tuple[list[dict], dict]:
    try:
        import Levenshtein
    except ImportError as exc:
        raise RuntimeError("This audit requires the authoritative Levenshtein package") from exc
    import numpy as np

    library = [sequence for _, sequence in iter_fasta(library_path)]
    top = [sequence for _, sequence in iter_fasta(top_path)]
    reference = [sequence for _, sequence in iter_fasta(reference_path)]
    reference_set = set(reference)
    if not reference_set:
        raise ValueError("Competition reference FASTA is empty")
    if len(library) != LIBRARY_SIZE or len(set(library)) != len(library):
        raise ValueError(f"Expected {LIBRARY_SIZE} unique library sequences")
    if not all(is_valid_sequence(sequence) for sequence in library):
        raise ValueError("Library contains an invalid amino-acid sequence")
    if len(top) != TOP_K or len(set(top)) != TOP_K:
        raise ValueError(f"Expected {TOP_K} unique top sequences")
    if not set(top).issubset(set(library)):
        raise ValueError("Top list contains a sequence outside the source library")
    if set(library) & reference_set:
        raise ValueError("Library has exact overlap with the competition reference")

    score_rows = read_keyed_csv(scores_path, "score_")
    synthesis_rows = read_keyed_csv(synthesis_path, "synthesis_")
    scores_complete = all(sequence in score_rows for sequence in top)
    similarity = [max((Levenshtein.ratio(sequence, ref) for ref in reference), default=0.0)
                  for sequence in top]
    if max(similarity) > TOP_SIMILARITY_THRESHOLD:
        raise ValueError("Top list exceeds the inclusive 0.8 Levenshtein similarity limit")

    # Connected components at >=80% pairwise Levenshtein similarity provide an
    # explicit family-redundancy diagnostic. The graph is intentionally small
    # (100 nodes), and transitive links are retained.
    parent = list(range(len(top)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    near_pairs = 0
    for i, left in enumerate(top):
        for j in range(i):
            if Levenshtein.ratio(left, top[j]) >= 0.8:
                near_pairs += 1
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    families = [find(i) for i in range(len(top))]

    numeric_score_names: set[str] = set()
    for row in score_rows.values():
        for key, raw in row.items():
            if key in {"score_rank", "score_sequence"}:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                numeric_score_names.add(key)
    rows = []
    for rank, sequence in enumerate(top, start=1):
        properties = compute_properties(sequence)
        row = {
            "rank": rank, "sequence": sequence, "in_library": True,
            "valid": is_plausible(sequence), "reference_similarity_max": similarity[rank - 1],
            "family_id": families[rank - 1], "library_sha256": file_hash(library_path),
        }
        row.update(score_rows.get(sequence, {}))
        row.update(synthesis_rows.get(sequence, {}))
        row.update({"length": properties.length, "charge": properties.charge,
                    "hydrophobicity_kd": properties.hydrophobicity_kd,
                    "hydrophobic_moment": properties.hydrophobic_moment})
        rows.append(row)

    summary: dict = {
        "kind": "top100_readiness_v1", "library_count": len(library), "top_count": len(top),
        "library_unique": len(set(library)) == len(library), "top_unique": len(set(top)) == len(top),
        "library_valid": all(is_valid_sequence(sequence) for sequence in library),
        "top_membership_complete": True, "all_plausible": all(is_plausible(sequence) for sequence in top),
        "exact_reference_overlap_count": len(set(library) & reference_set),
        "reference_similarity_max": max(similarity), "reference_similarity_threshold": TOP_SIMILARITY_THRESHOLD,
        "top_pairwise_near_identity_pairs_ge_0_8": near_pairs,
        "top_family_component_count_ge_0_8": len(set(families)),
        "score_source": str(scores_path) if scores_path else None,
        "score_coverage_count": sum(sequence in score_rows for sequence in top),
        "score_coverage_complete": scores_complete,
        "synthesis_assessment": "provided" if synthesis_path else "not_assessed",
        "synthesis_coverage_count": sum(sequence in synthesis_rows for sequence in top),
        "score_summary": {},
        "random_25_subset_diagnostic": {
            "status": "conditional_on_fixed_surrogate_scores", "draws": draws, "seed": seed,
            "subset_size": min(25, len(top)),
            "does_not_estimate_wet_lab_success": True, "metrics": {},
            "unique_families": {"mean": None, "p05": None, "p95": None},
        },
        "inputs": {"library_sha256": file_hash(library_path), "top_sha256": file_hash(top_path),
                   "library_path": str(library_path.resolve()), "top_path": str(top_path.resolve()),
                   "reference_path": str(reference_path.resolve()),
                   "reference_sha256": file_hash(reference_path),
                   "scores_path": str(scores_path.resolve()) if scores_path else None,
                   "scores_sha256": file_hash(scores_path) if scores_path else None,
                   "synthesis_path": str(synthesis_path.resolve()) if synthesis_path else None,
                   "synthesis_sha256": file_hash(synthesis_path) if synthesis_path else None},
        "limitations": [
            "Activity and hemolysis fields are model predictions, not experimental measurements.",
            "No predicted score estimates wet-lab success or replaces prospective assays.",
            "Synthesis suitability is not assessed unless an independently sourced table is supplied.",
        ],
    }
    for name in sorted(numeric_score_names):
        values = []
        for sequence in top:
            raw = score_rows.get(sequence, {}).get(name)
            try:
                value = float(raw) if raw is not None else float("nan")
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
        if values:
            summary["score_summary"][name] = {
                "coverage": len(values), "mean": float(np.mean(values)),
                "p05": float(np.percentile(values, 5)), "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
        # Only scalar score columns are eligible for subset mean simulations.
        if values and name != "score_rank":
            raw_all = [score_rows.get(sequence, {}).get(name) for sequence in top]
            try:
                all_values = np.asarray([float(value) for value in raw_all], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(all_values).all():
                continue
            rng = np.random.default_rng(seed)
            sampled = np.stack([rng.choice(len(top), size=min(25, len(top)), replace=False)
                                 for _ in range(draws)])
            means = all_values[sampled].mean(axis=1)
            summary["random_25_subset_diagnostic"]["metrics"][name] = {
                "mean_of_subset_means": float(means.mean()), "p05": float(np.percentile(means, 5)),
                "p95": float(np.percentile(means, 95)),
            }
    rng = np.random.default_rng(seed)
    sampled = np.stack([rng.choice(len(top), size=min(25, len(top)), replace=False)
                        for _ in range(draws)])
    family_counts = np.asarray([len({families[int(index)] for index in row}) for row in sampled])
    summary["random_25_subset_diagnostic"]["unique_families"] = {
        "mean": float(family_counts.mean()), "p05": float(np.percentile(family_counts, 5)),
        "p95": float(np.percentile(family_counts, 95)),
    }
    summary["random_25_subset_diagnostic"]["draws"] = draws
    return rows, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--top", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--synthesis-assessment", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    if args.draws < 1:
        parser.error("--draws must be positive")
    if args.out.exists():
        raise FileExistsError(f"Use a new output directory: {args.out}")
    rows, summary = audit(args.library, args.top, args.reference, args.scores,
                          args.synthesis_assessment, args.draws, args.seed)
    args.out.mkdir(parents=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (args.out / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out / "REPORT.md").write_text(render_report(summary))
    files = {name: file_hash(args.out / name) for name in ("candidates.csv", "summary.json", "REPORT.md")}
    (args.out / "complete.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    print(f"Top-100 audit written to {args.out.resolve()}")


def render_report(summary: dict) -> str:
    lines = ["# Top-100 readiness report", "",
             "This report checks sequence validity, exact novelty and the supplied prediction table. Model scores are not assay results.", "",
             "## Sequence checks", "",
             f"- Library sequences: {summary['library_count']}",
             f"- Top sequences: {summary['top_count']}",
             f"- All top sequences are plausible: {summary['all_plausible']}",
             f"- Maximum reference similarity: {summary['reference_similarity_max']:.4f} (limit {summary['reference_similarity_threshold']:.2f})",
             f"- Top pairwise pairs at >=0.8 similarity: {summary['top_pairwise_near_identity_pairs_ge_0_8']}",
             f"- Connected sequence families at >=0.8 similarity: {summary['top_family_component_count_ge_0_8']}", "",
             "## Supplied score coverage", ""]
    if summary["score_summary"]:
        lines.extend(["| Score field | Coverage | Mean | 5th pct | Median | 95th pct |", "|---|---:|---:|---:|---:|---:|"])
        for name, item in summary["score_summary"].items():
            lines.append(f"| {name} | {item['coverage']} | {item['mean']:.4f} | {item['p05']:.4f} | {item['p50']:.4f} | {item['p95']:.4f} |")
    else:
        lines.append("No per-sequence score table was supplied; model quality fields are unavailable.")
    lines.extend(["", "## Random 25-subset diagnostic", "",
                  "Sampling summaries condition on the fixed top-100 and supplied scores. They do not estimate wet-lab success.", ""])
    diag = summary["random_25_subset_diagnostic"]
    lines.append(f"- Draws: {diag['draws']} of {diag['subset_size']}; seed: {diag['seed']}")
    lines.append(f"- Distinct >=0.8 sequence families per random subset: mean {diag['unique_families']['mean']:.2f}, 5th–95th percentile {diag['unique_families']['p05']:.0f}–{diag['unique_families']['p95']:.0f}")
    lines.extend(["", "## Missing evidence", "", f"- Per-sequence score coverage complete: {summary['score_coverage_complete']}",
                  f"- Synthesis suitability: {summary['synthesis_assessment']} ({summary['synthesis_coverage_count']} of 100 have a supplied row)",
                  "- Per-genus probabilities are not included in the generated score file.",
                  "- Hemolysis evidence is a prediction only when present in the supplied score file; it is not experimental safety evidence.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
