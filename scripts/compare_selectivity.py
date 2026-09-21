"""CPU-only paired C0/R1 selection comparison on frozen hybrid libraries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import compare_top100 as baseline
import numpy as np
from cache_selectivity_risk import _build_union
from experiment_utils import (
    REPO_ROOT,
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
)
from selection_cache import load_source, separate_output, verify_source

from amp_challenge_2027.config import MDR_PANEL_GENERA
from amp_challenge_2027.data import write_fasta
from amp_challenge_2027.props import compute_properties, is_plausible
from amp_challenge_2027.selection_audit import selection_stages
from amp_challenge_2027.selectivity import (
    LAMBDA,
    adjusted_score,
    criteria_for,
    paired_policy_deltas,
    sequence_digest,
    validate_protocol,
    validate_risk,
)


def _read_risk_cache(root: Path, cells: list[dict], source_identity: dict,
                     protocol_path: Path, reference: Path, *,
                     strict_runtime_code: bool = True) -> tuple[dict, dict[int, np.ndarray]]:
    recipe = json.loads((root / "run.json").read_text())
    if (recipe.get("kind") != "selectivity_risk_cache_v1"
            or recipe.get("source_run_sha256") != source_identity["run_sha256"]
            or recipe.get("protocol_sha256") != sha256(protocol_path)
            or recipe.get("reference_sha256") != sha256(reference)):
        raise ValueError("Risk cache does not match source, protocol, or reference")
    current_code = code_identity()
    source_head = source_identity["source_run"]["classifiers"]["hemolysis"]
    union, maps = _build_union(cells)
    if ((strict_runtime_code and recipe.get("runtime", {}).get("code") != current_code)
            or recipe.get("hemolysis_head_files") != source_head.get("files")
            or recipe.get("backbone_revision") != source_identity["backbones"].get("hemolysis")
            or recipe.get("union_sequence_sha256") != sequence_digest(union)
            or recipe.get("union_unique_sequences") != len(union)):
        raise ValueError("Risk-cache code, model, backbone or ordered-sequence identity differs")
    for seed, item in maps.items():
        record = recipe.get("cells", {}).get(str(seed), {})
        if (record.get("library_sha256") != item["cell"]["record"]["library_sha256"]
                or record.get("index_map_sha256") != sequence_digest([str(int(i)) for i in item["indices"]])):
            raise ValueError(f"Risk-cache aligned index map differs for seed {seed}")
    complete = json.loads((root / "complete.json").read_text())
    if complete.get("kind") != "selectivity_risk_cache_complete_v1":
        raise ValueError("Risk cache lacks its full-completion marker")
    for name, expected in complete.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Risk-cache artifact mismatch: {name}")
    coverage = json.loads((root / "coverage.json").read_text())
    if coverage.get("complete") is not True or any(coverage.get("cells", {}).get(str(seed)) is not True for seed in (42, 43, 44)):
        raise ValueError("Risk cache is partial; comparison requires all three seeds")
    arrays = {}
    proofs = complete.get("cells", {})
    if set(proofs) != {"42", "43", "44"}:
        raise ValueError("Risk-cache completion marker lacks all seed cell hashes")
    for seed in (42, 43, 44):
        cell = next(c for c in cells if c["case"] == "hybrid" and c["seed"] == seed)
        directory = root / "cells" / "hybrid" / f"seed{seed}"
        verify_files(directory, "complete.json")
        identity = json.loads((directory / "identity.json").read_text())
        stored_indices = np.load(directory / "indices.npy", allow_pickle=False)
        if (identity.get("library_sha256") != cell["record"]["library_sha256"]
                or identity.get("sequence_sha256") != sequence_digest(cell["sequences"])
                or identity.get("risk_sha256") != sha256(directory / "risk.npy")
                or identity.get("indices_sha256") != sha256(directory / "indices.npy")
                or not np.array_equal(stored_indices, maps[seed]["indices"])
                or proofs[str(seed)].get("complete_sha256") != sha256(directory / "complete.json")
                or proofs[str(seed)].get("risk_sha256") != sha256(directory / "risk.npy")):
            raise ValueError(f"Risk sequence identity differs for seed {seed}")
        values = np.load(directory / "risk.npy", allow_pickle=False)
        arrays[seed] = validate_risk(values, len(cell["sequences"]))
    return recipe, arrays


def _top_metrics(cell: dict, stages: dict, risk: np.ndarray, reference: list[str], policy: str) -> tuple[dict, list[dict]]:
    from Levenshtein import ratio

    ids = stages["top"]
    top = [cell["sequences"][int(i)] for i in ids]
    top_risk = validate_risk(risk[ids], len(ids))
    if (len(top) != baseline.TOP_K or len(set(top)) != baseline.TOP_K
            or not set(top) <= set(cell["sequences"]) or not all(is_plausible(seq) for seq in top)):
        raise ValueError("Selection failed count, uniqueness, membership, or plausibility")
    nearest = [max(ratio(seq, ref) for ref in reference) for seq in top]
    if max(nearest) > baseline.TOP_SIMILARITY_THRESHOLD:
        raise ValueError("Selection failed exact Levenshtein reference novelty")
    distances = [1 - ratio(seq, other) for i, seq in enumerate(top) for other in top[:i]]
    panel = cell["scores"]["panel"][ids]
    mdr_indices = [i for i, genus in enumerate(baseline.PANEL_GENERA) if genus in MDR_PANEL_GENERA]
    summary = {
        "policy": policy, "case": "hybrid", "seed": cell["seed"], "top_size": len(top),
        "top_sequences": top, "library_sha256": cell["record"]["library_sha256"],
        "hemo_risk_mean": float(top_risk.mean()), "hemo_risk_median": float(np.median(top_risk)),
        "hemo_risk_p75": float(np.percentile(top_risk, 75)), "hemo_risk_max": float(top_risk.max()),
        "activity_mean": float(cell["scores"]["activity"][ids].mean()),
        "panel_mean_probability": float(panel.mean()),
        "breadth_mean": float((panel > 0.5).mean(axis=1).mean()),
        "mdr_mean": float((panel[:, mdr_indices] > 0.5).mean(axis=1).mean()),
        "conformity_mean": float(cell["scores"]["conformity"][ids].mean()),
        "precision_mean": float(cell["scores"]["precision"][ids].mean()),
        "top_mean_pairwise_distance": float(np.mean(distances)),
        "reference_similarity_max": float(max(nearest)), "all_valid": True,
        **{f"p_active:{genus}_mean": float(panel[:, j].mean()) for j, genus in enumerate(baseline.PANEL_GENERA)},
    }
    rows = []
    for rank, i in enumerate(ids):
        sequence = cell["sequences"][int(i)]
        prop = compute_properties(sequence)
        rows.append({"rank": rank + 1, "sequence": sequence, "library_index": int(i),
                     "base_combined_score": float(cell["combined"][i]), "risk_adjusted_score": float(stages["scores"][i]),
                     "hemo_risk": float(risk[i]), "reference_similarity_max": float(nearest[rank]),
                     **{name: float(values[i]) for name, values in cell["parts"].items()},
                     "panel_mean_probability": float(cell["scores"]["panel"][i].mean()),
                     **{f"p_active:{genus}": float(cell["scores"]["panel"][i, j]) for j, genus in enumerate(baseline.PANEL_GENERA)},
                     "length": prop.length, "charge": prop.charge, "hydrophobicity_kd": prop.hydrophobicity_kd,
                     "hydrophobic_moment": prop.hydrophobic_moment, "cysteine_count": prop.cysteine_count,
                     "fraction_positive": prop.fraction_positive, "fraction_hydrophobic": prop.fraction_hydrophobic})
    return summary, rows


def run(args) -> int:
    args.source, args.risk_cache, args.protocol, args.reference, args.out = map(
        lambda p: p.resolve(), (args.source, args.risk_cache, args.protocol, args.reference, args.out))
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if (baseline.TOP_K != protocol["top_size"] or baseline.RANKING_SEED != protocol["ranking_seed"]
            or baseline.TOP_SIMILARITY_THRESHOLD != protocol["novelty_threshold"]):
        raise ValueError("Production selector constants differ from the frozen protocol")
    if args.source != (REPO_ROOT / protocol["source"]).resolve():
        raise ValueError("Source path differs from the frozen protocol")
    separate_output(args.out, [args.source, args.risk_cache, args.protocol, args.reference])
    source_identity, all_cells, reference = load_source(args.source, args.reference)
    hybrid = sorted((cell for cell in all_cells if cell["case"] == "hybrid"), key=lambda c: c["seed"])
    risk_recipe, risks = _read_risk_cache(args.risk_cache, all_cells, source_identity, args.protocol, args.reference)
    recipe = {"kind": "selectivity_comparison_v1", "code": code_identity(), "source": source_identity,
              "risk_cache": risk_recipe, "protocol_sha256": sha256(args.protocol),
              "reference_sha256": sha256(args.reference), "policies": ["C0", "R1"], "lambda": LAMBDA}
    print("[selectivity] verified three frozen hybrid libraries and complete aligned risk arrays; CPU-only")
    if args.list:
        print("[selectivity] --list validates inputs only; no inference, selection output, or files written")
        return 0
    prepare_run(args.out, recipe)
    if (args.out / "complete.json").exists():
        complete = json.loads((args.out / "complete.json").read_text())
        verify_files(args.out, "complete.json")
        for cell_name, expected in complete.get("cells", {}).items():
            cell_dir = args.out / cell_name
            marker = cell_dir / "complete.json"
            if not marker.is_file() or sha256(marker) != expected:
                raise ValueError(f"Completed comparison cell changed: {cell_name}")
            verify_files(cell_dir, "complete.json")
        verify_source(source_identity)
        print("[selectivity] matching completed report already exists and verifies")
        return 0

    # Validate all C0 replays before any R1 is written.
    stages_by_seed = {}
    for cell in hybrid:
        stages = selection_stages(cell["sequences"], cell["combined"], set(reference),
                                  top_k=baseline.TOP_K, seed=baseline.RANKING_SEED,
                                  shortlist=protocol["shortlist"])
        selected = [cell["sequences"][int(i)] for i in stages["top"]]
        if selected != cell["top"]:
            raise ValueError(f"C0 replay differs from source top file for seed {cell['seed']}; R1 was not run")
        stages_by_seed[cell["seed"]] = stages

    anchor = validate_risk(risks[42], len(next(c for c in hybrid if c["seed"] == 42)["sequences"]))
    mu, sd = float(anchor.mean(dtype=np.float64)), float(anchor.std(dtype=np.float64))
    if sd <= 1e-8:
        raise ValueError("Anchor hemolysis scores are effectively constant; risk penalty is uninformative")
    normalization = {"seed": 42, "mean": mu, "std": sd, "denominator": sd,
                     "n": len(anchor), "risk_sha256": sha256(args.risk_cache / "cells/hybrid/seed42/risk.npy")}
    write_json(args.out / "normalization_anchor.json", normalization)
    rows = []
    for cell in hybrid:
        seed = cell["seed"]
        cell_risk = risks[seed]
        for policy in ("C0", "R1"):
            stages = stages_by_seed[seed] if policy == "C0" else selection_stages(
                cell["sequences"], adjusted_score(cell["combined"], cell_risk, normalization),
                set(reference), top_k=baseline.TOP_K, seed=baseline.RANKING_SEED,
                shortlist=protocol["shortlist"])
            if policy == "C0":
                stages = {**stages, "scores": cell["combined"]}
            else:
                stages = {**stages, "scores": adjusted_score(cell["combined"], cell_risk, normalization)}
            summary, score_rows = _top_metrics(cell, stages, cell_risk, reference, policy)
            rows.append(summary)
            cell_dir = args.out / policy / "hybrid" / f"seed{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            if (cell_dir / "complete.json").exists():
                verify_files(cell_dir, "complete.json")
                cached_summary = json.loads((cell_dir / "summary.json").read_text())
                if cached_summary != summary:
                    raise ValueError(f"Completed comparison cell differs from recomputed results: {policy}/seed{seed}")
                continue
            if policy == "C0":
                shutil.copyfile(cell["directory"] / "top.fasta", cell_dir / "top.fasta")
            else:
                write_fasta(summary["top_sequences"], cell_dir / "top.fasta", header_prefix="seq")
            with (cell_dir / "top_scores.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            write_json(cell_dir / "summary.json", summary)
            np.savez_compressed(cell_dir / "stages.npz", **{k: np.asarray(v) for k, v in stages.items() if k != "scores"}, scores=stages["scores"])
            mark_files(cell_dir, "complete.json", ["top.fasta", "top_scores.csv", "summary.json", "stages.npz"])

    deltas = paired_policy_deltas(rows)
    criteria = criteria_for(deltas)
    with (args.out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows])
    with (args.out / "paired_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    write_json(args.out / "seed_summary.json", {"n_generation_seeds": 3, "values": rows,
                                                "interpretation": "descriptive seeds, not independent model replicates"})
    write_json(args.out / "criteria.json", criteria)
    inventory = {"kind": "selectivity_comparison_inventory_v1", "source_run_sha256": source_identity["run_sha256"],
                 "risk_cache_sha256": sha256(args.risk_cache / "complete.json"),
                 "cells": [{"seed": cell["seed"], "library_sha256": cell["record"]["library_sha256"],
                            "risk_sha256": sha256(args.risk_cache / "cells/hybrid" / f"seed{cell['seed']}" / "risk.npy")} for cell in hybrid]}
    write_json(args.out / "inventory.json", inventory)
    report = ["# Frozen-pool hemolysis sensitivity comparison", "",
              "This is an exploratory selector sensitivity analysis. Predicted hemolysis risk is a model score under the frozen label definition, not percent lysis or a safety result. Activity and panel outputs are surrogate predictions.", "",
              f"Risk penalty: lambda={LAMBDA}; normalized once on all hybrid seed-42 library scores (mean {mu:.6g}, SD {sd:.6g}).", "",
              f"Predeclared criteria: **{'all passed' if criteria['all_seeds_pass'] else 'not all passed'}**; disposition: `{criteria['interpretation']}`. Passing only supports considering independent evaluation; it never promotes the selector.", "",
              "| Seed | Policy | Risk mean | Risk p75 | Activity mean | Panel mean | Breadth | MDR | Pairwise distance |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['seed']} | {row['policy']} | {row['hemo_risk_mean']:.4f} | {row['hemo_risk_p75']:.4f} | {row['activity_mean']:.4f} | {row['panel_mean_probability']:.4f} | {row['breadth_mean']:.4f} | {row['mdr_mean']:.4f} | {row['top_mean_pairwise_distance']:.4f} |")
    report.extend(["", "The three generation seeds are descriptive examples from one generator/checkpoint context, not independent model replicates. This experiment does not provide external validation, synthesis assessment, wet-lab outcomes, or probability of experimental selection.", ""])
    (args.out / "REPORT.md").write_text("\n".join(report))
    names = ["run.json", "normalization_anchor.json", "results.csv", "paired_deltas.csv", "seed_summary.json", "criteria.json", "inventory.json", "REPORT.md"]
    write_json(args.out / "complete.json", {"kind": "selectivity_comparison_complete_v1",
                                             "files": {name: sha256(args.out / name) for name in names},
                                             "cells": {f"{row['policy']}/hybrid/seed{row['seed']}": sha256(args.out / row["policy"] / "hybrid" / f"seed{row['seed']}" / "complete.json") for row in rows}})
    verify_source(source_identity)
    print(f"[selectivity] report written: {args.out}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("sweep_results/epoch58-top100-v1"))
    parser.add_argument("--risk-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("data/antibacterial.fasta"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="Validate inputs only; does not test output resumability")
    args = parser.parse_args(argv)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
