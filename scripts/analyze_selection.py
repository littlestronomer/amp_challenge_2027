"""Replay and diagnose immutable top-100 score caches on CPU; no neural inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import compare_top100 as baseline
import numpy as np
from experiment_utils import (
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from selection_cache import load_source, separate_output, verify_source

from amp_challenge_2027.data import write_fasta
from amp_challenge_2027.selection_audit import (
    distribution,
    fit_normalization,
    normalized_score,
    paired_deltas,
    rank_correlation,
    selection_stages,
)

TABLES = ("component_distribution", "effective_weights", "selection_funnel",
          "component_contributions", "rank_correlations")
CELL_FILES = ["top.fasta", "top_scores.csv", "summary.json", "stages.npz", "normalization.json"]
CELL_FILES += [f"{name}.csv" for name in TABLES]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=Path("sweep_results/epoch58-top100-v1"))
    result.add_argument("--reference", type=Path, default=baseline.ANTIBACTERIAL_FASTA)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--list", action="store_true", help="Validate all source caches without writing or selecting")
    return result


def top_report(cell: dict, stages: dict, combined: np.ndarray, reference: list[str]) -> tuple[dict, list[dict]]:
    from Levenshtein import ratio

    ids = stages["top"]
    top = [cell["sequences"][i] for i in ids]
    nearest = [max(ratio(seq, ref) for ref in reference) for seq in top]
    if max(nearest) > baseline.TOP_SIMILARITY_THRESHOLD:
        raise ValueError("Top selection failed exact reference novelty audit")
    distances = [1 - ratio(seq, other) for i, seq in enumerate(top) for other in top[:i]]
    old_risk = {row["sequence"]: float(row["hemo_risk"]) for row in cell["top_rows"]}
    risk = [old_risk.get(seq) for seq in top]
    complete_risk = all(value is not None for value in risk)
    summary = {"case": cell["case"], "seed": cell["seed"], "top_size": len(top), "top_sequences": top,
               "library_sha256": cell["record"]["library_sha256"],
               **{f"{name}_mean": float(values[ids].mean()) for name, values in cell["parts"].items()},
               "panel_mean_probability": float(cell["scores"]["panel"][ids].mean()),
               "reference_similarity_max": max(nearest), "top_mean_pairwise_distance": float(np.mean(distances)),
               "hemo_risk_known_count": sum(value is not None for value in risk),
               "hemo_risk_status": "complete_cached" if complete_risk else "incomplete_needs_scoring",
               "hemo_risk_mean": float(np.mean(risk)) if complete_risk else None,
               "hemo_risk_p75": float(np.percentile(risk, 75)) if complete_risk else None,
               "hemo_risk_max": float(np.max(risk)) if complete_risk else None}
    rows = []
    for rank, i in enumerate(ids):
        rows.append({"rank": rank + 1, "sequence": top[rank], "library_index": int(i),
                     "composite_score": float(combined[i]), "hemo_risk": risk[rank],
                     "reference_similarity_max": nearest[rank],
                     **{name: float(values[i]) for name, values in cell["parts"].items()},
                     "panel_mean_probability": float(cell["scores"]["panel"][i].mean()),
                     **{f"p_active:{genus}": float(cell["scores"]["panel"][i, j])
                        for j, genus in enumerate(baseline.PANEL_GENERA)}})
    return summary, rows


def diagnostics(cell: dict, policy: str, stages: dict, combined: np.ndarray,
                contributions: dict, normalization: dict) -> dict[str, list[dict]]:
    tag = {"policy": policy, "case": cell["case"], "seed": cell["seed"]}
    tables = {name: [] for name in TABLES}
    measures = {**cell["parts"], "panel_mean_probability": cell["scores"]["panel"].mean(axis=1),
                "composite": combined,
                **{f"panel:{genus}": cell["scores"]["panel"][:, j] for j, genus in enumerate(baseline.PANEL_GENERA)}}
    for name, weight in baseline.WEIGHTS.items():
        item = normalization[name]
        tables["effective_weights"].append({**tag, "component": name, "weight": weight, **item,
                                             "weight_over_denominator": weight / item["denominator"],
                                             "coefficient_in_average": weight / item["denominator"] / sum(baseline.WEIGHTS.values())})
    for stage, ids in stages.items():
        tables["selection_funnel"].append({**tag, "stage": stage, "n": len(ids)})
        for name, values in measures.items():
            tables["component_distribution"].append({**tag, "stage": stage, "component": name, **distribution(values[ids])})
            tables["rank_correlations"].append({**tag, "stage": stage, "component": name,
                                                 "versus": "composite", "n": len(ids),
                                                 "spearman": rank_correlation(values[ids], combined[ids])})
        for name, values in contributions.items():
            tables["component_contributions"].append({**tag, "stage": stage, "component": name, **distribution(values[ids])})
    return tables


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for policy in sorted({r["policy"] for r in rows}):
        for case in ("hybrid", "p3_s1"):
            group = [r for r in rows if r["policy"] == policy and r["case"] == case]
            for key in group[0]:
                if key in {"seed", "top_size"}:
                    continue
                values = [r[key] for r in group if isinstance(r.get(key), (float, int)) and not isinstance(r[key], bool)]
                if not values:
                    continue
                output.append({"policy": policy, "case": case, "metric": key, "n_seeds": len(values),
                               "n_seeds_expected": len(group),
                               "mean": float(np.mean(values)) if len(values) == len(group) else None,
                               "std": float(np.std(values, ddof=1)) if len(values) == len(group) and len(values) > 1 else None})
    return output


def run(args, *, policies: tuple[str, ...]) -> None:
    separate_output(args.out, [args.source, args.reference])
    identity, cells, reference = load_source(args.source, args.reference)
    anchor = next(c for c in cells if (c["case"], c["seed"]) == ("hybrid", 42))
    fixed = fit_normalization(anchor["parts"])
    for policy in policies:
        for cell in cells:
            print(f"[selection-audit] {policy}/{cell['case']}/seed{cell['seed']}: verified cache", flush=True)
    if args.list:
        return
    recipe = {"kind": "cached_selection_diagnosis_v1", "code": code_identity(), "inputs": identity,
              "policies": list(policies), "weights": baseline.WEIGHTS, "fixed_normalization": fixed,
              "anchor": {"case": "hybrid", "seed": 42, "library_sha256": anchor["record"]["library_sha256"],
                         "scores_sha256": anchor["proof"]["files"]["scores.npz"]},
              "interpretation": "exploratory generation seeds; surrogate scores, no biological validation"}
    prepare_run(args.out, recipe)
    if (args.out / "complete.json").exists():
        baseline.checked_stage(args.out, "complete.json", {"results.csv", "inventory.json", "normalization_anchor.json"})
    write_json(args.out / "inventory.json", identity)
    write_json(args.out / "normalization_anchor.json", {"anchor": recipe["anchor"], "components": fixed})
    summaries = []
    tables = {name: [] for name in TABLES}
    # Every P0 replay must pass before *any* P1 experiment is computed.
    for policy in policies:
        for cell in cells:
            directory = args.out / policy / cell["case"] / f"seed{cell['seed']}"
            if (directory / "complete.json").exists():
                baseline.checked_stage(directory, "complete.json", set(CELL_FILES))
            else:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"[selection-audit] selecting {policy}/{cell['case']}/seed{cell['seed']} (CPU)", flush=True)
                normalization = fit_normalization(cell["parts"]) if policy == "P0" else fixed
                combined, contributions = normalized_score(cell["parts"], baseline.WEIGHTS, normalization)
                if policy == "P0" and not np.array_equal(combined, cell["combined"]):
                    raise ValueError("P0 composite replay differs from production; stop")
                stages = selection_stages(cell["sequences"], combined, set(reference), top_k=baseline.TOP_K,
                                          seed=baseline.RANKING_SEED, shortlist=2000)
                top = [cell["sequences"][i] for i in stages["top"]]
                write_fasta(top, directory / "top.fasta")
                require_parity = policy == "P0" or (cell["case"], cell["seed"]) == ("hybrid", 42)
                parity = sha256(directory / "top.fasta") == cell["proof"]["files"]["top.fasta"]
                if require_parity and not parity:
                    raise ValueError("P0 or P1-anchor top FASTA is not byte-identical; investigate replay before proceeding")
                summary, details = top_report(cell, stages, combined, reference)
                summary.update({"policy": policy, "matches_original_top_bytes": parity})
                write_json(directory / "summary.json", summary)
                write_summary(directory / "top_scores.csv", details)
                np.savez_compressed(directory / "stages.npz", **stages)
                write_json(directory / "normalization.json", normalization)
                for name, rows in diagnostics(cell, policy, stages, combined, contributions, normalization).items():
                    write_summary(directory / f"{name}.csv", rows)
                mark_files(directory, "complete.json", CELL_FILES)
            summary = json.loads((directory / "summary.json").read_text())
            summaries.append(summary)
            for name in TABLES:
                with (directory / f"{name}.csv").open(newline="") as handle:
                    tables[name].extend(csv.DictReader(handle))
    verify_source(identity)
    if sha256(args.reference) != identity["source_run"]["reference_sha256"]:
        raise ValueError("Reference changed during analysis")
    for name, rows in tables.items():
        write_summary(args.out / f"{name}.csv", rows)
    def plain(rows):
        return [{k: v for k, v in row.items() if k != "top_sequences"} for row in rows]

    write_summary(args.out / "baseline_results.csv", plain([s for s in summaries if s["policy"] == "P0"]))
    write_summary(args.out / "results.csv", plain(summaries))
    write_summary(args.out / "seed_summary.csv", aggregate(summaries))
    write_summary(args.out / "case_paired_deltas.csv", paired_deltas(summaries, "policy"))
    root_files = ["inventory.json", "normalization_anchor.json", "baseline_results.csv", "results.csv",
                  "seed_summary.csv", "case_paired_deltas.csv"] + [f"{name}.csv" for name in TABLES]
    if len(policies) == 2:
        write_summary(args.out / "policy_paired_deltas.csv", paired_deltas(summaries, "case"))
        root_files.append("policy_paired_deltas.csv")
    mark_files(args.out, "complete.json", root_files)
    print(f"[selection-audit] complete: {args.out}/results.csv; no generation, neural inference or promotion", flush=True)


def main(argv: list[str] | None = None) -> None:
    run(parser().parse_args(argv), policies=("P0",))


if __name__ == "__main__":
    main()
