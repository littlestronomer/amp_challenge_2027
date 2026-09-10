"""Fresh, paired evaluation of baseline, specialist teacher and three OPD variants."""
import argparse
import json
from pathlib import Path

import pandas as pd
from experiment_utils import (
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from scale_validation import checked, frontier, growth, library_ids, sources
from selection_cache import separate_output
from validate_generator_scale import Scorers, generate_cell

from amp_challenge_2027.data import iter_fasta

VARIANTS = ("specialist", "anchored", "coverage")
METHODS = ("baseline", "teacher", *VARIANTS)


def inputs(root):
    original = sources(root)
    cells, records, signature = {}, {}, None
    for seed in (42, 43, 44):
        for variant in VARIANTS:
            path = root / f"opd-{variant}-seed{seed}-v1"
            checked(path, required=("run.json", "status.json", "policy/model.pt", "policy/config.json"))
            run = json.loads((path / "run.json").read_text())
            if (run.get("kind") != "anchored_opd_v1" or run["args"]["variant"] != variant or
                    run["args"]["seed"] != seed or run["sources"] != original or
                    run["teacher_cell"] != f"grpo/seed{seed}"):
                raise ValueError("OPD source identity/provenance differs")
            comparable = {"args": {k: v for k, v in run["args"].items() if k not in ("variant", "seed", "out", "list")},
                          "runtime": run["runtime"], "code": run["code"], "objective": run["objective"],
                          "coverage_protocol": run["coverage_protocol"], "sampling": run["sampling"]}
            if signature is not None and comparable != signature:
                raise ValueError("OPD settings, runtime or code differ across cells")
            signature = comparable
            cell = f"{variant}/seed{seed}"
            records[cell] = json.loads((path / "status.json").read_text())
            cells[cell] = {"checkpoint": str((path / "policy").resolve()), "root": str(path.resolve()),
                           "marker_sha256": sha256(path / "complete.json"), "sampling_seed": 60000+seed}
        for method, source_method in (("baseline", "baseline"), ("teacher", "grpo")):
            cells[f"{method}/seed{seed}"] = {**original["cells"][f"{source_method}/seed{seed}"], "sampling_seed": 60000+seed}
    return {**original, "cells": cells, "training_status": records}


def audit(dest, recipe):
    checked(dest, required=("run.json", "pool.csv", "status.json"))
    saved = json.loads((dest / "run.json").read_text())
    if saved["recipe"] != recipe or saved["cell"] != f"{dest.parent.name}/{dest.name}":
        raise ValueError("Wrong sampled cell")
    directory = dest / "audit"
    prepare_run(directory, {"sample_marker": sha256(dest / "complete.json"), "code": code_identity(),
                            "scales": recipe["scales"], "targets": [.5, .55, .6, .65], "shortlist": 2000})
    if (directory / "complete.json").exists():
        checked(directory)
        return
    frame = pd.read_csv(dest / "pool.csv")
    reference = {s for _, s in iter_fasta(Path(recipe["sources"]["reference"]))}
    rows, lengths = growth(frame, reference, recipe["scales"])
    # Pilot top100 uses all distinct non-reference draws; a complete50k run
    # uses only its constructed library, so selected candidates are members.
    ids = library_ids(frame.sequence.tolist(), reference)
    complete = len(ids) == 50000
    if complete:
        actual = [s for _, s in iter_fasta(dest / "library.fasta")]
        if actual != frame.iloc[ids].sequence.tolist():
            raise ValueError("Constructed library differs")
    summary, selected = frontier(frame.iloc[ids], reference)
    for row in summary:
        row["selection_source"] = "library50000" if complete else "pilot_pool"
    write_summary(directory / "growth.csv", rows)
    write_summary(directory / "lengths.csv", lengths)
    write_summary(directory / "frontier.csv", summary)
    names = ["run.json", "growth.csv", "lengths.csv", "frontier.csv"]
    if selected:
        write_summary(directory / "selected.csv", selected)
        names.append("selected.csv")
    mark_files(directory, "complete.json", names)


def report(root, recipe):
    tables = {k: [] for k in ("growth", "frontier", "lengths")}
    proofs, status, official, runtimes, audit_settings = {}, [], [], [], []
    for cell in sorted(recipe["sources"]["cells"]):
        dest = root / cell
        checked(dest, required=("run.json", "pool.csv", "status.json"))
        checked(dest / "audit", required=("run.json", "growth.csv", "frontier.csv", "lengths.csv"))
        sample_run = json.loads((dest / "run.json").read_text())
        audit_run = json.loads((dest / "audit/run.json").read_text())
        if (sample_run["recipe"] != recipe or sample_run["cell"] != cell or
                audit_run["sample_marker"] != sha256(dest / "complete.json")):
            raise ValueError("Mismatched sample/audit provenance")
        runtimes.append(sample_run["runtime"])
        audit_settings.append({k: v for k, v in audit_run.items() if k != "sample_marker"})
        method, seed = cell.split("/seed")
        labels = {"method": method, "seed": int(seed)}
        status.append({**labels, **json.loads((dest / "status.json").read_text()),
                       **recipe["sources"]["training_status"].get(cell, {})})
        proofs[cell] = {"sample": sha256(dest / "complete.json"), "audit": sha256(dest / "audit/complete.json")}
        for name in tables:
            tables[name].append(pd.read_csv(dest / f"audit/{name}.csv").assign(**labels))
        if (dest / "evaluation.json").exists():
            checked(dest, "evaluation.json", required=("evaluation_recipe.json", "library.fasta", "metrics.json"))
            er = json.loads((dest / "evaluation_recipe.json").read_text())
            if (er["seed"] != 2027 or er["esm_model"] != "facebook/esm2_t33_650M_UR50D" or
                    er["reference_sha256"] != sha256(Path(recipe["sources"]["reference"]))):
                raise ValueError("Official evaluation recipe differs")
            official.append({**labels, **json.loads((dest / "metrics.json").read_text())})
            proofs[cell]["evaluation"] = sha256(dest / "evaluation.json")
    if any(x != runtimes[0] for x in runtimes[1:]) or any(x != audit_settings[0] for x in audit_settings[1:]):
        raise ValueError("Evaluation runtime or audit recipes differ")
    output = root / "report"
    output.mkdir(exist_ok=True)
    write_summary(output / "status.csv", status)
    for name, values in tables.items():
        pd.concat(values, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    if official:
        write_summary(output / "official.csv", official)
    paired, matched = [], []
    all_growth = pd.concat(tables["growth"], ignore_index=True)
    metrics = ("raw_joint_yield_per_1000", "raw_evaluation_yield_per_1000", "separated_yield_per_1000",
               "raw_unique_novel_fraction", "raw_evaluation_risk_mean")
    for (seed, draws), group in all_growth.groupby(["seed", "draws"]):
        index = group.set_index("method")
        for variant in VARIANTS:
            for baseline in ("baseline", "teacher"):
                paired.append({"seed": int(seed), "draws": int(draws), "comparison": f"{variant}_minus_{baseline}",
                               **{k: float(index.loc[variant, k]-index.loc[baseline, k]) for k in metrics}})
    write_summary(output / "paired_growth.csv", paired)
    aggregate = pd.DataFrame(paired).groupby(["draws", "comparison"])[list(metrics)].agg(["mean", "std", "count"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.to_csv(output / "paired_growth_summary.csv")
    all_frontier = pd.concat(tables["frontier"], ignore_index=True)
    for (seed, target), group in all_frontier.groupby(["seed", "target_distance"]):
        index = group.set_index("method")
        for variant in VARIANTS:
            for baseline in ("baseline", "teacher"):
                a, b = index.loc[variant], index.loc[baseline]
                full = bool(a.complete and b.complete)
                difference = float(abs(a.achieved_distance-b.achieved_distance)) if full else None
                eligible = bool(full and difference <= .01)
                matched.append({"seed": int(seed), "target_distance": float(target),
                                "comparison": f"{variant}_minus_{baseline}", "both_complete": full,
                                "matched_within_001": eligible, "distance_difference": difference,
                                "activity_delta": float(a.activity-b.activity) if eligible else None,
                                "evaluation_risk_delta": float(a.evaluation_risk-b.evaluation_risk) if eligible else None})
    write_summary(output / "matched_frontier.csv", matched)
    write_json(output / "report.json", {"inputs": proofs, "official_evaluations": len(official),
        "kl_stopped_training_cells": [c for c, s in recipe["sources"]["training_status"].items() if s["kl_stopped"]],
        "deployment_approved": False, "limitations": [
            "No independent biological validation; same inspected predictor lineage",
            "Only one fresh sampling seed per teacher/student pair; descriptive comparisons",
            "Default8192 draws cannot establish50k library FBD/MMD preservation",
            "Regularized variant uses an approximate moment penalty, not a hard official-metric constraint",
            "Training cost differs; donor GRPO training cost is additional",
            "Mixture-inference and constrained-GRPO baselines are not implemented in this OPD experiment"]})
    print(pd.DataFrame(status).to_string(index=False))
    print(pd.DataFrame(paired).round(5).to_string(index=False))
    print(f"[opd-eval] reports: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "audit", "evaluate", "report"))
    parser.add_argument("--train-root", type=Path, default=Path("sweep_results"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if not 2048 <= args.draws <= 200000 or args.draws % 32:
        raise ValueError("Draws must be a multiple of32 between2048 and200000")
    pinned = inputs(args.train_root)
    if args.command == "sample":
        separate_output(args.out, [Path(pinned["evaluator"])] + [Path(p) for p in pinned["common"]["inputs"]] +
                        [Path(c["checkpoint"]) for c in pinned["cells"].values()] +
                        [Path(c["root"]) for c in pinned["cells"].values() if "root" in c])
        recipe = {"kind": "opd_evaluation_v1", "sources": pinned, "draws": args.draws,
                  "scales": sorted({n for n in (2048, 8192, 32768, args.draws) if n <= args.draws}),
                  "device": args.device, "code": code_identity(), "sampling": "full categorical mask; fixed batches32; seed60000+training seed"}
    else:
        recipe = json.loads((args.out / "run.json").read_text())
        if recipe["kind"] != "opd_evaluation_v1" or pinned != recipe["sources"]:
            raise ValueError("OPD evaluation inputs differ")
    cells = [c for c in sorted(pinned["cells"]) if (args.method is None or c.startswith(args.method+"/")) and
             (args.seed is None or c.endswith(f"seed{args.seed}"))]
    if args.list:
        print(json.dumps({"recipe": recipe, "cells": cells}, indent=2))
        return
    if args.command == "sample":
        prepare_run(args.out, recipe)
        from pilot_grpo import strict_runtime

        strict_runtime(42)
        scorers = Scorers(recipe, args.device)
        for cell in cells:
            generate_cell(args.out / cell, recipe, cell, scorers)
    elif args.command == "audit":
        for cell in cells:
            print(f"[opd-audit] {cell}", flush=True)
            audit(args.out / cell, recipe)
    elif args.command == "evaluate":
        for cell in cells:
            dest = args.out / cell
            checked(dest, required=("run.json", "pool.csv", "status.json"))
            saved = json.loads((dest / "run.json").read_text())
            if saved["recipe"] != recipe or saved["cell"] != cell:
                raise ValueError("Wrong sampled cell")
            if not json.loads((dest / "status.json").read_text())["library_complete"]:
                print(f"[opd-eval] skip {cell}: no complete50k library", flush=True)
                continue
            evaluate_library(dest, reference=Path(pinned["reference"]), esm_model="facebook/esm2_t33_650M_UR50D", device=args.device, seed=2027)
    else:
        report(args.out, recipe)


if __name__ == "__main__":
    main()
