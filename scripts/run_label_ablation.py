"""Prepare and run fixed-population development ablations. No test command exists."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from audit_label_observations import rows
from benchmark_reward_generalization import embedding_cache, load_prepared, task_arrays
from compare_top100 import checked_stage
from experiment_utils import (
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from selection_cache import separate_output

from amp_challenge_2027.generalization import calibrate_temperature, support_report
from amp_challenge_2027.reward_audit import prediction_report
from amp_challenge_2027.reward_benchmark import fit_head, predict_head

ARMS = ("original", "subset_original", "subset_candidate", "full_candidate")
DEV = ("train", "validation", "calibration")
PREP_FILES = {"manifest.json", "dataset.json", "support.csv", "transitions.csv", "preflight.json"}
HEAD_FILES = {"head.pt", "history.csv", "validation.npy", "calibration.npy"}


def normalized_records(records):
    """JSON family IDs are integers; CSV IDs are strings. Preserve identity."""
    result = copy.deepcopy(records)
    for rec in result:
        family = rec["family"]
        if isinstance(family, bool) or not isinstance(family, (str, int)) or str(family) == "":
            raise ValueError(f"Invalid family identifier: {family!r}")
        rec["family"] = str(family)
    return result


def candidate_records(candidate_rows, outputs, family_map, task):
    records = {}
    for row in candidate_rows:
        seq = row["sequence"]
        if seq not in family_map:
            raise ValueError("Unmapped candidate sequence: do not extend existing family splits")
        assignment = family_map[seq]
        if assignment["split"] == "test":
            continue
        if assignment["split"] not in DEV or row["label"] not in ("active", "inactive"):
            raise ValueError("Invalid candidate label or split")
        output = row["organism"] if task == "panel" else outputs[0]
        if output not in outputs:
            raise ValueError(f"Unknown output {output}")
        rec = records.setdefault(seq, {"sequence": seq, "split": assignment["split"], "family": assignment["family"],
                                       "targets": [0] * len(outputs), "mask": [0] * len(outputs)})
        j = outputs.index(output)
        if rec["mask"][j]:
            raise ValueError("Duplicate candidate sequence/output")
        rec["targets"][j], rec["mask"][j] = int(row["label"] == "active"), 1
    return [records[s] for s in sorted(records)]


def make_arms(old_records, candidate, outputs):
    old = {r["sequence"]: r for r in normalized_records(old_records) if r["split"] in DEV}
    candidate = normalized_records(candidate)
    common_old, common_new, transitions = [], [], []
    for rec in candidate:
        previous = old.get(rec["sequence"])
        if previous is None:
            continue
        if (previous["split"], previous["family"]) != (rec["split"], rec["family"]):
            raise ValueError(f"Partition differs for {rec['sequence']}: "
                             f"original=({previous['split']}, {previous['family']!r}), "
                             f"candidate=({rec['split']}, {rec['family']!r})")
        mask = [int(a and b) for a, b in zip(previous["mask"], rec["mask"], strict=True)]
        if not any(mask):
            continue
        for dest, source in ((common_old, previous), (common_new, rec)):
            item = copy.deepcopy(source)
            item["mask"] = mask.copy()
            dest.append(item)
        for j, output in enumerate(outputs):
            if mask[j]:
                transitions.append({"split": rec["split"], "output": output,
                                    "old": previous["targets"][j], "candidate": rec["targets"][j]})
    # Every arm uses identical candidate-defined validation/calibration labels.
    evaluation = [r for r in common_new if r["split"] != "train"]
    sources = dict(zip(ARMS, (list(old.values()), common_old, common_new, candidate), strict=True))
    arms = {arm: {"outputs": outputs, "records": sorted(copy.deepcopy(
        [r for r in source if r["split"] == "train"] + evaluation), key=lambda r: r["sequence"])}
        for arm, source in sources.items()}
    return arms, transitions


def prepare(args):
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Duplicate tasks are not allowed")
    separate_output(args.out, [args.prepared, args.candidates])
    if args.out.exists():
        raise ValueError("Preparation needs a new directory")
    recipe, old = load_prepared(args.prepared)
    marker = checked_stage(args.candidates, "complete.json", {"report.json"} | {f"{t}_candidates.csv" for t in args.tasks})
    candidate_report = json.loads((args.candidates / "report.json").read_text())
    family_marker = checked_stage(args.prepared, "complete.json", {"families.csv"})
    if candidate_report.get("kind") != "molar_candidates_v1" or candidate_report["family_marker"] != family_marker:
        raise ValueError("Candidate family source differs from original prepared dataset")
    family_map = {r["sequence"]: r for r in rows(args.prepared / "families.csv", ["sequence", "family", "split"])}
    tasks, support, transitions = {}, [], []
    for task in args.tasks:
        outputs = old[task]["outputs"]
        candidate = candidate_records(rows(args.candidates / f"{task}_candidates.csv", ["sequence", "organism", "label"]), outputs, family_map, task)
        tasks[task], changes = make_arms(old[task]["records"], candidate, outputs)
        for arm, data in tasks[task].items():
            for item in support_report(data["records"], outputs, task):
                if item["split"] in DEV:
                    # Per-output family counts, not families with only other outputs observed.
                    j = outputs.index(item["output"])
                    item["families"] = len({r["family"] for r in data["records"] if r["split"] == item["split"] and r["mask"][j]})
                    support.append({"arm": arm, **item})
        counts = Counter((r["split"], r["output"], r["old"], r["candidate"]) for r in changes)
        transitions.extend({"task": task, "split": s, "output": o, "old": a, "candidate": b, "count": n}
                           for (s, o, a, b), n in sorted(counts.items()))
    failures = [r for r in support if not r["both_classes"]]
    protocol = copy.deepcopy(recipe["protocol"])
    protocol["architectures"] = ["mlp"]
    manifest = {"kind": "paired_label_ablation_v1", "backbone": recipe["backbone"], "revision": recipe["revision"],
                "protocol": protocol, "tasks": args.tasks, "arms": list(ARMS), "code": code_identity(),
                "candidate_marker": marker, "original_marker": family_marker,
                "evaluation": "common original-observed/candidate-observed outputs; candidate labels fixed across all arms",
                "test_predictions_permitted": False, "validation_is_development": True,
                "baseline": "fresh original-training arm, not historical deployed-model score",
                "changes_to_class_weights": "recomputed from each arm training labels by existing fitter"}
    args.out.mkdir(parents=True)
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "dataset.json", tasks)
    write_summary(args.out / "support.csv", support)
    # Ensure even an empty transition table is represented without a missing seal input.
    if transitions:
        write_summary(args.out / "transitions.csv", transitions)
    else:
        raise ValueError("No common observations: ablation cannot be prepared")
    write_json(args.out / "preflight.json", {"eligible": not failures, "failures": failures,
               "support_rule": "both classes in every output/arm/development partition; no automatic dropping"})
    mark_files(args.out, "complete.json", sorted(PREP_FILES))
    print(json.dumps({"eligible": not failures, "failures": failures, "fits": len(args.tasks) * 4 * 3,
                      "transition_counts": transitions}, indent=2))


def train(args):
    separate_output(args.out, [args.prepared])
    checked_stage(args.prepared, "complete.json", PREP_FILES)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if manifest.get("kind") != "paired_label_ablation_v1" or not json.loads((args.prepared / "preflight.json").read_text())["eligible"]:
        raise ValueError("Preparation failed support gate")
    tasks = json.loads((args.prepared / "dataset.json").read_text())
    if set(tasks) != set(manifest["tasks"]):
        raise ValueError("Task inventory differs")
    for arms in tasks.values():
        if set(arms) != set(ARMS):
            raise ValueError("Arm inventory differs")
        evaluations = []
        assignments, families = {}, {}
        for data in arms.values():
            for r in data["records"]:
                if r["split"] not in DEV:
                    raise ValueError("Test records forbidden in ablation training")
                for mapping, key in ((assignments, r["sequence"]), (families, r["family"])):
                    if mapping.setdefault(key, r["split"]) != r["split"]:
                        raise ValueError("Cross-partition sequence/family")
            evaluations.append([r for r in data["records"] if r["split"] != "train"])
        if any(e != evaluations[0] for e in evaluations[1:]):
            raise ValueError("Evaluation population/labels differ between arms")
    run = {"kind": "paired_label_ablation_training_v1", "manifest": manifest, "code": code_identity(),
           "prepared_marker": sha256(args.prepared / "complete.json"), "device": args.device, "batch_size": args.batch_size}
    if args.list:
        print(json.dumps(run, indent=2))
        return
    import torch

    prepare_run(args.out, run)
    embeddings = embedding_cache(args.out, {f"{t}/{a}": data for t, arms in tasks.items() for a, data in arms.items()},
                                 DEV, manifest, device=args.device, batch_size=args.batch_size)
    summary, inventory, seed_summary = [], {}, []
    for task, arms in tasks.items():
        for arm, data in arms.items():
            tx, ty, tm = task_arrays(data, "train", embeddings)
            vx, vy, vm = task_arrays(data, "validation", embeddings)
            cx, cy, cm = task_arrays(data, "calibration", embeddings)
            val_all, cal_all = [], []
            for seed in manifest["protocol"]["training_seeds"]:
                cell = f"{task}/{arm}/seed{seed}"
                dest = args.out / cell
                if (dest / "complete.json").exists():
                    checked_stage(dest, "complete.json", HEAD_FILES)
                    val, cal = np.load(dest / "validation.npy", allow_pickle=False), np.load(dest / "calibration.npy", allow_pickle=False)
                else:
                    print(f"[ablation] fitting {cell}", flush=True)
                    state, history, val = fit_head(tx, ty, tm, vx, vy, vm, protocol=manifest["protocol"], seed=seed, architecture="mlp", device=args.device)
                    cal = predict_head(state, cx, len(data["outputs"]), "mlp", device=args.device)
                    dest.mkdir(parents=True, exist_ok=True)
                    torch.save(state, dest / "head.pt")
                    write_summary(dest / "history.csv", history)
                    np.save(dest / "validation.npy", val, allow_pickle=False)
                    np.save(dest / "calibration.npy", cal, allow_pickle=False)
                    mark_files(dest, "complete.json", sorted(HEAD_FILES))
                if val.shape != vy.shape or cal.shape != cy.shape or not np.isfinite(val).all() or not np.isfinite(cal).all():
                    raise ValueError("Invalid cached predictions")
                val_all.append(val)
                cal_all.append(cal)
                seed_metrics, _, _ = prediction_report(val, vy, vm, ty, tm, temperature=1., outputs=data["outputs"])
                seed_summary.append({"task": task, "arm": arm, "seed": seed, **seed_metrics["uncalibrated"]})
                inventory[cell] = sha256(dest / "complete.json")
            val_mean, cal_mean = np.mean(val_all, axis=0), np.mean(cal_all, axis=0)
            calibration = calibrate_temperature(cal_mean, cy, cm)
            metrics, per_output, bins = prediction_report(val_mean, vy, vm, ty, tm, temperature=calibration["temperature"], outputs=data["outputs"])
            dest = args.out / task / arm
            write_json(dest / "calibration.json", calibration)
            write_json(dest / "validation_metrics.json", metrics)
            write_summary(dest / "validation_per_output.csv", per_output)
            write_summary(dest / "validation_reliability.csv", bins)
            mark_files(dest, "complete.json", ["calibration.json", "validation_metrics.json", "validation_per_output.csv", "validation_reliability.csv"])
            inventory[f"{task}/{arm}"] = sha256(dest / "complete.json")
            summary.append({"task": task, "arm": arm, "temperature": calibration["temperature"], **metrics["stored_temperature"]})
    checked_stage(args.prepared, "complete.json", PREP_FILES)
    if sha256(args.prepared / "complete.json") != run["prepared_marker"]:
        raise ValueError("Prepared inputs changed during fit")
    deltas = []
    for task in tasks:
        by_arm = {r["arm"]: r for r in summary if r["task"] == task}
        for candidate, base in (("subset_original", "original"), ("subset_candidate", "subset_original"), ("full_candidate", "subset_candidate")):
            deltas.append({"task": task, "comparison": f"{candidate}_minus_{base}",
                           **{key: by_arm[candidate][key] - by_arm[base][key] for key in by_arm[base]
                              if key.startswith("macro_") and by_arm[candidate][key] is not None and by_arm[base][key] is not None}})
    write_summary(args.out / "validation_summary.csv", summary)
    write_summary(args.out / "validation_deltas.csv", deltas)
    write_summary(args.out / "validation_seeds.csv", seed_summary)
    write_json(args.out / "model_inventory.json", inventory)
    write_json(args.out / "embedding_inventory.json", {s: sha256(args.out / "embeddings" / s / "complete.json") for s in DEV})
    mark_files(args.out, "complete.json", ["run.json", "validation_summary.csv", "validation_deltas.csv", "validation_seeds.csv", "model_inventory.json", "embedding_inventory.json"])
    print("[ablation] complete; development validation only, no test predictions or deployment promotion")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--prepared", type=Path, default=Path("sweep_results/reward-generalization-data-v1"))
    prep.add_argument("--candidates", type=Path, default=Path("sweep_results/molar-candidates-v1"))
    prep.add_argument("--tasks", nargs="+", choices=("activity", "panel", "hemolysis"), default=["panel", "hemolysis"])
    prep.add_argument("--out", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--prepared", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--batch-size", type=int, default=128)
    fit.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
