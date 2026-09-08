"""CPU-only curation and four-way family split. Never changes deployed artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compare_top100 import checked_stage
from experiment_utils import (
    REPO_ROOT,
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from selection_cache import separate_output

from amp_challenge_2027.generalization import (
    FRACTIONS,
    TASKS,
    assign_families,
    audit_boundary,
    curate_labels,
    family_components,
    support_report,
)

FILES = {"dataset.json", "families.csv", "curation.json", "curation_events.json", "support.csv", "preflight.json"}
BACKBONE = "facebook/esm2_t12_35M_UR50D"


def verify_hashes(files):
    for path, expected in files.items():
        if not Path(path).is_file() or sha256(Path(path)) != expected:
            raise ValueError(f"Pinned input changed or missing: {path}")


def validate_protocol(protocol):
    if protocol.get("kind") != "reward_generalization_protocol_v1" or set(protocol.get("tasks", {})) != set(TASKS):
        raise ValueError("Require a three-task generalization protocol")
    if protocol.get("fractions") != list(FRACTIONS) or not .5 <= protocol.get("similarity_threshold", 0) <= 1:
        raise ValueError("Invalid four-way split recipe")
    if (not isinstance(protocol.get("split_seed"), int) or protocol.get("architectures") != ["linear", "mlp"]
            or len(protocol.get("training_seeds", [])) != 3
            or len(set(protocol["training_seeds"])) != 3
            or not all(isinstance(s, int) and s >= 0 for s in protocol["training_seeds"])):
        raise ValueError("Require linear/MLP controls and three predeclared seeds")
    for field in ("epochs", "patience", "head_batch_size", "bootstrap_replicates"):
        if not isinstance(protocol.get(field), int) or protocol[field] < 1:
            raise ValueError(f"Invalid protocol value: {field}")
    for field in ("learning_rate", "weight_decay"):
        value = protocol.get(field)
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"Invalid protocol value: {field}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "experiments/reward_generalization_v1.json")
    parser.add_argument("--reconstruction", type=Path, default=Path("sweep_results/reward-reconstruction-v1"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    reconstruction = args.reconstruction.resolve()
    checked_stage(reconstruction, "complete.json", {"results.csv"})
    old_run = json.loads((reconstruction / "run.json").read_text())
    if old_run.get("kind") != "reward_validation_reconstruction_v1":
        raise ValueError("Require the completed reconstruction to pin the backbone revision")
    inputs = {str(p.resolve()): sha256(p) for p in
              (args.protocol, reconstruction / "run.json", reconstruction / "complete.json")}
    revisions = set()
    for task in TASKS:
        root = reconstruction / task
        checked_stage(root, "complete.json", {"backbone.json"})
        info = json.loads((root / "backbone.json").read_text())
        revision = info.get("resolved_revision", "")
        if info.get("model") != BACKBONE or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Require an exact 35M backbone commit from completed inference")
        revisions.add(revision)
        for p in (root / "backbone.json", root / "complete.json"):
            inputs[str(p)] = sha256(p)
    if len(revisions) != 1:
        raise ValueError("Backbone revisions differ across tasks; do not silently mix representations")
    for spec in protocol["tasks"].values():
        inputs[str((REPO_ROOT / spec["data"]).resolve())] = spec["sha256"]
    verify_hashes(inputs)
    separate_output(args.out, [reconstruction, *(Path(p) for p in inputs)])
    # Never let a benchmark be accidentally placed inside the deployed checkpoint tree.
    separate_output(args.out, [REPO_ROOT / "checkpoint"])
    recipe = {"kind": "reward_generalization_data_v1", "code": code_identity(), "inputs": inputs,
              "protocol": protocol, "backbone": BACKBONE, "revision": revisions.pop(),
              "label_policy": "unanimous sequence/output labels; conflicts masked, same-label repeats collapsed",
              "split_policy": "joint three-task connected components, including excluded valid sequences as bridges",
              "scope": "internal family-held-out benchmark for freshly trained heads; not an external or prospective test"}
    prepare_run(args.out, recipe)
    if (args.out / "complete.json").exists():
        checked_stage(args.out, "complete.json", FILES | {"run.json"})
        print(f"[generalization] verified prepared dataset: {args.out}")
        return
    tasks, inventories, events, union = {}, {}, [], set()
    for task in TASKS:
        records, excluded, bridges, info = curate_labels(REPO_ROOT / protocol["tasks"][task]["data"], task)
        if not records:
            raise ValueError(f"No retained labels: {task}")
        tasks[task] = {"records": records, "outputs": info["outputs"]}
        inventories[task] = info
        events.extend(excluded)
        union.update(bridges)
    def progress(message):
        print(f"[generalization] {message}", flush=True)
    groups = family_components(sorted(union), protocol["similarity_threshold"], progress=progress)
    assignment = assign_families(groups, protocol["split_seed"])
    boundary = audit_boundary(assignment, protocol["similarity_threshold"], progress=progress)
    support = []
    for task, data in tasks.items():
        for rec in data["records"]:
            rec.update({"family": groups[rec["sequence"]], "split": assignment[rec["sequence"]]})
        support.extend(support_report(data["records"], data["outputs"], task))
    issues = []
    for task in TASKS:
        rows = [r for r in support if r["task"] == task]
        if any(not r["both_classes"] for r in rows if r["split"] == "train"):
            issues.append(f"{task}: at least one training output lacks both classes")
        if not any(r["both_classes"] for r in rows if r["split"] == "validation"):
            issues.append(f"{task}: no defined validation AUROC for checkpoint selection")
        if any(r["sequences"] == 0 for r in rows):
            issues.append(f"{task}: empty partition")
    sizes = Counter(groups.values())
    preflight = {"eligible_for_training": not issues, "issues": issues, "boundary": boundary,
                 "families": len(sizes), "largest_family_sequences": max(sizes.values()),
                 "unique_graph_sequences": len(groups), "partition_counts": dict(Counter(assignment.values())),
                 "target_fractions_are_approximate": True,
                 "test_policy": "No fitting/model selection on test. Separate --unlock-test invocation after all fits are sealed."}
    verify_hashes(inputs)
    write_json(args.out / "dataset.json", {"tasks": tasks})
    write_json(args.out / "curation.json", inventories)
    write_json(args.out / "curation_events.json", {"events": events})
    write_json(args.out / "preflight.json", preflight)
    write_summary(args.out / "families.csv", [{"sequence": s, "family": groups[s], "split": assignment[s]} for s in sorted(union)])
    write_summary(args.out / "support.csv", support)
    mark_files(args.out, "complete.json", sorted(FILES | {"run.json"}))
    print(json.dumps(preflight, indent=2))
    if issues:
        raise SystemExit("Preparation recorded, but training blocked by support. Inspect; do not search for a favorable test split.")


if __name__ == "__main__":
    main()
