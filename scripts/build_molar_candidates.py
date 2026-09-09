"""Build diagnostic molar-only labels. Never trains or creates a new holdout."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_label_observations import rows
from compare_top100 import checked_stage
from experiment_utils import code_identity, sha256, write_json, write_summary

from amp_challenge_2027.config import AMINO_ACIDS, MAX_LENGTH, MIN_LENGTH
from amp_challenge_2027.measurement_bounds import Bound, activity_label


def decision(row, task):
    seq = row["sequence"]
    if not MIN_LENGTH <= len(seq) <= MAX_LENGTH or set(seq) - set(AMINO_ACIDS):
        return "", "invalid_sequence"
    if str(row["eligible_target"]).lower() != "true":
        return "", "outside_targets"
    if row["syntax"] not in ("exact", "censored"):
        return "", "non_scalar_or_missing"
    if row["conversion"] != "molar":
        return "", "not_direct_molar"
    if task in ("activity", "panel"):
        label = activity_label(Bound(row["operator"], float(row["estimated_um"])), success=4 if task == "activity" else 16)
    else:
        label = row["evidence"]
    if label not in ("active", "inactive", "ambiguous"):
        raise ValueError("Unexpected evidence label")
    return label, "selected"


def build(observations, task):
    groups, events = defaultdict(list), []
    for row in observations:
        label, reason = decision(row, task)
        organism = row["panel_genus"] if task == "panel" else ""
        events.append({"task": task, "source_line": row["source_line"], "sequence": row["sequence"],
                       "organism": organism, "target": row["target"], "label": label, "reason": reason})
        if reason == "selected":
            groups[(row["sequence"], organism)].append((label, row["source_line"], row["target"]))
    candidates, grouped = [], []
    for (seq, organism), evidence in sorted(groups.items()):
        values = {e[0] for e in evidence}
        status = "conflict_masked" if {"active", "inactive"} <= values else "ambiguous_masked" if "ambiguous" in values else "retained"
        label = next(iter(values)) if status == "retained" else ""
        grouped.append({"sequence": seq, "organism": organism, "status": status, "label": label,
                        "source_lines": json.dumps([e[1] for e in evidence]),
                        "targets": json.dumps(sorted({e[2] for e in evidence}))})
        if label:
            candidates.append({"sequence": seq, "organism": organism, "label": label})
    return candidates, grouped, events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("sweep_results/label-observations-v1"))
    parser.add_argument("--prepared", type=Path, required=True, help="Existing completed family partition, used only for coverage")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    for source in (args.source.resolve(), args.prepared.resolve()):
        if out.exists() or out == source or out in source.parents or source in out.parents:
            raise ValueError("Use a new output directory separate from sources")
    marker = checked_stage(args.source, "complete.json", {"report.json", "activity_observations.csv", "hemolysis_observations.csv", "label_comparison.csv"})
    family_marker = checked_stage(args.prepared, "complete.json", {"families.csv"})
    if json.loads((args.source / "report.json").read_text()).get("kind") != "label_observation_audit_v1":
        raise ValueError("Unexpected source kind")
    families = {}
    for row in rows(args.prepared / "families.csv", ["sequence", "family", "split"]):
        if row["sequence"] in families or row["split"] not in ("train", "validation", "calibration", "test"):
            raise ValueError("Invalid family mapping")
        families[row["sequence"]] = row
    source_rows = {task: rows(args.source / f"{task}_observations.csv", ["sequence", "source_line", "evidence"])
                   for task in ("activity", "hemolysis")}
    old_rows = rows(args.source / "label_comparison.csv", ["task", "sequence", "old_label", "genus"])
    outputs, summaries = {}, {}
    for task in ("activity", "panel", "hemolysis"):
        candidate, groups, events = build(source_rows["hemolysis" if task == "hemolysis" else "activity"], task)
        old = defaultdict(set)
        for row in old_rows:
            if row["task"] == task:
                old[(row["sequence"].strip().upper(), row["genus"] if task == "panel" else "")].add(row["old_label"])
        population = []
        for row in candidate:
            key = row["sequence"], row["organism"]
            prior = old.get(key, set())
            partition = families.get(row["sequence"], {})
            population.append({**row, "old_labels": "|".join(sorted(prior)),
                               "old_label_unambiguous": len(prior) == 1 and prior <= {"active", "inactive"},
                               "family": partition.get("family", ""), "split": partition.get("split", "unmapped")})
        summaries[task] = {
            "candidate_rows": len(candidate), "unique_sequences": len({r["sequence"] for r in candidate}),
            "class_counts": dict(Counter(r["label"] for r in candidate)),
            "observation_decisions": dict(Counter(r["reason"] for r in events)),
            "group_decisions": dict(Counter(r["status"] for r in groups)),
            "old_unique_keys": len(old), "common_unambiguous_old_keys": sum(r["old_label_unambiguous"] for r in population),
            "coverage": {split: {"rows": sum(r["split"] == split for r in population),
                                 "families": len({r["family"] for r in population if r["split"] == split and r["family"]}),
                                 "class_counts": dict(Counter(r["label"] for r in population if r["split"] == split))}
                         for split in ("train", "validation", "calibration", "test", "unmapped")}}
        outputs[task] = candidate, groups, events, population
    if checked_stage(args.source, "complete.json", set(marker["files"])) != marker or checked_stage(args.prepared, "complete.json", {"families.csv"}) != family_marker:
        raise ValueError("Source changed during build")
    out.mkdir(parents=True, exist_ok=False)
    for task, tables in outputs.items():
        for name, table in zip(("candidates", "groups", "observation_decisions", "population"), tables):
            # Empty tasks are explicit in report; never pretend they are trainable.
            if table:
                write_summary(out / f"{task}_{name}.csv", table)
    report = {"kind": "molar_candidates_v1", "source_marker": marker, "family_marker": family_marker,
              "code": code_identity(), "summary": summaries, "training_authorized": False,
              "policy": {"activity_positive_um": 4, "panel_positive_um": 16, "inactive_gt_um": 32,
                         "hemolysis": "source audit explicit-band rule", "conflicts": "mask",
                         "selected_ambiguous": "mask entire sequence/output group"},
              "limitations": ["Molar-only is not verified unmodified chemistry",
                               "Excluded non-scalar/mass observations do not veto retained molar evidence",
                               "Binary unanimity is across retained measured contexts, not universal activity/safety",
                               "Existing partition used for coverage only; test already inspected",
                               "Unmapped sequences are not assigned to any split; could bridge existing families",
                               "Population tables are comparison candidates, not independent test labels",
                               "Original full provenance stays in the hash-verified source observation audit"]}
    write_json(out / "report.json", report)
    write_json(out / "complete.json", {"files": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
