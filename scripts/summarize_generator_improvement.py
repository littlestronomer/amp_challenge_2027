"""Paired three-seed fresh-yield results; no automatic policy promotion."""
import argparse
import json
import statistics
from pathlib import Path

from experiment_utils import mark_files, sha256, write_json, write_summary
from selection_cache import separate_output
from summarize_grpo_pilot import METRICS, RAW_METRICS, read_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    separate_output(args.out, args.runs)
    if args.out.exists():
        raise ValueError("Use a new summary directory")
    records = [read_run(p, improvement=True) for p in args.runs]
    if {rows[0]["seed"] for rows, _ in records} != {42, 43, 44} or any(sig != records[0][1] for _, sig in records[1:]):
        raise ValueError("Require seeds42/43/44 with identical method, settings, artifacts, code and runtime")
    all_rows, deltas = [], []
    for rows, _ in records:
        all_rows.extend(rows)
        arms = {r["arm"]: r for r in rows}
        optimized = arms[rows[0]["method"] + "_selection"]
        for control in ("baseline_selection_equal_eval", "baseline_selection_matched_total"):
            baseline = arms[control]
            complete = all(r["selection_complete"].lower() == "true" for r in (optimized, baseline))
            for metric in METRICS + RAW_METRICS:
                a, b = optimized[metric], baseline[metric]
                # Raw results do not depend on obtaining 100 eligible selections.
                valid = metric in RAW_METRICS or complete
                deltas.append({"seed": optimized["seed"], "comparison": optimized["method"] + "_minus_" + control,
                               "metric": metric, "delta": a-b if valid and a is not None and b is not None else None})
    summary = []
    for comparison, metric in sorted({(r["comparison"], r["metric"]) for r in deltas}):
        values = [r["delta"] for r in deltas if r["comparison"] == comparison and r["metric"] == metric and r["delta"] is not None]
        summary.append({"comparison": comparison, "metric": metric, "n": len(values),
                        "mean_delta": statistics.mean(values) if values else None,
                        "std_delta": statistics.stdev(values) if len(values) > 1 else None})
    args.out.mkdir(parents=True)
    write_summary(args.out / "per_seed.csv", all_rows)
    write_summary(args.out / "paired_deltas.csv", deltas)
    write_summary(args.out / "paired_summary.csv", summary)
    write_json(args.out / "report.json", {
        "inputs": {str(p): sha256(p / "complete.json") for p in args.runs},
        "kl_stopped_seeds": sorted({r["seed"] for r in all_rows if r["kl_stopped"]}),
        "teaching_shortfall_seeds": sorted({r["seed"] for r in all_rows if r["teaching_shortfall"]}),
        "deployment_approved": False,
        "limitations": ["Primary comparison: equal fresh draws; matched-total is a selection-budget control",
                        "Unique yield changes with draw count: do not treat unequal-draw raw yields as controlled comparisons",
                        "Predictor gates are not experimental safety; evaluator shares backbone/data lineage",
                        "Three training seeds with one fresh sampling seed each; descriptive statistics only"]})
    mark_files(args.out, "complete.json", ["per_seed.csv", "paired_deltas.csv", "paired_summary.csv", "report.json"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
