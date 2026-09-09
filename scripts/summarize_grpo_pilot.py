"""Compare the three prespecified deterministic pilot seeds, without selecting a winner."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from experiment_utils import sha256, verify_files, write_json, write_summary
from selection_cache import separate_output

ARMS = {"baseline_activity_control", "baseline_selection_equal_eval", "baseline_selection_matched_total", "grpo_selection"}
METRICS = ("activity", "reward_risk", "evaluation_risk", "selected_mean_pairwise_distance", "unique_fraction", "selected_mean_length")
RAW_METRICS = ("raw_reward_yield_per_1000", "raw_evaluation_yield_per_1000", "raw_joint_yield_per_1000",
               "raw_unique_novel_fraction", "raw_evaluation_risk_mean", "raw_evaluation_risk_p75",
               "raw_prefix256_pairwise_distance", "raw_mean_length", "pool_activity_mean", "pool_risk_mean")


def read_run(root, improvement=False):
    marker = json.loads((root / "complete.json").read_text())
    if not {"run.json", "summary.csv", "status.json"} <= set(marker["files"]):
        raise ValueError("Incomplete pilot marker")
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in marker["files"]):
        raise ValueError("Invalid completion path")
    verify_files(root, "complete.json")
    run = json.loads((root / "run.json").read_text())
    kind = "generator_improvement_pilot_v1" if improvement else "grpo_pilot_v2"
    if run.get("kind") != kind or run.get("runtime", {}).get("warn_only") is not False or run["runtime"].get("attention") != "math_only":
        raise ValueError("Require corrected deterministic v2 pilot; do not mix with v1")
    optimized = "raft_selection" if improvement and run["args"]["method"] == "raft" else "grpo_selection"
    expected_arms = (ARMS - {"grpo_selection"}) | {optimized}
    with (root / "summary.csv").open() as handle:
        data = list(csv.DictReader(handle))
    if len(data) != 4 or {r["arm"] for r in data} != expected_arms:
        raise ValueError("Missing/duplicate comparison arms")
    indexed = {r["arm"]: r for r in data}
    status = json.loads((root / "status.json").read_text())
    if int(indexed["baseline_selection_matched_total"]["draws"]) != int(indexed[optimized]["draws"]) + status["training_draws"]:
        raise ValueError("Matched-total scoring budget differs")
    for row in data:
        row["seed"] = run["args"]["seed"]
        for key in METRICS + (RAW_METRICS if improvement else ()):
            row[key] = float(row[key]) if row[key] else None
            if row[key] is not None and not math.isfinite(row[key]):
                raise ValueError("Nonfinite summary metric")
        row["kl_stopped"] = status["kl_stopped"]
        if improvement:
            row["teaching_shortfall"] = status.get("teaching_shortfall", False)
            row["training_draws"] = status["training_draws"]
            row["method"] = run["args"]["method"]
    signature = {"args": {k: v for k, v in run["args"].items() if k not in ("seed", "out", "list")},
                 "inputs": run["inputs"], "code": run["code"], "runtime": run["runtime"],
                 "configs": run["configs"], "revision": run["revision"], "evaluator_marker": run["evaluator_marker"],
                 "objective": run["objective"], "sampling": run["sampling"],
                 "kl_beta": run["kl_beta"], "clip": run["clip"], "update_epochs": run["update_epochs"]}
    return data, signature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    separate_output(args.out, args.runs)
    if args.out.exists():
        raise ValueError("Use a new summary directory")
    all_rows, signatures = [], []
    for root in args.runs:
        data, signature = read_run(root)
        all_rows.extend(data)
        signatures.append(signature)
    if {r["seed"] for r in all_rows} != {42, 43, 44} or any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Need seeds42/43/44 with identical code, artifacts, settings and runtime")
    deltas = []
    for seed in (42, 43, 44):
        arms = {r["arm"]: r for r in all_rows if r["seed"] == seed}
        for baseline in ("baseline_selection_equal_eval", "baseline_selection_matched_total"):
            complete = all(arms[name]["selection_complete"].lower() == "true" for name in (baseline, "grpo_selection"))
            deltas.append({"seed": seed, "comparison": "grpo_minus_" + baseline,
                           "selections_complete": complete,
                           **{key: arms["grpo_selection"][key] - arms[baseline][key]
                              if complete and arms["grpo_selection"][key] is not None and arms[baseline][key] is not None else None for key in METRICS}})
    statistics_rows = []
    for comparison in sorted({r["comparison"] for r in deltas}):
        for metric in METRICS:
            values = [r[metric] for r in deltas if r["comparison"] == comparison and r[metric] is not None]
            statistics_rows.append({"comparison": comparison, "metric": metric, "n": len(values),
                                    "mean_delta": statistics.mean(values) if values else None,
                                    "std_delta": statistics.stdev(values) if len(values) > 1 else None})
    args.out.mkdir(parents=True)
    write_summary(args.out / "per_seed.csv", all_rows)
    write_summary(args.out / "paired_deltas.csv", deltas)
    write_summary(args.out / "paired_summary.csv", statistics_rows)
    write_json(args.out / "report.json", {"source_markers": {str(p): sha256(p / "complete.json") for p in args.runs},
                "all_selections_complete": all(r["selection_complete"].lower() == "true" for r in all_rows),
                "limitations": ["Three-seed descriptive comparison, not a significance test",
                                "Incomplete selections remain visible; do not treat partial sets as top100 successes",
                                "Evaluator shares data/backbone lineage; no biological safety claim"]})
    print(json.dumps(statistics_rows, indent=2))
    print(json.dumps({"all_selections_complete": all(r["selection_complete"].lower() == "true" for r in all_rows),
                      "kl_stopped_seeds": sorted({r["seed"] for r in all_rows if r["kl_stopped"]})}, indent=2))
    print("[pilot-summary] Inspect per_seed.csv for selection shortfalls and KL stops; no automatic promotion")


if __name__ == "__main__":
    main()
