"""Summarize constrained-selectivity profiles without promoting an output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import code_identity, mark_files, prepare_run, sha256, verify_files, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    solutions, protocol, out = [p.resolve() for p in (args.solutions, args.protocol, args.out)]
    run = {"kind": "selectivity_report_v1", "code": code_identity(),
           "solutions_run_sha256": sha256(solutions / "run.json"),
           "protocol_sha256": sha256(protocol)}
    if args.pools:
        run["pools_run_sha256"] = sha256(args.pools.resolve() / "run.json")
    if args.list:
        print(f"[selectivity-report] input={solutions}")
        return 0
    prepare_run(out, run)
    if (out / "complete.json").exists():
        verify_files(out, "complete.json")
        print(f"[selectivity-report] matching completed output already exists: {out}")
        return 0
    rows = []
    for path in sorted(solutions.glob("seed*/ceiling-*/solution.json")):
        verify_files(path.parent, "complete.json")
        seed = path.parent.parent.name.removeprefix("seed")
        profile = path.parent.name.removeprefix("ceiling-")
        value = json.loads(path.read_text())
        row = {"seed": int(seed), "risk_ceiling": float(profile), "status": value["status"],
               "message": value["message"], **value.get("summary", {})}
        rows.append(row)
    if not rows:
        raise ValueError("No solution cells found")
    import csv
    fields = sorted({key for row in rows for key in row})
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Constrained selectivity report", "", "This is a surrogate-score audit; it does not establish wet-lab activity or safety.", "",
             "| Seed | Risk ceiling | Status | Risk mean | Activity mean | Pairwise distance |", "|---:|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['risk_ceiling']:.2f} | {row['status']} | "
                     f"{row.get('risk_mean', float('nan')):.4f} | {row.get('activity_mean', float('nan')):.4f} | "
                     f"{row.get('pairwise_distance_mean', float('nan')):.4f} |" )
    lines += ["", "Recommendation: retain the incumbent unless a profile is independently reviewed and passes the predeclared constraints across all seeds."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    decision = {"recommendation": "retain_incumbent", "reason": "No automatic promotion is permitted", "rows": len(rows)}
    write_json(out / "decision.json", decision)
    write_json(out / "run.json", run)
    mark_files(out, "complete.json", ["run.json", "comparison.csv", "REPORT.md", "decision.json"])
    print(f"[selectivity-report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
