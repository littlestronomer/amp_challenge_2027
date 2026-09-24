"""Summarize constrained-selectivity profiles without promoting an output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import (
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
)
from selection_cache import separate_output
from selectivity_artifacts import checked_stage

from amp_challenge_2027.selectivity_research.solver import SOLVER_VERSION


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    solutions, protocol, out = [p.resolve() for p in (args.solutions, args.protocol, args.out)]
    separate_output(out, [solutions, protocol] + ([args.pools] if args.pools else []))
    source_run, source_cells = checked_stage(solutions, "selectivity_solution_v1")
    if source_run.get("solver_version") != SOLVER_VERSION:
        raise ValueError(
            "Legacy solver outputs are invalid for inference; rerun with the repaired selector"
        )
    if source_run["protocol_sha256"] != sha256(protocol):
        raise ValueError("Report protocol differs from solve protocol")
    run = {
        "kind": "selectivity_report_v1",
        "code": code_identity(),
        "solutions_run_sha256": sha256(solutions / "run.json"),
        "solutions_complete_sha256": sha256(solutions / "complete.json"),
        "protocol_sha256": sha256(protocol),
    }
    if args.pools:
        checked_stage(args.pools.resolve(), "selectivity_pool_v1")
        if source_run["pools_run_sha256"] != sha256(args.pools / "run.json"):
            raise ValueError("Report pool differs from solve pool")
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
    expected = {f"seed{s}/ceiling-{p}" for s in source_run["seeds"] for p in source_run["profiles"]}
    if set(source_cells) != expected:
        raise ValueError("Solution stage is missing expected seed/profile cells")
    for cell in sorted(source_cells):
        path = solutions / cell / "solution.json"
        verify_files(path.parent, "complete.json")
        seed = path.parent.parent.name.removeprefix("seed")
        profile = path.parent.name.removeprefix("ceiling-")
        value = json.loads(path.read_text())
        meta = value["metadata"]
        row = {
            "seed": int(seed),
            "risk_ceiling": float(profile),
            "status": value["status"],
            "message": value["message"],
            "origin": meta.get("solution_origin", "none"),
            "improved": meta.get("improved_vs_incumbent", False),
            "termination_status": meta.get("termination_status", value["status"]),
            **value.get("summary", {}),
        }
        rows.append(row)
    if not rows:
        raise ValueError("No solution cells found")
    import csv

    fields = sorted({key for row in rows for key in row})
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Constrained selectivity report",
        "",
        "This is a surrogate-score audit; it does not establish wet-lab activity or safety.",
        "",
        "| Seed | Risk ceiling | Status | Origin | Risk mean | Activity mean | Risk delta |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]

    def number(row, field):
        return f"{row[field]:.4f}" if field in row else "—"

    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['risk_ceiling']:.2f} | {row['status']} | "
            f"{row['origin']} | {number(row, 'risk_mean')} | {number(row, 'activity_mean')} | "
            f"{number(row, 'risk_delta_vs_incumbent')} |"
        )
    lines += [
        "",
        "Negative risk delta means lower predicted risk than the frozen incumbent.",
        "incumbent_fallback is a baseline recovery, not an optimization gain. unknown_timeout and unknown_cut_limit do not establish infeasibility.",
        "",
        "Recommendation: retain the incumbent pending review across all three seeds and independent predictor evidence.",
        "",
        "## Solver messages",
        "",
    ]
    lines += [f"- seed {r['seed']}, ceiling {r['risk_ceiling']}: {r['message']}" for r in rows]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    decision = {
        "recommendation": "retain_incumbent",
        "reason": "No automatic promotion is permitted",
        "rows": len(rows),
    }
    write_json(out / "decision.json", decision)
    write_json(out / "run.json", run)
    mark_files(out, "complete.json", ["run.json", "comparison.csv", "REPORT.md", "decision.json"])
    print(f"[selectivity-report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
