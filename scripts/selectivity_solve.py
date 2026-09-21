"""Run the constrained MILP profiles on each imported hybrid pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import code_identity, mark_files, prepare_run, sha256, verify_files, write_json
from amp_challenge_2027.config import PANEL_GENERA
from amp_challenge_2027.data import write_fasta
from amp_challenge_2027.selectivity_research.contracts import load_protocol
from amp_challenge_2027.selectivity_research.pool import read_pool
from amp_challenge_2027.selectivity_research.solver import solve_pool


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--max-cut-rounds", type=int, default=20)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    pools, eligibility, protocol_path, reference, out = [p.resolve() for p in
        (args.pools, args.eligibility, args.protocol, args.reference, args.out)]
    protocol = load_protocol(protocol_path)
    seeds = [int(x) for x in protocol.get("seeds", [42, 43, 44])]
    profiles = [str(x) for x in protocol["risk_ceilings"]]
    run = {"kind": "selectivity_solution_v1", "code": code_identity(),
           "pools_run_sha256": sha256(pools / "run.json"),
           "eligibility_run_sha256": sha256(eligibility / "run.json"),
           "protocol_sha256": sha256(protocol_path), "reference_sha256": sha256(reference),
           "seeds": seeds, "profiles": profiles, "time_limit": args.time_limit,
           "mip_gap": args.mip_gap, "max_cut_rounds": args.max_cut_rounds}
    if args.list:
        print(f"[selectivity-solve] seeds={seeds} risk ceilings={profiles}; CPU-only MILP")
        return 0
    prepare_run(out, run)
    if (out / "complete.json").exists():
        verify_files(out, "complete.json")
        print(f"[selectivity-solve] matching completed output already exists: {out}")
        return 0
    write_json(out / "run.json", run)
    manifest = {}
    reference_set = set()
    from amp_challenge_2027.data import read_reference_set
    reference_set = read_reference_set(reference)
    for seed in seeds:
        frame = read_pool(pools / f"seed{seed}" / "pool.csv", PANEL_GENERA)
        eligible_table = pd.read_csv(eligibility / f"seed{seed}" / "eligibility.csv")
        if len(eligible_table) != len(frame) or not np.array_equal(
                eligible_table["library_index"].to_numpy(int), frame["library_index"].to_numpy(int)):
            raise ValueError(f"Eligibility index map differs for seed {seed}")
        eligible = eligible_table["eligible"].to_numpy(bool)
        for profile in profiles:
            result = solve_pool(frame, eligible, PANEL_GENERA, protocol, profile,
                                reference=reference_set, time_limit=args.time_limit,
                                mip_rel_gap=args.mip_gap, max_cut_rounds=args.max_cut_rounds)
            cell = out / f"seed{seed}" / f"ceiling-{profile}"
            cell.mkdir(parents=True, exist_ok=True)
            write_json(cell / "solution.json", result.as_dict())
            write_json(cell / "solver_status.json", {"status": result.status,
                "message": result.message, "metadata": result.metadata})
            write_json(cell / "constraint_margins.json", result.summary)
            files = ["solution.json", "solver_status.json", "constraint_margins.json"]
            if result.indices.size:
                rows = frame.iloc[result.indices].copy()
                rows.insert(0, "rank", np.arange(1, len(rows) + 1))
                rows.to_csv(cell / "selected.csv", index=False)
                files += ["selected.csv"]
                if result.status in {"optimal", "feasible_unproven_optimal"} and len(result.indices) == 100:
                    write_fasta(rows["sequence"].tolist(), cell / "top.fasta", header_prefix=f"seed{seed}-{profile}-seq")
                    rows.to_csv(cell / "top_scores.csv", index=False)
                    files += ["top.fasta", "top_scores.csv"]
            mark_files(cell, "complete.json", files)
            manifest[f"seed{seed}/ceiling-{profile}"] = {"status": result.status,
                "solution_sha256": sha256(cell / "solution.json"), "selected": int(result.indices.size)}
    write_json(out / "cells.json", manifest)
    mark_files(out, "complete.json", ["run.json", "cells.json"])
    print(f"[selectivity-solve] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
