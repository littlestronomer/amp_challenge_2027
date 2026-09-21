"""Apply cheap deterministic eligibility gates to imported scored pools."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import code_identity, mark_files, prepare_run, sha256, verify_files, write_json
from amp_challenge_2027.config import PANEL_GENERA
from amp_challenge_2027.data import read_reference_set
from amp_challenge_2027.selectivity_research.contracts import load_protocol
from amp_challenge_2027.selectivity_research.eligibility import scan
from amp_challenge_2027.selectivity_research.pool import read_pool


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    pools, protocol_path, reference, out = [p.resolve() for p in
        (args.pools, args.protocol, args.reference, args.out)]
    protocol = load_protocol(protocol_path)
    if protocol.get("reference") and Path(protocol["reference"]).resolve() != reference:
        raise ValueError("Reference differs from constrained-selectivity protocol")
    if not (pools / "complete.json").exists():
        raise ValueError("Pool import is incomplete")
    seeds = [int(x) for x in protocol.get("seeds", [42, 43, 44])]
    frames = {seed: read_pool(pools / f"seed{seed}" / "pool.csv", PANEL_GENERA) for seed in seeds}
    run = {"kind": "selectivity_eligibility_v1", "code": code_identity(),
           "pools_run_sha256": sha256(pools / "run.json"), "protocol_sha256": sha256(protocol_path),
           "reference_sha256": sha256(reference), "seeds": seeds}
    print(f"[selectivity-scan] loaded {len(frames)} pools")
    if args.list:
        for seed, frame in frames.items():
            eligible, _ = scan(frame, read_reference_set(reference))
            print(f"[selectivity-scan] seed={seed} rows={len(frame)} eligible={int(eligible.sum())}")
        return 0
    prepare_run(out, run)
    if (out / "complete.json").exists():
        verify_files(out, "complete.json")
        print(f"[selectivity-scan] matching completed output already exists: {out}")
        return 0
    reference_set = read_reference_set(reference)
    manifest = {}
    write_json(out / "run.json", run)
    for seed, frame in frames.items():
        eligible, reasons = scan(frame, reference_set)
        table = pd.DataFrame({"library_index": frame["library_index"].astype(int),
                              "eligible": eligible, "reason": reasons})
        path = out / f"seed{seed}" / "eligibility.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        manifest[str(seed)] = {"rows": len(frame), "eligible": int(eligible.sum()),
                               "sha256": sha256(path)}
    write_json(out / "cells.json", manifest)
    mark_files(out, "complete.json", ["run.json", "cells.json"])
    print(f"[selectivity-scan] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
