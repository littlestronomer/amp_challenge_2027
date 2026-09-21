"""Materialize a provenance-bound scored pool for constrained selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_selectivity import _read_risk_cache as read_risk_cache
from experiment_utils import code_identity, mark_files, prepare_run, sha256, verify_files, write_json
from selection_cache import load_source, separate_output

from amp_challenge_2027.config import PANEL_GENERA
from amp_challenge_2027.selectivity_research.pool import build_rows, write_pool


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--risk-cache", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    source, risk_cache, cache_protocol, reference, out = [p.resolve() for p in
        (args.source, args.risk_cache, args.cache_protocol, args.reference, args.out)]
    separate_output(out, [source, risk_cache, cache_protocol, reference])
    identity, cells, _ = load_source(source, reference)
    recipe, risks = read_risk_cache(risk_cache, cells, identity, cache_protocol, reference)
    hybrid = sorted((c for c in cells if c["case"] == "hybrid"), key=lambda c: c["seed"])
    if [c["seed"] for c in hybrid] != [42, 43, 44]:
        raise ValueError("Expected hybrid cells for seeds 42, 43, 44")
    run = {"kind": "selectivity_pool_v1", "code": code_identity(),
           "source_run_sha256": identity["run_sha256"], "source_root": str(source),
           "risk_cache_run_sha256": sha256(risk_cache / "run.json"),
           "cache_protocol_sha256": sha256(cache_protocol), "reference_sha256": sha256(reference),
           "genera": PANEL_GENERA, "seeds": [42, 43, 44],
           "source_identity": identity, "risk_recipe": recipe}
    print(f"[selectivity-pool] verified {len(hybrid)} hybrid cells and aligned risk arrays")
    if args.list:
        for cell in hybrid:
            print(f"[selectivity-pool] seed={cell['seed']} library={len(cell['sequences'])}")
        return 0
    prepare_run(out, run)
    if (out / "complete.json").exists():
        verify_files(out, "complete.json")
        print(f"[selectivity-pool] matching completed output already exists: {out}")
        return 0
    write_json(out / "source_identity.json", identity)
    cell_manifest = {}
    for cell in hybrid:
        directory = out / f"seed{cell['seed']}"
        write_pool(directory / "pool.csv", build_rows(cell, risks[cell["seed"]], PANEL_GENERA))
        cell_manifest[str(cell["seed"])] = {"pool_sha256": sha256(directory / "pool.csv"),
                                              "library_sha256": cell["record"]["library_sha256"]}
    write_json(out / "cells.json", cell_manifest)
    mark_files(out, "complete.json", ["run.json", "source_identity.json", "cells.json"])
    print(f"[selectivity-pool] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
