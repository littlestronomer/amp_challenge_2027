"""Read and verify frozen research artifacts, including nested data files."""

import json
from pathlib import Path

from experiment_utils import sha256, verify_files


def checked_stage(root: Path, kind: str) -> tuple[dict, dict]:
    if not verify_files(root, "complete.json"):
        raise ValueError(f"Incomplete stage: {root}")
    marker = json.loads((root / "complete.json").read_text())
    if not {"run.json", "cells.json"} <= marker["files"].keys():
        raise ValueError(f"Missing manifest hashes: {root}")
    run = json.loads((root / "run.json").read_text())
    if run.get("kind") != kind:
        raise ValueError(f"Unexpected stage kind: {root}")
    cells = json.loads((root / "cells.json").read_text())
    for key, item in cells.items():
        if kind == "selectivity_pool_v1":
            if not key.isdigit():
                raise ValueError("Invalid pool seed")
            path, expected = root / f"seed{key}/pool.csv", item["pool_sha256"]
        elif kind == "selectivity_eligibility_v1":
            if not key.isdigit():
                raise ValueError("Invalid eligibility seed")
            path, expected = root / f"seed{key}/eligibility.csv", item["sha256"]
        else:
            cell = (root / key).resolve()
            if root.resolve() not in cell.parents:
                raise ValueError("Unsafe solution cell path")
            if not verify_files(cell, "complete.json"):
                raise ValueError(f"Incomplete solution cell: {cell}")
            if item.get("marker_sha256") != sha256(cell / "complete.json"):
                raise ValueError(f"Solution marker changed: {cell}")
            path, expected = cell / "solution.json", item["solution_sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact changed: {path}")
    return run, cells
