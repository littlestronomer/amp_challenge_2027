"""Inventory deployed heads and find exact training-member matches on CPU.

No backbone downloads, calibration fitting, guessed split seeds, or safety claims.
A tensor match establishes head identity, not a training/validation data lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import compare_top100 as baseline
from experiment_utils import code_identity, mark_files, prepare_run, sha256, write_json
from selection_cache import separate_output


def tensor_fingerprint(state: dict) -> str:
    """Serialization-independent identity; distinguish key, shape, dtype and tensor bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        description = json.dumps([name, str(value.dtype), list(value.shape)])
        digest.update(description.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metadata_issue(cfg: dict, task: str) -> str | None:
    temperature = cfg.get("temperature")
    if (cfg.get("esm_model") != baseline.BACKBONE or cfg.get("checkpoint_format") != "head-only"
            or cfg.get("unfreeze_layers") != 0 or cfg.get("task") != task
            or not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        return "Require explicit frozen-35M head metadata and a finite positive stored temperature"
    if task == "panel" and cfg.get("genera") != baseline.PANEL_GENERA:
        return "Panel label order differs from the comparison"
    return None


def inspect_head(root: Path, stem: str, task: str, candidates: list[Path], expected: dict | None) -> dict:
    import torch

    path, config_path = root / f"{stem}.pt", root / f"{stem}_config.json"
    result = {"directory": str(root.resolve()), "stem": stem, "task": task, "issues": [],
              "matches": [], "status": "provenance_incomplete", "validation_reproduced": False,
              "files": {p.name: sha256(p) for p in (path, config_path) if p.is_file()}}
    if not config_path.is_file():
        result["issues"].append(f"Missing {config_path.name}; shared config.json is NOT a fallback")
    else:
        try:
            cfg = json.loads(config_path.read_text())
            result["config"] = cfg
            issue = metadata_issue(cfg, task)
            if issue:
                result["issues"].append(issue)
        except (ValueError, TypeError, AttributeError) as error:
            result["issues"].append(f"Invalid per-artifact configuration: {error}")
    if expected is not None and result["files"] != expected["files"]:
        result["issues"].append("Deployed artifact/config hashes differ from the recorded top-100 comparison")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        baseline.validate_head(state, len(baseline.PANEL_GENERA) if task == "panel" else 1)
        identity = tensor_fingerprint(state)
        result["tensor_sha256"] = identity
    except Exception as error:
        result["issues"].append(f"Cannot validate complete head: {type(error).__name__}: {error}")
        result["status"] = "invalid_artifact"
        return result
    result["candidate_errors"] = []
    for candidate in candidates:
        if candidate.name != path.name or candidate.resolve() == path.resolve():
            continue
        try:
            other = torch.load(candidate, map_location="cpu", weights_only=True)
            baseline.validate_head(other, len(baseline.PANEL_GENERA) if task == "panel" else 1)
            if tensor_fingerprint(other) == identity:
                result["matches"].append({"path": str(candidate.resolve()), "sha256": sha256(candidate),
                                          "byte_identical": sha256(candidate) == sha256(path),
                                          "tensor_identical": True})
        except Exception as error:
            result["candidate_errors"].append({"path": str(candidate), "error": f"{type(error).__name__}: {error}"})
    if result["issues"]:
        result["status"] = "invalid_artifact"
    # Even a unique match cannot prove that today's labels or split belong to it.
    result["identity_status"] = ("no_matching_member" if not result["matches"] else
                                 "unique_tensor_match" if len(result["matches"]) == 1 else "multiple_tensor_matches")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-dir", type=Path, default=baseline.REWARD_DIR)
    parser.add_argument("--hemo-dir", type=Path, default=baseline.REWARD_HEMO_DIR)
    parser.add_argument("--source", type=Path, default=Path("sweep_results/epoch58-top100-v1"),
                        help="Completed comparison whose deployed artifact hashes must match")
    parser.add_argument("--training-run", action="append", type=Path, default=[],
                        help="Explicit original run directory; inspect its heads and immediate member*/ heads (repeatable)")
    parser.add_argument("--record", action="append", type=Path, default=[],
                        help="Additional original label/split/log file to inventory by hash, not treat as validated (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = sorted({p.resolve() for p in [args.reward_dir, args.hemo_dir, *args.training_run]})
    for path in roots:
        if not path.is_dir():
            raise ValueError(f"Missing explicit artifact/training directory: {path}")
    separate_output(args.out, [*roots, args.source, *args.record])
    manifest_path = args.source / "run.json"
    source = json.loads(manifest_path.read_text())
    if source.get("kind") != "paired_top100_v1" or set(source.get("classifiers", {})) != {"activity", "panel", "hemolysis"}:
        raise ValueError("Expected the original paired top-100 classifier manifest")
    candidates = sorted({p.resolve() for root in roots for pattern in ("classifier*.pt", "member*/classifier*.pt")
                         for p in root.glob(pattern) if p.is_file()})
    records = set(p.resolve() for p in args.record)
    for root in roots:
        for pattern in ("*.json", "member*/*.json"):
            records.update(p.resolve() for p in root.glob(pattern) if p.is_file())
    inputs = {str(p): sha256(p) for p in sorted(records | set(candidates) | {manifest_path.resolve()})}
    recipe = {"kind": "reward_artifact_inventory_v1", "code": code_identity(), "inputs": inputs,
              "search_roots": list(map(str, roots)), "source_classifiers": source["classifiers"]}
    prepare_run(args.out, recipe)
    if (args.out / "complete.json").exists():
        baseline.checked_stage(args.out, "complete.json", {"inventory.json"})
        report = json.loads((args.out / "inventory.json").read_text())
    else:
        report = {"artifacts": {}, "records": {str(p): sha256(p) for p in sorted(records)},
                  "training_provenance_verified": False,
                  "next_required_evidence": [
                      "Original training data snapshot/hash and label definition for each matching artifact",
                      "Original train/validation membership and training command/seed, bound to that data and head",
                      "Original resolved backbone revision, runtime, and precise validation metrics",
                      "Reproduction with stored temperature, then calibration and cross-split similarity audit",
                  ],
                  "caveat": "Head identity/compatibility is not validation or biological safety; do not guess a winner from AUROC"}
        for name, root, stem, task in (("activity", args.reward_dir, "classifier", "binary"),
                                       ("panel", args.reward_dir, "classifier_panel", "panel"),
                                       ("hemolysis", args.hemo_dir, "classifier", "binary")):
            report["artifacts"][name] = inspect_head(root, stem, task, candidates, source["classifiers"][name])
        if {path: sha256(Path(path)) for path in inputs} != inputs:
            raise ValueError("Artifact or provenance records changed during inventory")
        write_json(args.out / "inventory.json", report)
        mark_files(args.out, "complete.json", ["inventory.json"])
    for name, item in report["artifacts"].items():
        print(f"[reward-audit] {name}: {item['status']}; {len(item['matches'])} tensor matches", flush=True)
        for issue in item["issues"]:
            print(f"  {issue}", flush=True)
    print(f"[reward-audit] inspect {args.out}/inventory.json; validation has NOT been reproduced", flush=True)
    if any(item["status"] == "invalid_artifact" for item in report["artifacts"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
