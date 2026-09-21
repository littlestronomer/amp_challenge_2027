"""Explicit, hash-linked inventory for competition readiness evidence.

This module reads only configured files. It never executes training, loads
models, downloads data, or assumes that an intact hash proves a scientific claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

METRIC_FIELDS = (
    "source_id", "relative_file", "locator", "task", "model_or_recipe",
    "population_or_split", "metric", "value", "evidence_status",
)
ALLOWED_STATUSES = {"verified", "unverified", "missing", "invalid", "unsupported"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe_path(root: Path, relative: str) -> Path:
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError(f"Source file must be a relative, non-traversing path: {relative}")
    resolved_root = root.resolve()
    resolved = (root / item).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Source path escapes its configured root: {relative}")
    return resolved


def _validate_config(config: dict) -> list[dict]:
    if not isinstance(config, dict):
        raise ValueError("Evidence config must be a JSON object")
    if config.get("kind") != "competition_evidence_inventory_v1":
        raise ValueError("Unsupported evidence config kind")
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ValueError("config.sources must be a list")
    ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in ids:
            raise ValueError(f"Missing or duplicate source id: {source_id!r}")
        ids.add(source_id)
        if not isinstance(source.get("kind"), str) or not source["kind"]:
            raise ValueError(f"Source {source_id}: kind must be a nonempty string")
        if not isinstance(source.get("root"), str) or not isinstance(source.get("files"), list):
            raise ValueError(f"Source {source_id}: root and files are required")
        if source.get("adapter", "generic") not in {"generic", "family_benchmark_test"}:
            raise ValueError(f"Source {source_id}: unsupported evidence adapter {source.get('adapter')!r}")
        if not isinstance(source.get("required", False), bool):
            raise ValueError(f"Source {source_id}: required must be a boolean")
        if any(not isinstance(name, str) or not name for name in source["files"]):
            raise ValueError(f"Source {source_id}: files must be nonempty relative paths")
        if source.get("inspection_status") not in {"not_inspected", "already_inspected", "unknown"}:
            raise ValueError(f"Source {source_id}: invalid inspection_status")
        marker = source.get("marker")
        if marker is not None:
            if not isinstance(marker, dict) or not isinstance(marker.get("name"), str) or not marker["name"]:
                raise ValueError(f"Source {source_id}: marker must name a file")
            if marker.get("type") == "generation_manifest":
                if not isinstance(marker.get("outputs"), list) or not all(isinstance(x, str) and x for x in marker["outputs"]):
                    raise ValueError(f"Source {source_id}: generation manifest outputs must be a list")
            elif not isinstance(marker.get("files"), list) or not all(isinstance(x, str) and x for x in marker["files"]):
                raise ValueError(f"Source {source_id}: marker.files must be a list")
        hashes = source.get("expected_sha256", {})
        if not isinstance(hashes, dict) or any(
            not isinstance(name, str) or not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
            for name, value in hashes.items()
        ):
            raise ValueError(f"Source {source_id}: expected_sha256 must map paths to 64-digit SHA-256 values")
        if set(hashes) - set(source["files"]):
            raise ValueError(f"Source {source_id}: expected_sha256 names a file not listed in files")
        for relative in source["files"]:
            item = Path(relative)
            if item.is_absolute() or ".." in item.parts:
                raise ValueError(f"Source {source_id}: file path must be relative and non-traversing")
        if marker is not None and marker.get("type") == "strict_repeatability":
            if not isinstance(marker.get("runs"), list) or marker["runs"] != ["run1", "run2"]:
                raise ValueError(f"Source {source_id}: strict repeatability requires run1 and run2")
    return sorted(sources, key=lambda item: item["id"])


def _read_marker(root: Path, source: dict) -> dict:
    spec = source.get("marker")
    if spec is None:
        return {"status": "unverified", "reason": "No producer completion marker configured"}
    marker_path = _safe_path(root, spec["name"])
    if not marker_path.is_file():
        return {"status": "missing", "reason": f"Missing marker {spec['name']}"}
    try:
        marker = json.loads(marker_path.read_text())
        if spec.get("type") == "generation_manifest":
            if marker.get("kind") not in {"amp_generation_manifest_v1", "amp_generation_manifest_v2"} or marker.get("status") != "complete":
                raise ValueError("invalid generation manifest identity/status")
            checked = []
            for name in spec["outputs"]:
                item = marker.get("outputs", {}).get(name)
                if not isinstance(item, dict):
                    raise ValueError(f"generation manifest lacks output {name}")
                path = _safe_path(root, item.get("path", ""))
                if not path.is_file() or sha256(path) != item.get("sha256"):
                    return {"status": "invalid", "reason": f"Generation output hash mismatch: {name}"}
                checked.append(path.name)
            checked.append(spec["name"])
            return {"status": "verified", "sha256": sha256(marker_path), "checked_outputs": checked}
        files = marker.get("files")
        if not isinstance(files, dict):
            raise ValueError("marker has no files object")
        for name in spec["files"]:
            expected = files.get(name)
            path = _safe_path(root, name)
            if not path.is_file():
                return {"status": "invalid", "reason": f"Marker dependency missing: {name}"}
            if not isinstance(expected, str) or sha256(path) != expected:
                return {"status": "invalid", "reason": f"Marker hash mismatch: {name}"}
        if spec.get("type") in {"selectivity_risk_cache", "selectivity_comparison"}:
            expected_kind = ("selectivity_risk_cache_complete_v1" if spec["type"] == "selectivity_risk_cache"
                             else "selectivity_comparison_complete_v1")
            if marker.get("kind") != expected_kind:
                raise ValueError("wrong selectivity completion marker kind")
            cells = marker.get("cells")
            wanted = {"42", "43", "44"} if spec["type"] == "selectivity_risk_cache" else {
                f"{policy}/hybrid/seed{seed}" for policy in ("C0", "R1") for seed in (42, 43, 44)}
            if not isinstance(cells, dict) or set(cells) != wanted:
                raise ValueError("selectivity marker lacks complete expected cell coverage")
            if spec["type"] == "selectivity_risk_cache":
                coverage = json.loads(_safe_path(root, "coverage.json").read_text())
                status = json.loads(_safe_path(root, "status.json").read_text())
                if (coverage.get("complete") is not True
                        or any(coverage.get("cells", {}).get(seed) is not True for seed in wanted)
                        or status.get("status") != "complete"):
                    raise ValueError("risk cache is marked incomplete in its coverage/status reports")
            for key, proof in cells.items():
                parts = key.split("/")
                cell_root = root / (Path("cells/hybrid") / f"seed{key}" if spec["type"] == "selectivity_risk_cache"
                                   else Path(parts[0]) / parts[1] / parts[2])
                cell_marker_path = cell_root / "complete.json"
                expected_cell_hash = proof.get("complete_sha256") if spec["type"] == "selectivity_risk_cache" else proof
                if not cell_marker_path.is_file() or sha256(cell_marker_path) != expected_cell_hash:
                    raise ValueError(f"selectivity cell marker mismatch: {key}")
                cell_marker = json.loads(cell_marker_path.read_text())
                dependencies = cell_marker.get("files", {})
                if not isinstance(dependencies, dict) or not dependencies:
                    raise ValueError(f"selectivity cell marker has no dependencies: {key}")
                for relative, expected_hash in dependencies.items():
                    path = _safe_path(cell_root, relative)
                    if not path.is_file() or sha256(path) != expected_hash:
                        raise ValueError(f"selectivity cell artifact mismatch: {key}/{relative}")
                if spec["type"] == "selectivity_risk_cache":
                    if sha256(cell_root / "risk.npy") != proof.get("risk_sha256"):
                        raise ValueError(f"selectivity risk array mismatch: {key}")

        if spec.get("type") == "strict_repeatability":
            comparison_path = _safe_path(root, "comparison.json")
            comparison = json.loads(comparison_path.read_text())
            if comparison.get("kind") != "strict_generation_repeatability_v1":
                raise ValueError("unsupported strict-repeatability comparison")
            manifests = []
            output_hashes = []
            for run_name in spec["runs"]:
                run_root = _safe_path(root, run_name)
                manifest_path = _safe_path(run_root, "generation_manifest.json")
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("kind") not in {"amp_generation_manifest_v1", "amp_generation_manifest_v2"} or manifest.get("status") != "complete":
                    raise ValueError(f"invalid {run_name} generation manifest")
                artifacts = {}
                for logical, item in manifest.get("outputs", {}).items():
                    path = _safe_path(run_root, item.get("path", ""))
                    if not path.is_file() or sha256(path) != item.get("sha256"):
                        raise ValueError(f"{run_name} {logical} output does not match its manifest")
                    artifacts[logical] = item["sha256"]
                if set(artifacts) != {"library", "top", "top_scores"}:
                    raise ValueError(f"{run_name} manifest does not cover expected outputs")
                manifests.append(manifest)
                output_hashes.append(artifacts)
            def comparable_recipe(value):
                recipe = dict(value.get("recipe", {}))
                recipe.pop("out_dir", None)
                return recipe
            if comparable_recipe(manifests[0]) != comparable_recipe(manifests[1]):
                raise ValueError("strict-run effective recipes differ")
            if output_hashes[0] != output_hashes[1]:
                raise ValueError("strict-run library/top/score bytes differ")
            if comparison.get("outputs_equal") is not True:
                raise ValueError("comparison report does not attest equal output hashes")
            if (comparison.get("run1_output_sha256") != output_hashes[0]
                    or comparison.get("run2_output_sha256") != output_hashes[1]
                    or comparison.get("run1_manifest_sha256") != sha256(_safe_path(root / "run1", "generation_manifest.json"))
                    or comparison.get("run2_manifest_sha256") != sha256(_safe_path(root / "run2", "generation_manifest.json"))):
                raise ValueError("strict-repeatability report hashes do not match verified source artifacts")
            return {"status": "verified", "sha256": sha256(marker_path),
                    "checked_files": spec["files"], "outputs_equal": True,
                    "runtime_comparison": "same recipe and byte-identical outputs; runtime metadata reviewed separately"}
        return {"status": "verified", "sha256": sha256(marker_path), "checked_files": spec["files"]}
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return {"status": "invalid", "reason": f"Cannot verify marker: {exc}"}


def _verified_family_metrics(source: dict, relative: str, path: Path,
                             producer_verified: bool) -> list[dict]:
    """Parse only the fixed schema emitted by benchmark_reward_generalization.py."""
    if source.get("kind") != "family_benchmark_test" or relative != "results.csv":
        return []
    rows: list[dict] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"task", "architecture", "sequences", "families", "macro_auroc",
                    "macro_average_precision", "macro_brier", "macro_log_loss",
                    "macro_ece_10_equal_width", "auroc_ci_lower", "auroc_ci_upper"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("family benchmark CSV schema does not match the supported producer")
        for index, row in enumerate(reader, start=2):
            locator = f"row:{index}"
            for metric in sorted(required - {"task", "architecture", "sequences", "families"}):
                raw = row.get(metric, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    if raw in {"", "null", "None"}:
                        value = None
                    else:
                        raise ValueError(f"non-numeric {metric} at {locator}")
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"non-finite {metric} at {locator}")
                rows.append({
                    "source_id": source["id"], "relative_file": relative, "locator": locator,
                    "task": row["task"], "model_or_recipe": row["architecture"],
                    "population_or_split": "family-held-out test; see producer manifest",
                    "metric": metric, "value": value,
                    "evidence_status": "verified" if producer_verified else "unverified",
                })
    return rows


def collect_evidence(repo_root: Path, config_path: Path, out_dir: Path) -> int:
    """Collect configured source hashes and return the required-source exit code."""
    repo_root = repo_root.resolve()
    config = json.loads(config_path.read_text())
    sources = _validate_config(config)
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing existing evidence output directory: {out_dir}")
    # Validate every configured path before creating any output.
    for source in sources:
        root = _safe_path(repo_root, source["root"])
        for relative in source["files"]:
            _safe_path(root, relative)
        marker = source.get("marker")
        if marker is not None:
            _safe_path(root, marker["name"])
            for relative in marker.get("outputs", marker.get("files", [])):
                _safe_path(root, relative)
    out_dir.mkdir(parents=True)

    inventory_sources = []
    metrics = []
    required_failure = False
    invalid_failure = False
    for source in sources:
        root = _safe_path(repo_root, source["root"])
        entry = {
            "id": source["id"], "kind": source["kind"], "root": source["root"],
            "adapter": source.get("adapter", "family_benchmark_test" if source["kind"] == "family_benchmark_test" else "generic"),
            "required": bool(source.get("required", False)),
            "inspection_status": source["inspection_status"], "files": [],
        }
        root_exists = root.is_dir()
        entry["root_status"] = "present" if root_exists else "missing"
        if not root_exists and entry["required"]:
            required_failure = True
        if not root_exists:
            entry["marker"] = {"status": "missing", "reason": "Source root is absent"}
            for relative in source["files"]:
                entry["files"].append({"path": relative, "status": "missing"})
            inventory_sources.append(entry)
            continue

        marker = _read_marker(root, source)
        entry["marker"] = marker
        marker_verified = set(marker.get("checked_files", marker.get("checked_outputs", []))) if marker["status"] == "verified" else set()
        if marker["status"] == "verified" and source.get("marker") is not None:
            marker_verified.add(source["marker"]["name"])
        if marker["status"] == "invalid":
            invalid_failure = True
        if marker["status"] == "missing" and entry["required"] and source.get("marker") is not None:
            required_failure = True
        for relative in source["files"]:
            path = _safe_path(root, relative)
            expected = source.get("expected_sha256", {}).get(relative)
            file_item = {"path": relative}
            if not path.is_file():
                file_item["status"] = "missing"
                if entry["required"]:
                    required_failure = True
            else:
                digest = sha256(path)
                marker_name = (source.get("marker") or {}).get("name")
                marker_verified_file = relative in marker_verified
                if relative == marker_name and marker["status"] == "invalid":
                    file_item.update(status="invalid", reason="Configured producer completion marker is invalid",
                                     size_bytes=path.stat().st_size, sha256=digest)
                    invalid_failure = True
                    entry["files"].append(file_item)
                    continue
                if expected and digest != expected:
                    file_item.update(status="invalid", reason="Configured SHA-256 does not match",
                                     size_bytes=path.stat().st_size, sha256=digest, expected_sha256=expected)
                    invalid_failure = True
                else:
                    file_item.update(status="verified" if ((expected is not None) or marker_verified_file) else "unverified",
                                     size_bytes=path.stat().st_size, sha256=digest,
                                     expected_sha256=expected)
                try:
                    metrics.extend(_verified_family_metrics(
                        source, relative, path, marker.get("status") == "verified"
                    ))
                except (OSError, ValueError, csv.Error) as exc:
                    file_item["status"] = "invalid"
                    file_item["reason"] = str(exc)
                    invalid_failure = True
            entry["files"].append(file_item)
        inventory_sources.append(entry)

    strict = next((s for s in inventory_sources if s["kind"] == "strict_generation"), None)
    repeat = next((s for s in inventory_sources if s["kind"] == "strict_repeatability"), None)
    strict_verified = bool(strict and strict["marker"]["status"] == "verified")
    repeat_verified = bool(repeat and repeat["marker"]["status"] == "verified")
    status = {
        "engineering": ("Strict generation output hashes verified; two-run byte equality verified." if strict_verified and repeat_verified
                        else "Strict generation output hashes verified for one run; repeatability is not established." if strict_verified
                        else "No verified strict-generation output manifest is present."),
        "predictor_evidence": "Inventoried predictor reports are engineering or retrospective evidence; none establishes independent biological validity.",
        "selection_evidence": "Selection evidence is identified per configured source; score-based audits remain surrogate evidence.",
        "competition_standing": "Official score/rank is unavailable unless an official submission receipt verifies.",
        "missing_evidence": [],
        "next_action": "Review missing/invalid sources, then inspect COMPETITION_STATUS.md and choose one bounded next experiment.",
    }
    for source in inventory_sources:
        if source["marker"]["status"] in {"missing", "invalid"}:
            status["missing_evidence"].append({"source_id": source["id"], "path": source["marker"].get("reason", "marker"),
                                                "status": source["marker"]["status"]})
        for item in source["files"]:
            if item["status"] in {"missing", "invalid"}:
                status["missing_evidence"].append({"source_id": source["id"], **item})
    if required_failure or invalid_failure:
        status["next_action"] = "Resolve required missing or invalid sources; do not infer a scientific status from the partial inventory."

    write_json(out_dir / "inventory.json", {"kind": "competition_evidence_inventory_v1",
               "sources": inventory_sources, "required_sources_missing": required_failure,
               "invalid_inputs": invalid_failure})
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in sorted(metrics, key=lambda item: (item["source_id"], item["relative_file"], item["locator"], item["task"], item["model_or_recipe"], item["metric"])):
            writer.writerow(row)
    (out_dir / "STATUS.md").write_text(_status_markdown(status, inventory_sources))
    complete_files = {name: sha256(out_dir / name) for name in ("inventory.json", "metrics.csv", "STATUS.md")}
    write_json(out_dir / "complete.json", {"files": complete_files})
    return 2 if required_failure or invalid_failure else 0


def _status_markdown(status: dict, sources: list[dict]) -> str:
    lines = ["# Competition evidence status", "", "Generated from configured local artifacts. A file hash establishes integrity only, not scientific validity.", "",
             "## Snapshot", "", f"- Engineering: {status['engineering']}",
             f"- Predictor evidence: {status['predictor_evidence']}",
             f"- Selection evidence: {status['selection_evidence']}",
             f"- Competition standing: {status['competition_standing']}", "", "## Sources", "",
             "| ID | Kind | Root | Required | Root | Marker |", "|---|---|---|---:|---|---|"]
    for source in sources:
        lines.append(f"| {source['id']} | {source['kind']} | `{source['root']}` | {source['required']} | {source['root_status']} | {source['marker']['status']} |")
    lines.extend(["", "## Evidence status definitions", "",
                  "- `present`: configured root exists; this does not verify contents.",
                  "- `verified`: configured hashes/producer marker match the current bytes; this does not validate scientific claims.",
                  "- `unverified`: bytes exist but have no applicable verified hash marker.",
                  "- `missing`: expected root/file/marker is absent.",
                  "- `invalid`: bytes or marker contradict configured integrity checks.",
                  "", "## Missing or invalid evidence", ""])
    missing = []
    for source in sources:
        missing.extend((source["id"], item) for item in source["files"] if item["status"] in {"missing", "invalid"})
        if source["marker"]["status"] in {"missing", "invalid"}:
            missing.append((source["id"], {"path": source["marker"].get("reason", "marker"),
                                            "status": source["marker"]["status"]}))
    if missing:
        lines.extend(f"- `{source_id}/{item['path']}`: {item['status']} {item.get('reason', '')}" for source_id, item in missing)
    else:
        lines.append("- No required artifact is currently missing or invalid.")
    lines.extend(["", "## Next action", "", status["next_action"], ""])
    return "\n".join(lines)
