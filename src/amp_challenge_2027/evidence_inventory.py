"""Explicit, hash-linked inventory for competition readiness evidence.

This module reads only configured files. It never executes training, loads
models, downloads data, or assumes that an intact hash proves a scientific claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
        if not isinstance(source.get("root"), str) or not isinstance(source.get("files"), list):
            raise ValueError(f"Source {source_id}: root and files are required")
        if source.get("inspection_status") not in {"not_inspected", "already_inspected", "unknown"}:
            raise ValueError(f"Source {source_id}: invalid inspection_status")
        marker = source.get("marker")
        if marker is not None:
            if not isinstance(marker, dict) or not isinstance(marker.get("name"), str):
                raise ValueError(f"Source {source_id}: marker must name a file")
            if marker.get("type") == "generation_manifest":
                if not isinstance(marker.get("outputs"), list):
                    raise ValueError(f"Source {source_id}: generation manifest outputs must be a list")
            elif not isinstance(marker.get("files"), list):
                raise ValueError(f"Source {source_id}: marker.files must be a list")
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
            if marker.get("kind") != "amp_generation_manifest_v1" or marker.get("status") != "complete":
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
    out_dir.mkdir(parents=True)

    inventory_sources = []
    metrics = []
    required_failure = False
    invalid_failure = False
    for source in sources:
        root = _safe_path(repo_root, source["root"])
        entry = {
            "id": source["id"], "kind": source["kind"], "root": source["root"],
            "required": bool(source.get("required", False)),
            "inspection_status": source["inspection_status"], "files": [],
        }
        root_exists = root.is_dir()
        entry["root_status"] = "verified" if root_exists else "missing"
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
        if marker["status"] == "invalid":
            invalid_failure = True
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
                marker_verified_file = relative in marker_verified or relative == marker_name
                file_item.update(status="verified" if ((expected and digest == expected) or marker_verified_file) else "unverified",
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

    status = {
        "engineering": "Historical validation is recorded; run strict generation before release.",
        "predictor_evidence": "See inventoried reports; hashes alone do not establish biological validity.",
        "selection_evidence": "Top-100 audit status follows its configured artifact source.",
        "competition_standing": "Official score/rank is unavailable unless a receipt is configured.",
        "missing_evidence": [],
        "next_action": "Review missing/invalid sources, then inspect COMPETITION_STATUS.md and choose one bounded next experiment.",
    }
    for source in inventory_sources:
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
    lines.extend(["", "## Missing or invalid evidence", ""])
    missing = []
    for source in sources:
        missing.extend((source["id"], item) for item in source["files"] if item["status"] in {"missing", "invalid"})
    if missing:
        lines.extend(f"- `{source_id}/{item['path']}`: {item['status']} {item.get('reason', '')}" for source_id, item in missing)
    else:
        lines.append("- No required artifact is currently missing or invalid.")
    lines.extend(["", "## Next action", "", status["next_action"], ""])
    return "\n".join(lines)
