"""File provenance and strict evaluation shared by the remote experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity() -> dict:
    """Hash source as well as HEAD, so uncommitted edits cannot reuse caches."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    paths = [REPO_ROOT / "uv.lock", REPO_ROOT / "pyproject.toml"]
    paths += sorted((REPO_ROOT / "src").rglob("*.py"))
    paths += sorted((REPO_ROOT / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(sha256(path).encode())
    versions = {}
    for name in ("torch", "numpy", "transformers", "seqme", "scipy", "Levenshtein"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "commit": commit, "source_sha256": digest.hexdigest(),
        "python": platform.python_version(), "packages": versions,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def metrics_for_json(df, *, expected_names: list[str] | None = None) -> dict:
    """Require real metric values while retaining absent deviations as null."""
    from sweep_selection import _collect_metrics

    row: dict = {}
    names = _collect_metrics(row, df, dataset_name="library")
    if expected_names is not None and set(names) != set(expected_names):
        raise ValueError(f"Incomplete metric results: expected {expected_names}, got {names}")
    if not names or any(not math.isfinite(row[name]) for name in names):
        raise ValueError("Evaluation returned missing or non-finite primary metric values")
    for key, value in row.items():
        # seqme represents an unreported deviation as NaN in its DataFrame.
        # This is not a missing primary measurement and is valid as JSON null.
        if math.isnan(value):
            row[key] = None
        elif not math.isfinite(value):
            raise ValueError(f"Evaluation returned a non-finite statistic: {key}")
    return row


def prepare_run(directory: Path, recipe: dict) -> None:
    """Resume only an identical recipe; never mix results from different inputs."""
    manifest = directory / "run.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != recipe:
            differences = _recipe_differences(previous, recipe)
            detail = ", ".join(differences[:12]) or "unknown recipe difference"
            if len(differences) > 12:
                detail += f", ... ({len(differences) - 12} more)"
            raise ValueError(
                f"Experiment inputs changed: {detail}. Use a new --out directory "
                f"(existing run: {directory}; suggested: {directory}-v2)"
            )
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing nonempty experiment directory without run.json: {directory}")
    else:
        write_json(manifest, recipe)


def _recipe_differences(old, new, prefix="") -> list[str]:
    """Return bounded key-path diagnostics without exposing arbitrary values."""
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(set(old) | set(new)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                result.append(_difference_category(path))
            else:
                result.extend(_recipe_differences(old[key], new[key], path))
        return result
    if isinstance(old, list) and isinstance(new, list):
        if old == new:
            return []
        return [_difference_category(prefix)]
    if old != new:
        return [_difference_category(prefix)]
    return []


def _difference_category(path: str) -> str:
    lower = path.lower()
    if any(word in lower for word in ("commit", "source_sha", "code", "python", "package", "device", "runtime")):
        category = "code/runtime"
    elif any(word in lower for word in ("checkpoint", "model", "classifier", "backbone", "temperature", "head")):
        category = "model"
    elif any(word in lower for word in ("protocol", "weight", "seed", "threshold", "top_k", "shortlist", "policy")):
        category = "protocol"
    else:
        category = "data"
    return f"{category}:{path or '<root>'}"


def verify_files(directory: Path, marker: str) -> bool:
    path = directory / marker
    if not path.exists():
        return False
    for name, expected in json.loads(path.read_text())["files"].items():
        artifact = directory / name
        if not artifact.exists() or sha256(artifact) != expected:
            raise ValueError(f"Cached experiment artifact changed or missing: {artifact}")
    return True


def mark_files(directory: Path, marker: str, names: list[str]) -> None:
    write_json(directory / marker, {"files": {name: sha256(directory / name) for name in names}})


def evaluate_library(directory: Path, *, reference: Path, esm_model: str, device: str, seed: int | None = None) -> dict:
    """Use a subprocess so the 650M embedder is released between experiment cells."""
    seeded_recipe = directory / "evaluation_recipe.json"
    if seed is not None or seeded_recipe.exists():
        recipe = {"library_sha256": sha256(directory / "library.fasta"), "reference_sha256": sha256(reference),
                  "esm_model": esm_model, "device": device, "seed": seed}
        if seeded_recipe.exists():
            if json.loads(seeded_recipe.read_text()) != recipe:
                raise ValueError("Evaluation recipe changed; use a new output directory")
        elif (directory / "evaluation.json").exists():
            raise ValueError("Existing evaluation has no RNG-seed provenance")
        else:
            write_json(seeded_recipe, recipe)
    if not verify_files(directory, "evaluation.json"):
        command = [
            sys.executable, str(REPO_ROOT / "scripts/eval_official.py"),
            "--library", str((directory / "library.fasta").resolve()),
            "--reference", str(reference.resolve()), "--esm-model", esm_model,
            "--device", device, "--strict", "--out", str((directory / "metrics.csv").resolve()),
            "--json-out", str((directory / "metrics.json").resolve()),
        ]
        if seed is not None:
            command += ["--seed", str(seed)]
        print(f"[experiment] evaluating {directory} with {esm_model}", flush=True)
        log_path = directory / "evaluation.log"
        try:
            with log_path.open("w") as log:
                subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            from collections import deque

            with log_path.open() as log:
                tail = "".join(deque(log, maxlen=40))
            raise RuntimeError(f"Evaluation failed; see {log_path}\n{tail}") from error
        mark_files(directory, "evaluation.json", ["library.fasta", "metrics.csv", "metrics.json"] +
                   (["evaluation_recipe.json"] if seeded_recipe.exists() else []))
    return json.loads((directory / "metrics.json").read_text())


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
