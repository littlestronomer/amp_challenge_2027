"""Provenance helpers for strict, reproducible competition generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from amp_challenge_2027.inference_metadata import load_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Model directory missing: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Model directory has no artifacts: {directory}")
    return {str(path.relative_to(directory)): sha256(path) for path in files}


def _backbone_revision(scorer) -> str | None:
    model = getattr(scorer, "_model", None)
    model = getattr(model, "esm", model)
    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else None


def write_generation_manifest(args, scorer, reference_path: Path, output_paths: dict[str, Path]) -> Path:
    """Write a manifest after strict generation has passed all output checks."""
    project = Path(__file__).resolve().parents[2]
    primary = Path(args.checkpoint).resolve()
    secondary = Path(args.blend_checkpoint).resolve()
    artifacts = {
        "primary_generator": {"path": str(primary.relative_to(project)), "files": artifact_hashes(primary)},
        "secondary_generator": {"path": str(secondary.relative_to(project)), "files": artifact_hashes(secondary)},
    }
    head_specs = {
        "activity": ("checkpoint/reward", "classifier"),
        "breadth": ("checkpoint/reward", "classifier_panel"),
        "mdr": ("checkpoint/reward", "classifier_panel"),
        "safety": ("checkpoint/reward_hemo", "classifier"),
    }
    scorer_artifacts = {}
    for name in scorer.names:
        if name in head_specs:
            directory, stem = head_specs[name]
            root = project / directory
            checkpoint = root / f"{stem}.pt"
            config = load_config(root, stem)
            scorer_artifacts[name] = {
                "checkpoint_sha256": sha256(checkpoint),
                "effective_metadata": config,
                "backbone_revision": _backbone_revision(scorer_by_name(scorer, name)),
            }
        elif name == "precision":
            precision_scorer = scorer_by_name(scorer, name)
            model = getattr(precision_scorer, "_model", None)
            config = getattr(model, "config", None)
            scorer_artifacts[name] = {
                "model": getattr(config, "_name_or_path", None),
                "backbone_revision": _backbone_revision(precision_scorer),
                "cache_key": getattr(precision_scorer, "_cache_path", None).name
                if getattr(precision_scorer, "_cache_path", None) else None,
            }
        else:
            scorer_artifacts[name] = {"backbone_revision": None}
    packages = {}
    for name in ("torch", "transformers", "huggingface-hub", "numpy", "levenshtein"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=project)
        dirty_hash = hashlib.sha256(diff).hexdigest()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=project, text=True)
    except (OSError, subprocess.CalledProcessError):
        head, dirty_hash, status = None, None, ""
    source_files = ["pyproject.toml", "uv.lock", "src/amp_challenge_2027/inference_metadata.json"]
    source_files += [str(path.relative_to(project)) for path in
                     sorted((project / "src/amp_challenge_2027").rglob("*.py"))]
    revisions = {name: item.get("backbone_revision") for name, item in scorer_artifacts.items()
                 if "backbone_revision" in item}
    manifest = {
        "kind": "amp_generation_manifest_v1",
        "status": "complete",
        "recipe": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if key != "strict"},
        "source": {"git_head": head, "tracked_diff_sha256": dirty_hash,
                   "working_tree_status": status.splitlines(),
                   "source_file_sha256": {name: sha256(project / name) for name in source_files}},
        "runtime": {"python": platform.python_version(), "packages": packages, "platform": sys.platform},
        "device": args.device,
        "reference": {"path": str(reference_path.relative_to(project)), "sha256": sha256(reference_path)},
        "generators": artifacts,
        "scorers": scorer_artifacts,
        "actual_components": sorted(scorer.names),
        "all_backbone_revisions_pinned": bool(revisions) and all(revisions.values()),
        "outputs": {name: {"path": path.name, "sha256": sha256(path)}
                    for name, path in output_paths.items()},
        "limitations": [
            "Resolved backbone revision is null when the runtime did not expose an immutable revision.",
            "Predicted activity and hemolysis scores are not experimental measurements.",
        ],
    }
    path = next(iter(output_paths.values())).parent / "generation_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def scorer_by_name(scorer, name: str):
    for component_name, _weight, function in scorer.components:
        if component_name == name:
            owner = getattr(function, "__self__", None)
            if owner is not None:
                return owner
            # Panel component functions are bound to PanelScorer; conformity
            # and precision are likewise bound methods.
    return None
