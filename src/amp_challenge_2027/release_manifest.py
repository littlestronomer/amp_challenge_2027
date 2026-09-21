"""Provenance helpers for strict, reproducible competition generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
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


def _config_provenance(root: Path, stem: str, checkpoint: Path, config: dict) -> dict:
    explicit = root / f"{stem}_config.json"
    legacy = root / "config.json"
    registry = Path(__file__).with_suffix(".json")
    if explicit.is_file():
        source, origin = explicit, "explicit_file"
    elif legacy.is_file():
        source, origin = legacy, "legacy_fallback_file"
    else:
        source, origin = registry, "head_hash_bound_registry"
    return {"origin": origin, "source_path": str(source.relative_to(root.parent.parent)
            if source.is_relative_to(root.parent.parent) else source),
            "source_sha256": sha256(source) if source.is_file() else None,
            "head_sha256": sha256(checkpoint), "effective_metadata": config}


def _revision_record(scorer, *, applicable: bool, explicitly_pinned: bool = False) -> dict:
    revision = _backbone_revision(scorer) if applicable else None
    return {"applicable": applicable, "resolved_immutable_revision": revision,
            "resolved": bool(revision), "explicitly_pinned_at_load": bool(explicitly_pinned)}


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
                "config_provenance": _config_provenance(root, stem, checkpoint, config),
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
                "cache": _precision_cache_identity(precision_scorer, reference_path),
            }
        else:
            scorer_artifacts[name] = {"backbone_revision": None,
                                      "backbone": _revision_record(None, applicable=False)}
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
    neural_names = {"activity", "breadth", "mdr", "safety", "precision"}
    for name, item in scorer_artifacts.items():
        applicable = name in neural_names
        owner = scorer_by_name(scorer, name)
        item["backbone"] = _revision_record(owner, applicable=applicable, explicitly_pinned=False)
    revisions = [item["backbone"] for item in scorer_artifacts.values() if item["backbone"]["applicable"]]
    manifest = {
        "kind": "amp_generation_manifest_v2",
        "status": "complete",
        "recipe": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if key != "strict"},
        "source": {"git_head": head, "tracked_diff_sha256": dirty_hash,
                   "working_tree_status": status.splitlines(),
                   "source_file_sha256": {name: sha256(project / name) for name in source_files}},
        "runtime": {"python": platform.python_version(), "packages": packages, "platform": sys.platform},
        "device": {"logical_device": args.device,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "reference": {"path": str(reference_path.relative_to(project)), "sha256": sha256(reference_path)},
        "generators": artifacts,
        "scorers": scorer_artifacts,
        "actual_components": sorted(scorer.names),
        "backbone_revision_summary": {
            "all_applicable_revisions_resolved": bool(revisions) and all(item["resolved"] for item in revisions),
            "all_applicable_revisions_explicitly_pinned": bool(revisions) and all(item["explicitly_pinned_at_load"] for item in revisions),
            "non_neural_scorers_excluded": [name for name, item in scorer_artifacts.items() if not item["backbone"]["applicable"]],
        },
        "outputs": {name: {"path": path.name, "sha256": sha256(path)}
                    for name, path in output_paths.items()},
        "limitations": [
            "Resolved revisions record the runtime-loaded checkpoint; this does not mean the loader explicitly pinned the revision.",
            "Predicted activity and hemolysis scores are not experimental measurements.",
        ],
    }
    path = next(iter(output_paths.values())).parent / "generation_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def _precision_cache_identity(scorer, reference_path: Path) -> dict:
    cache_path = getattr(scorer, "_cache_path", None)
    if cache_path is None:
        return {"status": "unavailable", "reason": "Scorer exposes no reference embedding cache path"}
    if not Path(cache_path).is_file():
        return {"status": "unavailable", "reason": "Reference embedding cache file is absent",
                "path": str(cache_path)}
    try:
        import numpy as np
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(array.shape), str(array.dtype)
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": f"Cannot inspect cache: {type(exc).__name__}"}
    return {"status": "verified_content_identity", "path": str(cache_path),
            "sha256": sha256(Path(cache_path)), "shape": shape, "dtype": dtype,
            "reference_fasta_sha256": sha256(reference_path),
            "backbone_revision": _backbone_revision(scorer)}


def scorer_by_name(scorer, name: str):
    for component_name, _weight, function in scorer.components:
        if component_name == name:
            owner = getattr(function, "__self__", None)
            if owner is not None:
                return owner
            # Panel component functions are bound to PanelScorer; conformity
            # and precision are likewise bound methods.
    return None
