"""Read-only import of complete paired_top100_v1 caches, with strict provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import compare_top100 as baseline
import numpy as np
from experiment_utils import sha256

from amp_challenge_2027.data import iter_fasta, read_reference_set
from amp_challenge_2027.select import clean_candidates


def separate_output(out: Path, sources: list[Path]) -> None:
    out = out.resolve()
    for source in sources:
        source = source.resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Output must be separate from, not inside/above, every input")


def load_source(root: Path, reference_path: Path) -> tuple[dict, list[dict], list[str]]:
    root = root.resolve()
    manifest = json.loads((root / "run.json").read_text())
    expected = {"kind": "paired_top100_v1", "weights": baseline.WEIGHTS,
                "ranking_seed": baseline.RANKING_SEED, "novelty_candidates": 2000,
                "top_k": baseline.TOP_K, "novelty_threshold": baseline.TOP_SIMILARITY_THRESHOLD,
                "hemo_weight": 0.0, "reference_sha256": sha256(reference_path)}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Source reference or ranking protocol differs from the frozen comparison")
    classifiers = manifest.get("classifiers", {})
    if set(classifiers) != {"activity", "panel", "hemolysis"} or not manifest.get("code", {}).get("source_sha256"):
        raise ValueError("Missing source classifier/code provenance")
    for name, artifact in classifiers.items():
        stem = "classifier_panel" if name == "panel" else "classifier"
        cfg = artifact.get("config", {})
        temperature = cfg.get("temperature", 0)
        if (artifact.get("stem") != stem or set(artifact.get("files", {})) != {f"{stem}.pt", f"{stem}_config.json"}
                or cfg.get("esm_model") != baseline.BACKBONE or cfg.get("unfreeze_layers") != 0
                or cfg.get("checkpoint_format") != "head-only" or cfg.get("task") != ("panel" if name == "panel" else "binary")
                or not isinstance(temperature, (float, int)) or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("Invalid frozen source classifier provenance")
        if name == "panel" and cfg.get("genera") != baseline.PANEL_GENERA:
            raise ValueError("Source panel label order changed")
    records = manifest["cells"]
    if (len(records) != 6 or {(c["case"], c["seed"]) for c in records}
            != {(case, seed) for case in ("hybrid", "p3_s1") for seed in (42, 43, 44)}):
        raise ValueError("Require exactly six unique paired cells (generation seeds 42, 43, 44)")
    reference = sorted(read_reference_set(reference_path))
    if not reference:
        raise ValueError("Reference is empty")
    ref_set = set(reference)
    cells = []
    for record in sorted(records, key=lambda item: (item["seed"], item["case"])):
        directory = root / record["case"] / f"seed{record['seed']}"
        proof = baseline.checked_stage(directory, "complete.json",
                                       {"library.fasta", "scores.npz", "top.fasta", "top_scores.csv", "summary.json"})
        if sha256(directory / "library.fasta") != record["library_sha256"]:
            raise ValueError("Copied library differs from its original ordered source")
        sequences = [seq for _, seq in iter_fasta(directory / "library.fasta")]
        if len(sequences) != baseline.LIBRARY_SIZE or clean_candidates(sequences, ref_set) != sequences:
            raise ValueError("Source library must be complete, clean, unique and ordered")
        with np.load(directory / "scores.npz", allow_pickle=False) as data:
            scores = {key: data[key] for key in data.files}
        baseline.validate_scores(scores, len(sequences))
        top = [seq for _, seq in iter_fasta(directory / "top.fasta")]
        summary = json.loads((directory / "summary.json").read_text())
        with (directory / "top_scores.csv").open(newline="") as handle:
            top_rows = list(csv.DictReader(handle))
        if (len(top) != baseline.TOP_K or len(set(top)) != baseline.TOP_K or not set(top) <= set(sequences)
                or summary.get("top_sequences") != top or [row["sequence"] for row in top_rows] != top
                or summary.get("case") != record["case"] or summary.get("seed") != record["seed"]
                or summary.get("library_sha256") != record["library_sha256"]
                or summary.get("top_size") != baseline.TOP_K):
            raise ValueError("Cached top/summary identity or order mismatch")
        combined, parts = baseline.combine_scores(scores)
        index = {seq: i for i, seq in enumerate(sequences)}
        ids = [index[seq] for seq in top]
        for rank, (row, i) in enumerate(zip(top_rows, ids), start=1):
            if int(row["rank"]) != rank:
                raise ValueError("Cached top ranks are misaligned")
            numbers = {**{name: float(values[i]) for name, values in parts.items()},
                       "composite_score": float(combined[i]),
                       "panel_mean_probability": float(scores["panel"][i].mean()),
                       **{f"p_active:{genus}": float(scores["panel"][i, j]) for j, genus in enumerate(baseline.PANEL_GENERA)}}
            if any(not np.isclose(float(row[key]), value, rtol=1e-6, atol=1e-7) for key, value in numbers.items()):
                raise ValueError("Cached score rows do not match ordered library arrays")
        risk = np.array([float(row["hemo_risk"]) for row in top_rows])
        if not np.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
            raise ValueError("Invalid cached hemolysis probabilities")
        measured = {**{f"{key}_mean": float(values[ids].mean()) for key, values in parts.items()},
                    "panel_mean_probability": float(scores["panel"][ids].mean()),
                    "hemo_risk_mean": float(risk.mean()), "hemo_risk_p75": float(np.percentile(risk, 75)),
                    "hemo_risk_max": float(risk.max())}
        if any(not np.isclose(summary.get(key, np.nan), value, rtol=1e-6, atol=1e-7) for key, value in measured.items()):
            raise ValueError("Cached summary does not reproduce its score rows")
        cells.append({"case": record["case"], "seed": record["seed"], "directory": directory,
                      "record": record, "proof": proof, "sequences": sequences, "scores": scores,
                      "parts": parts, "combined": combined, "top": top, "top_rows": top_rows, "summary": summary})
    backbone = baseline.checked_stage(root, "reference_cache.json", {"reference_embeddings.npy", "backbones.json"})
    revisions = json.loads((root / "backbones.json").read_text())
    if set(revisions) != {"activity", "panel", "hemolysis", "precision"} or not all(revisions.values()):
        raise ValueError("Missing resolved scoring backbone revisions")
    identity = {"source": str(root), "run_sha256": sha256(root / "run.json"), "source_run": manifest,
                "reference_cache": backbone, "backbones": revisions,
                "cells": [{"case": c["case"], "seed": c["seed"], "proof": c["proof"]} for c in cells]}
    return identity, cells, reference


def verify_source(identity: dict) -> None:
    """Catch source changes during analysis and before trusting a resumed output."""
    root = Path(identity["source"])
    if sha256(root / "run.json") != identity["run_sha256"]:
        raise ValueError("Source manifest changed during analysis")
    for cell in identity["cells"]:
        directory = root / cell["case"] / f"seed{cell['seed']}"
        proof = baseline.checked_stage(directory, "complete.json", set(cell["proof"]["files"]))
        if proof != cell["proof"]:
            raise ValueError("Source cache changed during analysis")
    proof = baseline.checked_stage(root, "reference_cache.json", {"reference_embeddings.npy", "backbones.json"})
    if proof != identity["reference_cache"]:
        raise ValueError("Scoring backbone provenance changed during analysis")
