"""Paired top-100 audit of immutable incumbent and epoch-58 75/25 libraries.

No generation, training, library-membership changes, or automatic promotion.
Ranking uses the current fixed five-component recipe; hemolysis is audit-only.
All reported classifier scores are surrogates, not independent biological tests.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from experiment_utils import (
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
    write_summary,
)

from amp_challenge_2027.config import (
    ANTIBACTERIAL_FASTA,
    LIBRARY_SIZE,
    MDR_PANEL_GENERA,
    PANEL_GENERA,
    REWARD_DIR,
    REWARD_HEMO_DIR,
    TOP_K,
    TOP_SIMILARITY_THRESHOLD,
)
from amp_challenge_2027.data import iter_fasta, read_reference_set, write_fasta
from amp_challenge_2027.props import compute_properties, is_plausible
from amp_challenge_2027.score import (
    ActivityScorer,
    CompositeScorer,
    ConformityScorer,
    HemoScorer,
    PanelScorer,
    PrecisionProxyScorer,
)
from amp_challenge_2027.select import clean_candidates, select_library_and_top

WEIGHTS = {"activity": 0.5, "conformity": 0.25, "precision": 0.25, "breadth": 1.0, "mdr": 1.0}
BACKBONE = "facebook/esm2_t12_35M_UR50D"
PRECISION_MODEL = "facebook/esm2_t6_8M_UR50D"
RANKING_SEED = 42
SCORE_KEYS = {"activity", "panel", "conformity", "precision"}


def checked_stage(directory: Path, marker: str, required: set[str]) -> dict:
    path = directory / marker
    files = json.loads(path.read_text())["files"]
    if not required.issubset(files) or any(Path(name).name != name for name in files):
        raise ValueError(f"Invalid completion marker: {path}")
    verify_files(directory, marker)
    return {"marker_sha256": sha256(path), "files": files}


def classifier_inputs() -> dict:
    """Require explicit per-artifact frozen-head metadata, never shared fallback."""
    result = {}
    for name, root, stem in (
        ("activity", REWARD_DIR, "classifier"),
        ("panel", REWARD_DIR, "classifier_panel"),
        ("hemolysis", REWARD_HEMO_DIR, "classifier"),
    ):
        path = root / f"{stem}_config.json"
        cfg = json.loads(path.read_text())
        temperature = float(cfg.get("temperature", 0))
        if (cfg.get("unfreeze_layers") != 0 or cfg.get("checkpoint_format") != "head-only"
                or cfg.get("esm_model") != BACKBONE or cfg.get("task") != ("panel" if name == "panel" else "binary")
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError(f"Requires frozen head with explicit valid calibration metadata: {path}")
        if name == "panel" and cfg.get("genera") != PANEL_GENERA:
            raise ValueError("Panel output labels/order differ from the ranking recipe")
        result[name] = {"config": cfg, "directory": str(root.resolve()), "stem": stem,
                        "files": {p.name: sha256(p) for p in (path, root / f"{stem}.pt")}}
    return result


def validate_head(state: dict, outputs: int) -> None:
    import torch

    shapes = {"dense.weight": (480, 480), "dense.bias": (480,),
              "classifier.weight": (outputs, 480), "classifier.bias": (outputs,)}
    if set(state) != set(shapes):
        raise ValueError("Incomplete/unexpected frozen classifier head keys")
    for name, shape in shapes.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid head tensor: {name}")


def plan(args, reference: set[str]) -> list[dict]:
    cells = []
    ref_hash = sha256(args.reference)
    for seed in args.seeds:
        roots = (("hybrid", args.checkpoint_screen if seed == 42 else args.checkpoint_confirm),
                 ("p3_s1", args.blend_screen if seed == 42 else args.blend_confirm))
        for case, root in roots:
            root = root.resolve()
            if root == args.out.resolve() or root in args.out.resolve().parents or args.out.resolve() in root.parents:
                raise ValueError("Output must be separate from, not inside/above, a source sweep")
            run_path = root / "run.json"
            run = json.loads(run_path.read_text())
            expected_kind = "checkpoint_sweep_v1" if case == "hybrid" else "fixed_ratio_blend_v1"
            if run.get("kind") != expected_kind or seed not in run["seeds"] or run["library_size"] != LIBRARY_SIZE:
                raise ValueError(f"Wrong source kind, seed or library size: {root}")
            if case == "hybrid":
                if not any(c["name"] == "hybrid" and c.get("secondary") for c in run["cases"]):
                    raise ValueError("Source control is not the incumbent hybrid")
                recorded_reference = run["reference_sha256"]
            else:
                if [3, 1] not in run["ratios"] or run["primary_case"] != "ckpt_epoch58":
                    raise ValueError("Candidate source must contain the epoch-58 3:1 blend")
                recorded_reference = run["secondary_recipe"]["reference_sha256"]
            if recorded_reference != ref_hash:
                raise ValueError(f"Reference differs from source: {root}")
            directory = root / case / f"seed{seed}"
            generation = checked_stage(directory, "generation.json", {"library.fasta"})
            evaluation = checked_stage(directory, "evaluation.json", {"library.fasta", "metrics.json", "metrics.csv"})
            library = directory / "library.fasta"
            sequences = [seq for _, seq in iter_fasta(library)]
            if len(sequences) != LIBRARY_SIZE or clean_candidates(sequences, reference) != sequences:
                raise ValueError(f"Expected a clean, unique, complete library: {library}")
            cells.append({"case": case, "seed": seed, "library": str(library),
                          "library_sha256": sha256(library), "source_run_sha256": sha256(run_path),
                          "source_run": run, "generation": generation, "evaluation": evaluation})
    return cells


def validate_scores(scores: dict, size: int) -> None:
    if set(scores) != SCORE_KEYS:
        raise ValueError("Missing/unexpected scoring components; no fallback permitted")
    for name, values in scores.items():
        shape = (size, len(PANEL_GENERA)) if name == "panel" else (size,)
        low = -1.00001 if name == "precision" else 0
        high = 1.00001 if name == "precision" else 1
        if (values.shape != shape or not np.isfinite(values).all()
                or (values < low).any() or (values > high).any()):
            raise ValueError(f"Invalid {name} score array")


def combine_scores(scores: dict) -> tuple[np.ndarray, dict]:
    validate_scores(scores, len(scores["activity"]))
    mdr_indices = [i for i, genus in enumerate(PANEL_GENERA) if genus in MDR_PANEL_GENERA]
    parts = {key: scores[key] for key in ("activity", "conformity", "precision")}
    parts["breadth"] = (scores["panel"] > 0.5).mean(axis=1).astype(np.float32)
    parts["mdr"] = (scores["panel"][:, mdr_indices] > 0.5).mean(axis=1).astype(np.float32)
    # Exactly the production per-library z-score combination; not cross-library probabilities.
    scorer = CompositeScorer([(name, weight, lambda _seqs, v=parts[name]: v) for name, weight in WEIGHTS.items()])
    combined, _ = scorer.score([""] * len(scores["activity"]))
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite composite ranking scores")
    return combined, parts


class AuditScorers:
    def __init__(self, reference: list[str], artifacts: dict, out: Path, device: str):
        import torch

        from amp_challenge_2027.training import enable_determinism

        enable_determinism(RANKING_SEED)
        # The shared helper enables warn-only mode; experiments require hard failure.
        torch.use_deterministic_algorithms(True)
        loaded = {}
        for name, loader in (("activity", ActivityScorer.load), ("panel", PanelScorer.load), ("hemolysis", HemoScorer.load)):
            item = artifacts[name]
            state = torch.load(Path(item["directory"]) / f"{item['stem']}.pt", map_location="cpu", weights_only=True)
            validate_head(state, len(PANEL_GENERA) if name == "panel" else 1)
            scorer = loader(device=device)
            if scorer is None or not math.isclose(scorer._temperature, item["config"]["temperature"], rel_tol=1e-9):
                raise RuntimeError(f"Required {name} scorer unavailable or calibration changed")
            loaded[name] = scorer
        self.activity, self.panel, self.hemo = (loaded[k] for k in ("activity", "panel", "hemolysis"))
        self.conformity = ConformityScorer(reference, sample=12000, seed=RANKING_SEED)
        # Do not read or overwrite the legacy shared cache: bind embeddings to this run.
        self.precision = PrecisionProxyScorer(reference, esm_model=PRECISION_MODEL, device=device, cache=False)
        self.precision._cache_path = out / "reference_embeddings.npy"
        self.precision._ensure_model()
        model_info = {name: getattr(scorer._model.esm.config, "_commit_hash", None) for name, scorer in loaded.items()}
        model_info["precision"] = getattr(self.precision._model.config, "_commit_hash", None)
        if not all(isinstance(value, str) and value for value in model_info.values()):
            raise RuntimeError("Cannot record the resolved Hugging Face backbone revisions")
        info_path = out / "backbones.json"
        if info_path.exists() and json.loads(info_path.read_text()) != model_info:
            raise ValueError("Resolved pretrained backbone revision changed; use a new run")
        write_json(info_path, model_info)
        if (out / "reference_cache.json").exists():
            checked_stage(out, "reference_cache.json", {"reference_embeddings.npy", "backbones.json"})
            self.precision._ref_emb = np.load(self.precision._cache_path, allow_pickle=False)
        embeddings = self.precision._reference_embeddings()
        if embeddings.shape != (len(reference), self.precision._model.config.hidden_size) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid reference embedding cache")
        mark_files(out, "reference_cache.json", ["reference_embeddings.npy", "backbones.json"])

    def score_library(self, sequences: list[str]) -> dict:
        result = {}
        for name, fn in (("activity", self.activity.score), ("panel", self.panel._probs),
                         ("conformity", self.conformity.score), ("precision", self.precision.score)):
            print(f"[top100] scoring {name}: {len(sequences)} sequences", flush=True)
            result[name] = np.asarray(fn(sequences))
        validate_scores(result, len(sequences))
        return result

    def risk(self, top: list[str]) -> np.ndarray:
        return self.hemo.p_risky(top)


def audit_top(top: list[str], reference: list[str], risk: np.ndarray) -> tuple[dict, list[float]]:
    from Levenshtein import ratio

    if (len(top) != TOP_K or len(set(top)) != TOP_K or not all(is_plausible(seq) for seq in top)
            or risk.shape != (TOP_K,) or not np.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any()):
        raise ValueError("Incomplete/invalid top selection or hemolysis outputs")
    nearest = [max(ratio(seq, ref) for ref in reference) for seq in top]
    if max(nearest) > TOP_SIMILARITY_THRESHOLD:
        raise ValueError("Top selection failed exact Levenshtein reference novelty check")
    distances = [1 - ratio(seq, other) for i, seq in enumerate(top) for other in top[:i]]
    summary = {"hemo_risk_mean": float(risk.mean()), "hemo_risk_p75": float(np.percentile(risk, 75)),
               "hemo_risk_max": float(risk.max()), "reference_similarity_max": max(nearest),
               "top_mean_pairwise_distance": float(np.mean(distances))}
    return summary, nearest


def paired_summary(rows: list[dict]) -> list[dict]:
    output = []
    for seed in sorted({row["seed"] for row in rows}):
        pair = {row["case"]: row for row in rows if row["seed"] == seed}
        if set(pair) != {"hybrid", "p3_s1"}:
            continue
        a, b = pair["hybrid"], pair["p3_s1"]
        delta = {"seed": seed, "comparison": "p3_s1_minus_hybrid",
                 "top_overlap_count": len(set(a["top_sequences"]) & set(b["top_sequences"]))}
        for name in a:
            if name not in {"seed", "top_size"} and isinstance(a[name], (int, float)):
                delta[name] = b[name] - a[name]
        output.append(delta)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("checkpoint-screen", "checkpoints-seed44-v2"), ("checkpoint-confirm", "checkpoint58-confirm-v1"),
                          ("blend-screen", "epoch58-blends-v1"), ("blend-confirm", "epoch58-blend-confirm-v1")):
        parser.add_argument(f"--{name}", type=Path, default=Path("sweep_results") / default)
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42, 43, 44], default=[42, 43, 44])
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="Validate source hashes/metadata only; output resumability and code/runtime identity are checked when executing")
    args = parser.parse_args(argv)
    args.seeds = list(dict.fromkeys(args.seeds))
    reference = sorted(read_reference_set(args.reference))
    if not reference:
        raise ValueError("Reference must not be empty")
    cells = plan(args, set(reference))
    artifacts = classifier_inputs()
    recipe = {"kind": "paired_top100_v1", "code": code_identity(), "cells": cells,
              "classifiers": artifacts, "reference_sha256": sha256(args.reference), "device": args.device,
              "weights": WEIGHTS, "ranking_seed": RANKING_SEED, "precision_model": PRECISION_MODEL,
              "conformity_sample": 12000, "novelty_candidates": 2000, "top_k": TOP_K,
              "novelty_threshold": TOP_SIMILARITY_THRESHOLD, "hemo_weight": 0.0}
    for cell in cells:
        print(f"[top100] {cell['case']}/seed{cell['seed']}: {cell['library']}", flush=True)
    if args.list:
        return
    # Require the authoritative implementation before selection can use its fallback.
    import Levenshtein  # noqa: F401

    prepare_run(args.out, recipe)
    if (args.out / "reference_cache.json").exists():
        checked_stage(args.out, "reference_cache.json", {"reference_embeddings.npy", "backbones.json"})
    scorers, rows = None, []
    for cell in cells:
        directory = args.out / cell["case"] / f"seed{cell['seed']}"
        source = Path(cell["library"])
        if (directory / "complete.json").exists():
            checked_stage(directory, "complete.json", {"library.fasta", "top.fasta", "top_scores.csv", "summary.json", "scores.npz"})
        else:
            directory.mkdir(parents=True, exist_ok=True)
            if scorers is None:
                scorers = AuditScorers(reference, artifacts, args.out, args.device)
            sequences = [seq for _, seq in iter_fasta(source)]
            if (directory / "scoring.json").exists():
                checked_stage(directory, "scoring.json", {"scores.npz"})
                with np.load(directory / "scores.npz", allow_pickle=False) as data:
                    scores = {key: data[key] for key in data.files}
                validate_scores(scores, len(sequences))
            else:
                scores = scorers.score_library(sequences)
                validate_scores(scores, len(sequences))
                np.savez_compressed(directory / "scores.npz", **scores)
                mark_files(directory, "scoring.json", ["scores.npz"])
            combined, parts = combine_scores(scores)
            print(f"[top100] selecting and auditing {cell['case']}/seed{cell['seed']}", flush=True)
            selected = select_library_and_top(sequences, reference_set=set(reference), scores=combined,
                                              top_k=TOP_K, library_size=LIBRARY_SIZE, seed=RANKING_SEED,
                                              max_novelty_candidates=2000)
            if selected.library != sequences or not set(selected.top).issubset(sequences):
                raise ValueError("Selection changed the source library or escaped its membership")
            risk = np.asarray(scorers.risk(selected.top))
            summary, nearest = audit_top(selected.top, reference, risk)
            index = {seq: i for i, seq in enumerate(sequences)}
            top_indices = [index[seq] for seq in selected.top]
            details = []
            for rank, (seq, i) in enumerate(zip(selected.top, top_indices)):
                props = compute_properties(seq)
                row = {"rank": rank + 1, "sequence": seq, "composite_score": float(combined[i]),
                       **{name: float(values[i]) for name, values in parts.items()},
                       "panel_mean_probability": float(scores["panel"][i].mean()),
                       "hemo_risk": float(risk[rank]), "reference_similarity_max": nearest[rank],
                       "length": props.length, "charge": props.charge,
                       "hydrophobicity_kd": props.hydrophobicity_kd, "hydrophobic_moment": props.hydrophobic_moment}
                row.update({f"p_active:{genus}": float(scores["panel"][i, j]) for j, genus in enumerate(PANEL_GENERA)})
                details.append(row)
            summary.update({f"{name}_mean": float(parts[name][top_indices].mean()) for name in parts})
            summary.update({"panel_mean_probability": float(scores["panel"][top_indices].mean()),
                            "case": cell["case"], "seed": cell["seed"], "top_size": len(selected.top),
                            "library_sha256": cell["library_sha256"], "top_sequences": selected.top})
            if sha256(source) != cell["library_sha256"]:
                raise ValueError("Source library changed during the audit")
            if classifier_inputs() != artifacts:
                raise ValueError("Classifier artifacts changed during the audit")
            shutil.copyfile(source, directory / "library.fasta")
            write_fasta(selected.top, directory / "top.fasta")
            write_summary(directory / "top_scores.csv", details)
            write_json(directory / "summary.json", summary)
            mark_files(directory, "complete.json", ["library.fasta", "top.fasta", "top_scores.csv", "summary.json", "scores.npz"])
        if sha256(source) != cell["library_sha256"] or sha256(directory / "library.fasta") != cell["library_sha256"]:
            raise ValueError("Source or copied library changed")
        rows.append(json.loads((directory / "summary.json").read_text()))
        write_summary(args.out / "results.csv", [{k: v for k, v in row.items() if k != "top_sequences"} for row in rows])
        write_summary(args.out / "paired_deltas.csv", paired_summary(rows))
        print(f"[top100] complete {cell['case']}/seed{cell['seed']}", flush=True)


if __name__ == "__main__":
    main()
