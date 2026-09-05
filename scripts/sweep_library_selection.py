"""Compare limited score-based replacements in a fixed 50k baseline library.

Preserves exact counts in every joint length/charge bin. Scores can come from
an explicit sequence,score CSV or the correctly packaged frozen panel head.
Panel scores used for selection are not independent evidence of activity.
All outputs are experiments; submission artifacts are never promoted here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from experiment_utils import (
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_summary,
)

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA, REWARD_DIR
from amp_challenge_2027.data import iter_fasta, read_reference_set, write_fasta
from amp_challenge_2027.library_selection import select_stratified_library
from amp_challenge_2027.pipeline import clean_candidates


def read_scores(path: Path, pool: list[str]) -> np.ndarray:
    scores = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sequence", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("Scores CSV must contain sequence,score columns (higher is better)")
        for row in reader:
            seq = row["sequence"]
            if seq in scores:
                raise ValueError(f"Duplicate sequence in scores CSV: {seq}")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("Scores CSV contains a non-finite score")
            scores[seq] = score
    missing = set(pool) - scores.keys()
    if missing:
        raise ValueError(f"Scores CSV is missing {len(missing)} pool sequences")
    return np.asarray([scores[seq] for seq in pool], dtype=np.float32)


def panel_artifacts(directory: Path) -> dict[str, str]:
    """Reject the known stale shared config rather than guessing calibration."""
    path = directory / "classifier_panel_config.json"
    if not path.exists():
        raise ValueError(
            f"Missing {path}. Recover the actual frozen member's per-artifact config "
            "on the training machine, or supply --scores from an external predictor."
        )
    config = json.loads(path.read_text())
    temperature = float(config.get("temperature", 0))
    if (
        config.get("task") != "panel" or config.get("unfreeze_layers") != 0
        or config.get("checkpoint_format") != "head-only"
        or not math.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError("Panel scoring requires a frozen head and its verified calibration config")
    return {str(p.resolve()): sha256(p) for p in (path, directory / "classifier_panel.pt")}


def score_panel(pool: list[str], directory: Path, device: str) -> np.ndarray:
    import torch

    from amp_challenge_2027.score import PanelScorer
    from amp_challenge_2027.training import enable_determinism

    enable_determinism(42)
    state = torch.load(directory / "classifier_panel.pt", map_location="cpu", weights_only=True)
    expected = {"dense.weight", "dense.bias", "classifier.weight", "classifier.bias"}
    if set(state) != expected:
        raise ValueError("Panel checkpoint does not contain the complete, expected frozen head")
    scorer = PanelScorer.load(device=device, checkpoint_dir=directory)
    if scorer is None:
        raise RuntimeError("Panel scorer failed to load; no fallback is permitted in experiments")
    scores = scorer._probs(pool).mean(axis=1)
    del scorer, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--scores", type=Path, help="Optional sequence,score CSV; otherwise use frozen panel mean probability")
    parser.add_argument("--reward-dir", type=Path, default=REWARD_DIR)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5])
    parser.add_argument("--library-size", type=int, default=50000)
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true", help="Write selections; rerun without this flag for evaluation")
    args = parser.parse_args(argv)
    if args.library_size < 1 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in args.strengths):
        parser.error("library-size must be positive; strengths must be in [0, 1]")
    baseline = [seq for _, seq in iter_fasta(args.baseline)]
    reference = read_reference_set(args.reference)
    if not reference:
        raise ValueError("Reference must not be empty")
    if len(baseline) != args.library_size or clean_candidates(baseline, reference) != baseline:
        raise ValueError("Baseline must have exactly library-size valid, unique, non-reference sequences")
    pool = list(baseline)
    for path in args.pools:
        pool.extend(seq for _, seq in iter_fasta(path))
    pool = clean_candidates(pool, reference)
    score_artifacts = {str(args.scores.resolve()): sha256(args.scores)} if args.scores else panel_artifacts(args.reward_dir)
    recipe = {
        "kind": "library_selection_v1", "code": code_identity(),
        "baseline_sha256": sha256(args.baseline), "reference_sha256": sha256(args.reference),
        "pools": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.pools],
        "score_artifacts": score_artifacts, "score_kind": "external" if args.scores else "panel_mean_probability",
        "strengths": list(dict.fromkeys([0.0, *args.strengths])),
        "library_size": args.library_size, "esm_model": args.esm_model, "device": args.device,
    }
    prepare_run(args.out, recipe)
    scores_path = args.out / "pool_scores.csv"
    if verify_files(args.out, "scoring.json"):
        scores = read_scores(scores_path, pool)
    else:
        print(f"[library-sweep] scoring {len(pool)} candidates", flush=True)
        scores = read_scores(args.scores, pool) if args.scores else score_panel(pool, args.reward_dir, args.device)
        if scores.shape != (len(pool),) or not np.isfinite(scores).all():
            raise ValueError("Scorer produced invalid scores")
        with scores_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sequence", "score"])
            writer.writerows(zip(pool, scores))
        mark_files(args.out, "scoring.json", ["pool_scores.csv"])
    score_by_seq = dict(zip(pool, scores))
    rows = []
    baseline_set = set(baseline)
    for strength in recipe["strengths"]:
        # :g rounds to six significant figures and can merge distinct cells.
        label = str(strength) if strength else "0"
        directory = args.out / f"strength_{label}"
        directory.mkdir(parents=True, exist_ok=True)
        library = select_stratified_library(baseline, pool, scores, strength=strength)
        if not verify_files(directory, "generation.json"):
            if strength == 0:
                # Preserve headers/format as well as order for the anchor hash.
                (directory / "library.fasta").write_bytes(args.baseline.read_bytes())
            else:
                write_fasta(library, directory / "library.fasta")
            mark_files(directory, "generation.json", ["library.fasta"])
        selected_scores = np.asarray([score_by_seq[s] for s in library])
        row = {
            "strength": strength, "library_sha256": sha256(directory / "library.fasta"),
            "replaced": len(set(library) - baseline_set), "pool_size": len(pool),
            "selection_score_mean": float(selected_scores.mean()),
            "selection_score_p10": float(np.quantile(selected_scores, 0.1)),
        }
        if not args.generate_only:
            row.update(evaluate_library(directory, reference=args.reference, esm_model=args.esm_model, device=args.device))
        rows.append(row)
        write_summary(args.out / "results.csv", rows)
        print(f"[library-sweep] strength={strength:g}, replacements={row['replaced']}; selection scores are not independent validation", flush=True)


if __name__ == "__main__":
    main()
