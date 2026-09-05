"""Compare archived generator checkpoints on the SSH machine, without retraining.

Produces library-only experiments, clean candidate pools, hashes, logs and
650M seqme metrics. It never ranks a top-100 or promotes submission weights.
Sampling calls the trained model directly: no fallback and no implicit blend.
Use --include-hybrid to evaluate the shipped 75/25 baseline explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from experiment_utils import (
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
    write_summary,
)

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA, CHECKPOINT_DIR, GENERATOR_DIR
from amp_challenge_2027.data import read_reference_set, write_fasta
from amp_challenge_2027.generate import generate_with_model
from amp_challenge_2027.pipeline import clean_candidates, interleave_blend


def choose_snapshots(run_dir: Path, *, epochs: list[int] | None, max_snapshots: int) -> list[Path]:
    """Select exact requested epochs, or evenly spaced saved epochs (numeric order)."""
    if max_snapshots < 0:
        raise ValueError("max-snapshots must be nonnegative")
    paths = {}
    for path in (run_dir / "checkpoints").glob("ckpt_epoch*.pt"):
        match = re.fullmatch(r"ckpt_epoch(\d+)\.pt", path.name)
        if match:
            paths[int(match[1])] = path
    if epochs is not None:
        missing = set(epochs) - paths.keys()
        if missing:
            raise ValueError(f"Requested epochs are not saved: {sorted(missing)}")
        return [paths[epoch] for epoch in sorted(set(epochs))]
    ordered = [paths[epoch] for epoch in sorted(paths)]
    if len(ordered) <= max_snapshots:
        return ordered
    # Endpoints of equally sized intervals avoid spending a cell on epoch 1.
    ids = np.linspace(0, len(ordered), max_snapshots + 1, dtype=int)[1:] - 1
    return [ordered[i] for i in ids]


def export_snapshot(source: Path, config: Path, destination: Path) -> None:
    """Export a tensor-only training payload using its parent architecture config."""
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Not a generator training checkpoint: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(payload["model"], destination / "model.pt")
    shutil.copyfile(config, destination / "config.json")
    write_json(destination / "snapshot.json", {
        "source": str(source.resolve()), "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        # Historical best_val describes an earlier evaluation, not necessarily
        # the weights inside this particular snapshot. Do not compare it as loss.
        "recorded_best_val_is_not_snapshot_loss": True,
    })


def sample_pool(checkpoint: Path, reference: set[str], args, seed: int) -> list[str]:
    for name in ("config.json", "model.pt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    raw = generate_with_model(
        args.raw_count, seed=seed, length=50, device=args.device,
        checkpoint_dir=checkpoint, temperature=args.temperature,
        top_k=50, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        reference_set=reference,
    )
    clean = clean_candidates(raw, reference)
    print(f"[checkpoint-sweep] {checkpoint}: {len(raw)} raw -> {len(clean)} clean", flush=True)
    return clean


def generate_case(case: dict, directory: Path, reference: set[str], args, seed: int) -> None:
    from amp_challenge_2027.training import enable_determinism

    enable_determinism(seed)
    checkpoint = Path(case["directory"])
    if case.get("snapshot"):
        checkpoint = directory / "exported_model"
        export_snapshot(Path(case["snapshot"]), Path(case["directory"]) / "config.json", checkpoint)
    pool = sample_pool(checkpoint, reference, args, seed)
    if case.get("secondary"):
        secondary = sample_pool(Path(case["secondary"]), reference, args, seed)
        pool = interleave_blend(pool, secondary, per_primary=3, per_secondary=1, n=len(pool) + len(secondary))
    if len(pool) < args.library_size:
        raise RuntimeError(f"Only {len(pool)} clean candidates; raise --raw-count in a new --out")
    library = pool[:args.library_size]
    write_fasta(pool, directory / "pool.fasta")
    write_fasta(library, directory / "library.fasta")
    write_json(directory / "generation_stats.json", {
        "library_size": len(library), "pool_size": len(pool),
        "raw_count_per_generator": args.raw_count,
    })
    mark_files(directory, "generation.json", ["pool.fasta", "library.fasta", "generation_stats.json"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Training run with config.json and checkpoints/")
    parser.add_argument("--epochs", type=int, nargs="+", help="Exact saved epochs (default: automatic spaced selection)")
    parser.add_argument("--max-snapshots", type=int, default=4, help="Number of archived epochs in automatic mode; 0 = inference only")
    parser.add_argument("--include-hybrid", action="store_true", help="Also evaluate the shipped generators, explicitly blended 3:1")
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Generation seeds, not training seeds")
    parser.add_argument("--library-size", type=int, default=50000)
    parser.add_argument("--raw-count", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="List planned cells without loading models or writing files")
    parser.add_argument("--generate-only", action="store_true", help="Generate pools now; rerun without this flag to evaluate")
    args = parser.parse_args(argv)
    if args.library_size < 1 or args.raw_count < args.library_size:
        parser.error("require raw-count >= library-size > 0")
    snapshots = choose_snapshots(args.checkpoint_dir, epochs=args.epochs, max_snapshots=args.max_snapshots)
    cases = []
    if args.include_hybrid:
        cases.append({"name": "hybrid", "directory": str(GENERATOR_DIR.resolve()),
                      "secondary": str((CHECKPOINT_DIR / "generator_blend").resolve())})
    cases.append({"name": "inference", "directory": str(args.checkpoint_dir.resolve())})
    cases += [{"name": path.stem, "directory": str(args.checkpoint_dir.resolve()),
               "snapshot": str(path.resolve())} for path in snapshots]
    for case in cases:
        for seed in dict.fromkeys(args.seeds):
            print(f"[checkpoint-sweep] {case['name']}/seed{seed}: {case.get('snapshot', case['directory'])}", flush=True)
    if args.list:
        return
    reference = read_reference_set(args.reference)
    if not reference:
        raise ValueError("Reference must not be empty")
    artifacts: dict[str, str] = {}
    for case in cases:
        paths = [Path(case["directory"]) / "config.json", Path(case.get("snapshot", Path(case["directory"]) / "model.pt"))]
        if case.get("secondary"):
            paths += [Path(case["secondary"]) / name for name in ("config.json", "model.pt")]
        artifacts.update({str(path.resolve()): sha256(path) for path in paths})
    recipe = {
        "kind": "checkpoint_sweep_v1", "code": code_identity(), "artifacts": artifacts,
        "cases": cases, "seeds": list(dict.fromkeys(args.seeds)),
        "reference_sha256": sha256(args.reference), "library_size": args.library_size,
        "raw_count": args.raw_count, "temperature": args.temperature, "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty, "device": args.device, "esm_model": args.esm_model,
    }
    prepare_run(args.out, recipe)
    rows = []
    for case in cases:
        for seed in recipe["seeds"]:
            directory = args.out / case["name"] / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if not verify_files(directory, "generation.json"):
                print(f"[checkpoint-sweep] generating {directory}", flush=True)
                with (directory / "generation.log").open("w") as log, contextlib.redirect_stdout(log):
                    generate_case(case, directory, reference, args, seed)
            row = {"case": case["name"], "seed": seed, "library_sha256": sha256(directory / "library.fasta")}
            row.update(json.loads((directory / "generation_stats.json").read_text()))
            if not args.generate_only:
                row.update(evaluate_library(directory, reference=args.reference, esm_model=args.esm_model, device=args.device))
            rows.append(row)
            write_summary(args.out / "results.csv", rows)
            print(f"[checkpoint-sweep] complete {case['name']}/seed{seed}", flush=True)


if __name__ == "__main__":
    main()
