"""Fixed-ratio epoch-checkpoint + charge-conditioned library experiments.

Read primary pools from a completed checkpoint sweep. Generate the conditioned
pool once per seed into a verified, optionally shared cache. Build exact-quota
50k libraries without ranking, fallback generation, or changing model weights.
results.csv contains new blend measurements; controls.csv contains explicitly
labeled, previously recorded controls from the source sweep (not recomputed).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import re
import subprocess
from pathlib import Path

from experiment_utils import (
    REPO_ROOT,
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
    write_summary,
)

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA, CHECKPOINT_DIR
from amp_challenge_2027.data import iter_fasta, read_reference_set, write_fasta
from amp_challenge_2027.generate import generate_with_model
from amp_challenge_2027.library_blending import fixed_ratio_blend, parse_ratio, source_quotas
from amp_challenge_2027.pipeline import clean_candidates

SAMPLING_KEYS = ("raw_count", "temperature", "top_p", "repetition_penalty", "device")
CONTROL_METRICS = ("FBD", "MMD", "Precision", "Recall", "Conformity score", "Diversity", "Authenticity", "FKEA")


def runtime_identity(code: dict) -> dict:
    return {k: v for k, v in code.items() if k not in {"commit", "source_sha256"}}


def check_control_protocol(commit: str) -> None:
    """Recorded controls must use the same tracked local evaluator definition."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source sweep must record a full Git commit for its controls")
    paths = ["scripts/eval_official.py", "scripts/experiment_utils.py", "scripts/sweep_selection.py",
             "src/amp_challenge_2027/metrics_official.py", "src/amp_challenge_2027/data.py", "uv.lock"]
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", commit, "--", *paths], cwd=REPO_ROOT, text=True,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("Cannot verify source control evaluator; its Git revision must be available locally") from error
    if changed:
        raise ValueError(f"Control evaluation code differs; reevaluate source controls before blending:\n{changed}")


def verified_stage(directory: Path, marker: str, required: set[str], optional: set[str] | None = None) -> dict:
    path = directory / marker
    files = json.loads(path.read_text())["files"]
    if not required.issubset(files) or set(files) - required - (optional or set()):
        raise ValueError(f"Unexpected artifact manifest: {path}")
    verify_files(directory, marker)
    return {"marker_sha256": sha256(path), "files": files}


def load_clean_pool(path: Path, reference: set[str]) -> list[str]:
    sequences = [seq for _, seq in iter_fasta(path)]
    if not sequences or clean_candidates(sequences, reference) != sequences:
        raise ValueError(f"Pool must already be valid, unique and reference-free: {path}")
    return sequences


def source_inputs(args, reference: set[str], code: dict) -> tuple[dict, dict[int, list[str]], list[dict]]:
    manifest_path = args.source_sweep / "run.json"
    run = json.loads(manifest_path.read_text())
    if run.get("kind") != "checkpoint_sweep_v1":
        raise ValueError("--source-sweep must be a checkpoint sweep")
    cases = {case["name"]: case for case in run["cases"]}
    if args.primary_case not in cases or cases[args.primary_case].get("secondary"):
        raise ValueError("Primary case must be a standalone generator, not a hybrid")
    if "hybrid" not in cases or not cases["hybrid"].get("secondary"):
        raise ValueError("Source sweep must include the incumbent hybrid control")
    if not set(args.seeds).issubset(run["seeds"]):
        raise ValueError("Every requested seed must exist in the source sweep")
    if run["reference_sha256"] != sha256(args.reference):
        raise ValueError("Reference differs from the source sweep")
    if runtime_identity(run["code"]) != runtime_identity(code):
        raise ValueError("Runtime differs from the source sweep; retain the same environment")
    check_control_protocol(run["code"]["commit"])
    pools, controls, cells = {}, [], {}
    for seed in args.seeds:
        for name in ("hybrid", args.primary_case):
            directory = args.source_sweep / name / f"seed{seed}"
            generation = verified_stage(directory, "generation.json", {
                "library.fasta", "pool.fasta", "generation_stats.json",
            }, {"generation_source.json"})
            evaluation = verified_stage(directory, "evaluation.json", {
                "library.fasta", "metrics.csv", "metrics.json",
            })
            metrics = json.loads((directory / "metrics.json").read_text())
            if any(
                not isinstance(metrics.get(metric), (int, float)) or not math.isfinite(metrics[metric])
                for metric in CONTROL_METRICS
            ):
                raise ValueError(f"Incomplete control measurements: {directory}")
            library = load_clean_pool(directory / "library.fasta", reference)
            if len(library) != run["library_size"]:
                raise ValueError(f"Wrong control library size: {directory}")
            if name == args.primary_case:
                pools[seed] = load_clean_pool(directory / "pool.fasta", reference)
                if pools[seed][:run["library_size"]] != library:
                    raise ValueError("Primary control must be the source pool prefix")
            cells[f"{name}/seed{seed}"] = {"generation": generation, "evaluation": evaluation}
            controls.append({
                "case": name, "seed": seed, "evaluation_origin": "recorded_source_control",
                "source_run": str(args.source_sweep.resolve()), "source_commit": run["code"]["commit"],
                "library_sha256": generation["files"]["library.fasta"], **metrics,
            })
    info = {"directory": str(args.source_sweep.resolve()), "run_sha256": sha256(manifest_path),
            "manifest": run, "cells": cells}
    return info, pools, controls


def conditioned_pool(directory: Path, checkpoint: Path, sampling: dict, reference: set[str], seed: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "generation.json"
    if marker.exists():
        verified_stage(directory, "generation.json", {"pool.fasta", "generation_stats.json"})
    else:
        from amp_challenge_2027.training import enable_determinism

        print(f"[blend-sweep] sampling conditioned pool for seed{seed}; log: {directory / 'generation.log'}", flush=True)
        enable_determinism(seed)
        with (directory / "generation.log").open("w") as log, contextlib.redirect_stdout(log):
            raw = generate_with_model(
                sampling["raw_count"], seed=seed, length=50, device=sampling["device"],
                checkpoint_dir=checkpoint, temperature=sampling["temperature"],
                top_k=50, top_p=sampling["top_p"], repetition_penalty=sampling["repetition_penalty"],
                reference_set=reference, charge_conditioned=True,
            )
            pool = clean_candidates(raw, reference)
            print(f"[blend-sweep] {len(raw)} raw -> {len(pool)} clean conditioned candidates", flush=True)
        if not pool:
            raise RuntimeError("Conditioned generator produced no clean candidates; no fallback is permitted")
        write_fasta(pool, directory / "pool.fasta")
        write_json(directory / "generation_stats.json", {
            "seed": seed, "raw_count": len(raw), "pool_size": len(pool),
        })
        mark_files(directory, "generation.json", ["pool.fasta", "generation_stats.json"])
        if sampling["device"].startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    stats = json.loads((directory / "generation_stats.json").read_text())
    pool = load_clean_pool(directory / "pool.fasta", reference)
    if stats != {"seed": seed, "raw_count": sampling["raw_count"], "pool_size": len(pool)}:
        raise ValueError(f"Conditioned pool metadata differs: {directory}")
    return pool


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sweep", type=Path, required=True, help="Completed checkpoint sweep containing primary and hybrid controls")
    parser.add_argument("--primary-case", default="ckpt_epoch58")
    parser.add_argument("--secondary-checkpoint", type=Path, default=CHECKPOINT_DIR / "generator_blend")
    parser.add_argument("--secondary-cache", type=Path, help="Optional shared conditioned-pool cache; default: OUT/secondary")
    parser.add_argument("--ratios", nargs="+", default=["7:1", "3:1", "1:1"], help="Exact primary:conditioned integer weights")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Generation seeds, already present in the source sweep")
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="Validate inputs and show plan without writing files or loading models")
    parser.add_argument("--generate-only", action="store_true", help="Build blends; rerun the identical command without this flag to evaluate")
    args = parser.parse_args(argv)
    try:
        ratios = list(dict.fromkeys(parse_ratio(value) for value in args.ratios))
    except ValueError as error:
        parser.error(str(error))
    args.seeds = list(dict.fromkeys(args.seeds))
    if args.out.resolve() == args.source_sweep.resolve() or args.source_sweep.resolve() in args.out.resolve().parents:
        parser.error("Use a new --out outside the source sweep; source data must remain unchanged")
    cache = args.secondary_cache or args.out / "secondary"
    if cache.resolve() in {args.out.resolve(), args.source_sweep.resolve()} or args.source_sweep.resolve() in cache.resolve().parents:
        parser.error("Conditioned cache must be separate from the output root and source sweep")
    reference = read_reference_set(args.reference)
    if not reference:
        raise ValueError("Reference must not be empty")
    code = code_identity()
    source, pools, controls = source_inputs(args, reference, code)
    previous = source["manifest"]
    size = previous["library_size"]
    sampling = {key: previous[key] for key in SAMPLING_KEYS}
    config = json.loads((args.secondary_checkpoint / "config.json").read_text())
    if config.get("conditioning") != "charge":
        raise ValueError("Secondary checkpoint must be charge-conditioned")
    secondary_artifacts = {name: sha256(args.secondary_checkpoint / name) for name in ("config.json", "model.pt")}
    for seed in args.seeds:
        for ratio in ratios:
            n_primary, n_secondary = source_quotas(size, ratio)
            if len(pools[seed]) < n_primary:
                raise ValueError(f"Primary pool for seed{seed} cannot meet quota {n_primary}")
            print(f"[blend-sweep] p{ratio[0]}_s{ratio[1]}/seed{seed}: {n_primary} primary + {n_secondary} conditioned", flush=True)
    print(f"[blend-sweep] inherited sampling: {sampling}; embedder: {previous['esm_model']}", flush=True)
    print(f"[blend-sweep] recorded controls: {len(controls)}; conditioned cache: {cache}", flush=True)
    cache_recipe = {
        "kind": "conditioned_pool_cache_v1", "code": code, "artifacts": secondary_artifacts,
        "reference_sha256": previous["reference_sha256"], "sampling": sampling, "length": 50, "top_k": 50,
    }
    recipe = {
        "kind": "fixed_ratio_blend_v1", "code": code, "source": source,
        "primary_case": args.primary_case, "ratios": [list(ratio) for ratio in ratios], "seeds": args.seeds,
        "secondary_cache": str(cache.resolve()), "secondary_recipe": cache_recipe,
        "secondary_checkpoint": str(args.secondary_checkpoint.resolve()), "library_size": size,
        "esm_model": previous["esm_model"], "selection": "primary-prefix-with-secondary-reservation-v1",
    }
    if args.list:
        return
    prepare_run(args.out, recipe)
    prepare_run(cache, cache_recipe)
    write_summary(args.out / "controls.csv", controls)
    rows = []
    for seed in args.seeds:
        primary = pools[seed]
        secondary_dir = cache / f"seed{seed}"
        secondary = conditioned_pool(secondary_dir, args.secondary_checkpoint, sampling, reference, seed)
        pool_hashes = {
            "primary_pool_sha256": source["cells"][f"{args.primary_case}/seed{seed}"]["generation"]["files"]["pool.fasta"],
            "secondary_pool_sha256": sha256(secondary_dir / "pool.fasta"),
        }
        for ratio in ratios:
            case = f"p{ratio[0]}_s{ratio[1]}"
            directory = args.out / case / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if not verify_files(directory, "generation.json"):
                blend = fixed_ratio_blend(primary, secondary, size=size, ratio=ratio)
                write_fasta(blend.sequences, directory / "library.fasta")
                with (directory / "membership.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sequence", "source"])
                    writer.writerows(zip(blend.sequences, blend.sources))
                write_json(directory / "blend_stats.json", {
                    "library_size": size, "primary_count": blend.primary_count,
                    "secondary_count": blend.secondary_count, "primary_pool_size": len(primary),
                    "secondary_pool_size": len(secondary), "shared_pool_count": blend.shared_pool_count,
                    "selected_shared_count": blend.selected_shared_count, **pool_hashes,
                })
                mark_files(directory, "generation.json", ["library.fasta", "membership.csv", "blend_stats.json"])
            stats = json.loads((directory / "blend_stats.json").read_text())
            if any(stats.get(key) != value for key, value in pool_hashes.items()):
                raise ValueError("Component pool changed since this blend was built")
            row = {"case": case, "seed": seed, "primary_fraction": ratio[0] / sum(ratio),
                   "library_sha256": sha256(directory / "library.fasta"), **stats}
            if not args.generate_only:
                row.update(evaluate_library(directory, reference=args.reference, esm_model=previous["esm_model"], device=sampling["device"]))
            rows.append(row)
            write_summary(args.out / "results.csv", rows)
            print(f"[blend-sweep] complete {case}/seed{seed}", flush=True)


if __name__ == "__main__":
    main()
