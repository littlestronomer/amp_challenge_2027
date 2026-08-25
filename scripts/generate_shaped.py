"""Standalone generation path with property-distribution shaping.

This is a **parallel** entry point to ``uv run generate`` that adds one extra
step: after generating a candidate pool, it reshapes the library so its
biophysical property marginals (net charge, hydrophobicity, hydrophobic moment)
match the reference AMP set (antibacterial.fasta). Phase-1 Conformity measures
exactly this, and the default "first-N" selection leaves the distribution too
narrow — so this path is the lever for pushing Conformity.

The original ``generate`` entry point (``src/amp_challenge_2027/generate.py``) is
deliberately **left untouched** so the validated submission path stays intact.
This script reuses the original's building blocks via import:

    - ``generate.generate_with_model`` / ``generate_fallback`` — sampling
    - ``generate._score_with_classifier`` — Phase-2 activity scoring
    - ``generate._write_fasta`` / ``_torch_available`` / ``_cuda_available``
    - ``select.select_library_and_top`` — top-100 score+diversity selection
    - ``shape.shape_library_to_reference`` — the new distribution shaper

Outputs (default ``generate_shaped/``):
    library.fasta — 50,000 valid, unique, no-overlap, distribution-shaped peptides
    top.fasta     — top-100 ranked (classifier score + diversity)

Usage:
    uv run python scripts/generate_shaped.py
    uv run python scripts/generate_shaped.py --pool-multiplier 4 --temperature 1.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import (
    ANTIBACTERIAL_FASTA,
    DEFAULT_SEED,
    GENERATOR_DIR,
    LIBRARY_SIZE,
    MAX_LENGTH,
    TOP_K,
)
from amp_challenge_2027.data import read_reference_set

# Reuse the original generation path's building blocks (no modification to it).
from amp_challenge_2027.generate import (  # noqa: F401 (re-exported helpers)
    _cuda_available,
    _score_with_classifier,
    _torch_available,
    _write_fasta,
    generate_fallback,
    generate_with_model,
)
from amp_challenge_2027.select import filter_valid, remove_exact_overlap, select_library_and_top
from amp_challenge_2027.shape import property_stats, shape_library_to_reference

DEFAULT_OUT_DIR = Path("generate_shaped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a distribution-shaped AMP library (parallel to `generate`). "
            "Shapes charge/hydrophobicity/hmoment marginals to match the reference."
        )
    )
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if _torch_available() and _cuda_available() else "cpu",
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=GENERATOR_DIR)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument(
        "--pool-multiplier",
        type=float,
        default=3.0,
        help=(
            "Generate a candidate pool of this many × library_size clean sequences "
            "before shaping (larger → richer pool → better tail-filling)."
        ),
    )
    parser.add_argument("--n-bins", type=int, default=20, help="bins for continuous property axes")
    parser.add_argument(
        "--charge-conditioned",
        action="store_true",
        default=False,
        help=(
            "Generate with per-sequence charge bins drawn from the reference charge "
            "distribution (requires a charge-conditioned checkpoint, trained with "
            "train_generator.py sft --conditioning charge). Widens the pool's charge "
            "distribution to the reference — combines well with the property shaping."
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- 1. Detect trained generator ---------------------------------------
    use_model = (
        _torch_available()
        and args.checkpoint.exists()
        and (args.checkpoint / "config.json").exists()
    )

    # --- 2. Load reference set (overlap + novelty + shaping target) ---------
    reference_set: set[str] = set()
    if ANTIBACTERIAL_FASTA.exists():
        reference_set = read_reference_set(ANTIBACTERIAL_FASTA)
        print(f"[shape] loaded {len(reference_set)} reference sequences (shaping target)")
    else:
        print(f"[shape] WARNING: {ANTIBACTERIAL_FASTA} missing; cannot shape or check overlap")
        print("[shape] aborting — shaping requires the reference set.")
        sys.exit(1)

    # --- 3. Oversample a rich candidate pool (>> library_size) --------------
    # A richer pool gives the shaper more tail sequences to fill under-
    # represented property bins. We accumulate validity+overlap-filtered
    # candidates until we hit pool_target (or exhaust rounds).
    target = args.n_sequences
    pool_target = int(target * args.pool_multiplier)
    all_raw: list[str] = []
    round_idx = 0
    while True:
        round_seed = args.seed + round_idx
        if use_model:
            if round_idx == 0:
                print(f"[shape] using trained AR generator from {args.checkpoint}")
            batch_size = int(pool_target * 1.2) if round_idx == 0 else int(pool_target * 0.5)
            try:
                batch = generate_with_model(
                    batch_size,
                    seed=round_seed,
                    length=args.length,
                    device=args.device,
                    checkpoint_dir=args.checkpoint,
                    temperature=args.temperature,
                    top_k=args.sample_top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    reference_set=reference_set or None,
                    charge_conditioned=args.charge_conditioned or None,
                )
            except Exception as e:
                print(f"[shape] model inference failed ({e}); falling back to seeded sampler")
                use_model = False
                batch = generate_fallback(batch_size, seed=round_seed, length=args.length)
        else:
            if round_idx == 0:
                print("[shape] no trained generator found; using deterministic seeded sampler")
            batch_size = int(pool_target * 1.2) if round_idx == 0 else int(pool_target * 0.5)
            batch = generate_fallback(batch_size, seed=round_seed, length=args.length)

        all_raw.extend(batch)
        round_idx += 1

        # Recompute the clean pool (valid + unique + no-overlap).
        valid = filter_valid(all_raw, drop_duplicates=True)
        shaping_pool = remove_exact_overlap(valid, reference_set)
        if len(shaping_pool) >= pool_target or round_idx > 10:
            break
        print(
            f"[shape] round {round_idx}: {len(shaping_pool)}/{pool_target} clean candidates, "
            "generating more..."
        )

    print(
        f"[shape] generated {len(all_raw)} raw → {len(shaping_pool)} clean candidates "
        f"({round_idx} round(s))"
    )
    if len(shaping_pool) < target:
        print(
            f"[shape] WARNING: clean pool ({len(shaping_pool)}) < target ({target}); "
            "library will be undersized.",
            file=sys.stderr,
        )

    # --- 4. Shape the library to the reference property marginals -----------
    print("[shape] shaping property distribution to reference...")
    ref_list = sorted(reference_set)  # deterministic ordering of the target set
    shaped = shape_library_to_reference(
        shaping_pool,
        ref_list,
        library_size=target,
        seed=args.seed,
        n_bins=args.n_bins,
    )
    print(f"[shape] shaped library: {len(shaped)} sequences")
    print(f"[shape]   pool    : {property_stats(shaping_pool[:5000])}")
    print(f"[shape]   shaped  : {property_stats(shaped)}")
    print(f"[shape]   reference: {property_stats(ref_list[:5000])}")

    # --- 5. Score with the Phase-2 activity classifier ----------------------
    # Build a seq→score map aligned with `shaped` for the top-100 selection.
    seq_to_score: dict[str, float] | None = None
    try:
        lib_scores = _score_with_classifier(shaped, device=args.device)
        if lib_scores is not None:
            print(f"[shape] scored {len(lib_scores)} sequences with activity classifier")
            seq_to_score = dict(zip(shaped, lib_scores))
    except Exception as e:
        print(f"[shape] activity classifier unavailable ({e}); ranking by diversity only")

    scores = [seq_to_score.get(s, 0.0) for s in shaped] if seq_to_score is not None else None

    # --- 6. Top-100 selection (score + diversity) over the shaped library --
    # select_library_and_top re-filters (all pass), keeps `library_size` in the
    # given order (== shaped order), and picks the top-k by score+diversity.
    result = select_library_and_top(
        shaped,
        reference_set=reference_set,
        scores=scores,
        top_k=args.top_k,
        library_size=target,
        seed=args.seed,
    )
    print(
        f"[shape] selection: {len(result.library)} in library, "
        f"{len(result.top)} in top, rejected={result.rejected}"
    )

    # --- 7. Write outputs --------------------------------------------------
    library_path = args.out_dir / "library.fasta"
    top_path = args.out_dir / "top.fasta"
    _write_fasta(result.library, library_path)
    _write_fasta(result.top, top_path)
    print(f"[shape] wrote {len(result.library)} sequences → {library_path}")
    print(f"[shape] wrote top {len(result.top)} sequences → {top_path}")

    if len(result.library) < args.n_sequences:
        print(
            f"[shape] WARNING: library has {len(result.library)} < {args.n_sequences}.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
