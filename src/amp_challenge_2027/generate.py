"""Entry point: ``uv run generate`` → ``generate/library.fasta`` + ``generate/top.fasta``.

Reproducibility contract (verified by ``scripts/verify_submission.py``):
    Two runs with identical arguments produce byte-identical output. We achieve
    this by (a) seeding every RNG from a single integer, (b) writing outputs in
    sorted/stable order, and (c) falling back to a deterministic numpy-based
    generator when trained weights are absent, so the entry point always runs.

Outputs:
    generate/
      library.fasta  — 50,000 valid, unique, no-overlap peptides
      top.fasta      — top-100 ranked, plausible, novel (≤80% identity)

The generation strategy in priority order:
    1. Trained AR generator + trained reward ensemble (full pipeline).
    2. Trained AR generator + fallback property scorer.
    3. Deterministic seeded sampler (no torch) — the always-works fallback.

Strategy 3 guarantees ``uv run generate`` succeeds on a fresh clone even before
any model is trained, which is essential for incremental development and for
the validator (which runs on CI with only the light runtime deps installed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import (
    AMINO_ACIDS,
    ANTIBACTERIAL_FASTA,
    DEFAULT_SEED,
    GENERATE_OUTPUT_DIR,
    GENERATOR_DIR,
    LIBRARY_SIZE,
    MAX_LENGTH,
    MIN_LENGTH,
    TOP_K,
)
from amp_challenge_2027.data import read_reference_set
from amp_challenge_2027.select import select_library_and_top

# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_fasta(sequences: list[str], path: Path) -> None:
    """Write sequences to a FASTA file with stable ``>seq{i}`` headers.

    Writing order is the list order; selection guarantees that order is a pure
    function of (seed, inputs), so output is reproducible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


# ---------------------------------------------------------------------------
# Generation strategies
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def generate_with_model(
    n_sequences: int, *, seed: int, length: int, device: str, checkpoint_dir: Path,
    temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> list[str]:
    """Sample from the trained custom generator. Requires the [ml] extra."""
    import torch

    from amp_challenge_2027.generator import load_model, sample_sequences

    model, _config = load_model(checkpoint_dir, map_location=device)
    model.to(device)
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    sequences = sample_sequences(
        model,
        n_sequences=n_sequences, device=device, max_length=length, generator=g,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return sequences


def generate_fallback(n_sequences: int, *, seed: int, length: int) -> list[str]:
    """Deterministic seeded sampler with no heavy dependencies.

    Samples an AMP-like composition (cationic + hydrophobic enriched, matching
    the broad AMP property profile) and shuffles deterministically. This is a
    placeholder generator that exists so the pipeline runs before a model is
    trained — it is NOT competitive. It will be replaced by the trained model
    once weights exist in ``checkpoint/generator/``.
    """
    rng = np.random.default_rng(seed)
    # AMP-enriched alphabet: higher weight on K, R (cationic) and L, A, G, V, I.
    alphabet = list(AMINO_ACIDS)
    weights = np.array([
        # A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
        2, 1, 1, 1, 2, 3, 1, 2, 5, 4, 1, 1, 2, 1, 4, 2, 2, 3, 1, 1,
    ], dtype=np.float64)
    weights /= weights.sum()

    sequences: list[str] = []
    # Vary lengths uniformly across [MIN_LENGTH, MAX_LENGTH] for diversity.
    lengths = rng.integers(MIN_LENGTH, MAX_LENGTH + 1, size=n_sequences)
    for n in lengths:
        seq = "".join(rng.choice(alphabet, size=int(n), p=weights))
        sequences.append(seq)
    return sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AMP library + ranked top-100.")
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=GENERATE_OUTPUT_DIR,
        help="output directory for library.fasta and top.fasta",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=GENERATOR_DIR,
        help="trained generator checkpoint directory (default: checkpoint/generator)",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--sample-top-k", type=int, default=50, help="top-k sampling (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling (0 to disable)")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.2,
        help="penalize repeated residues (>1.0, 1.0 to disable)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)  # belt-and-suspenders for any global-RNG callers

    # --- 1. Generate raw candidates ----------------------------------------
    # Detect a trained model: <checkpoint>/config.json (+ weights).
    use_model = (
        _torch_available()
        and args.checkpoint.exists()
        and (args.checkpoint / "config.json").exists()
    )
    if use_model:
        print(f"[generate] using trained AR generator from {args.checkpoint}")
        try:
            sequences = generate_with_model(
                args.n_sequences,
                seed=args.seed,
                length=args.length,
                device=args.device,
                checkpoint_dir=args.checkpoint,
                temperature=args.temperature,
                top_k=args.sample_top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as e:
            print(f"[generate] model inference failed ({e}); falling back to seeded sampler")
            sequences = generate_fallback(args.n_sequences, seed=args.seed, length=args.length)
    else:
        print("[generate] no trained generator found; using deterministic seeded sampler")
        sequences = generate_fallback(args.n_sequences, seed=args.seed, length=args.length)

    # --- 2. Load reference set (for no-overlap + novelty) ------------------
    reference_set: set[str] = set()
    if ANTIBACTERIAL_FASTA.exists():
        reference_set = read_reference_set(ANTIBACTERIAL_FASTA)
        print(f"[generate] loaded {len(reference_set)} reference sequences for novelty checks")
    else:
        print(f"[generate] WARNING: {ANTIBACTERIAL_FASTA} missing; skipping overlap/novelty checks")

    # --- 3. Score (reward ensemble or fallback) ----------------------------
    scores: list[float] | None = None
    try:
        from amp_challenge_2027.reward import RewardEnsemble

        ensemble = RewardEnsemble.load_or_fallback(device="cpu")
        tag = "trained reward ensemble" if ensemble.is_trained else "fallback property scorer"
        print(f"[generate] scoring with {tag}")
        outputs = ensemble.score_batch(sequences)
        scores = [o.score for o in outputs]
    except Exception as e:
        print(f"[generate] scoring unavailable ({e}); ranking by diversity only")

    # --- 4. Select compliant library + diversity-ranked top-k -------------
    result = select_library_and_top(
        sequences,
        reference_set=reference_set,
        scores=scores,
        top_k=args.top_k,
        library_size=args.n_sequences,
        seed=args.seed,
    )
    print(
        f"[generate] selection: {len(result.library)} in library, "
        f"{len(result.top)} in top, rejected={result.rejected}"
    )

    # --- 5. Write outputs --------------------------------------------------
    library_path = args.out_dir / "library.fasta"
    top_path = args.out_dir / "top.fasta"
    _write_fasta(result.library, library_path)
    _write_fasta(result.top, top_path)
    print(f"[generate] wrote {len(result.library)} sequences → {library_path}")
    print(f"[generate] wrote top {len(result.top)} sequences → {top_path}")

    if len(result.library) < args.n_sequences:
        print(
            f"[generate] WARNING: library has {len(result.library)} < {args.n_sequences}; "
            "generate more raw candidates or relax filters.",
            file=sys.stderr,
        )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
