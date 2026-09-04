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
    0. ``--pool <fasta> [...]`` — blend/rerank pre-generated candidate pools
       (one per checkpoint) instead of sampling; see scripts/blend_libraries.py.
    1. Trained AR generator + composite scorer (full pipeline).
    2. Trained AR generator + fallback property scorer.
    3. Deterministic seeded sampler (no torch) — the always-works fallback.

Strategy 3 guarantees ``uv run generate`` succeeds on a fresh clone even before
any model is trained, which is essential for incremental development and for
the validator (which runs on CI with only the light runtime deps installed).

Ranking uses the composite scorer from ``pipeline.build_composite_scorer``
(activity classifier + property-conformity density + embedding-space precision
proxy, weighted by ``--w-*`` flags). Components unavailable in the current
environment are dropped automatically; with none available, selection falls
back to pure diversity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import (
    AMINO_ACIDS,
    ANTIBACTERIAL_FASTA,
    CHECKPOINT_DIR,
    DEFAULT_SEED,
    GENERATE_OUTPUT_DIR,
    GENERATOR_DIR,
    LIBRARY_SIZE,
    MAX_LENGTH,
    MIN_LENGTH,
    TOP_K,
)
from amp_challenge_2027.data import read_reference_set, write_fasta
from amp_challenge_2027.pipeline import (
    DEFAULT_WEIGHTS,
    build_composite_scorer,
    clean_candidates,
    load_pool_fastas,
    score_candidates,
)
from amp_challenge_2027.select import count_clean_candidates, select_library_and_top

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
    n_sequences: int,
    *,
    seed: int,
    length: int,
    device: str,
    checkpoint_dir: Path,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    reference_set: set[str] | None = None,
    charge_conditioned: bool | None = None,
) -> list[str]:
    """Sample from the trained custom generator. Requires the [ml] extra.

    ``charge_conditioned``: ``None`` (default) auto-detects from the
    checkpoint's ``config.json`` (``conditioning == "charge"``). When enabled,
    per-sequence charge bins are drawn from the *reference* charge distribution
    (requires ``reference_set``) so the output reproduces the reference charge
    histogram instead of the generator's narrow default — the fix for the
    under-produced charge tails.
    """

    import torch

    from amp_challenge_2027.generator import load_model, sample_sequences

    model, config = load_model(checkpoint_dir, map_location=device)
    model.to(device)
    model.eval()

    use_charge = (
        getattr(config, "conditioning", "none") == "charge"
        if charge_conditioned is None
        else charge_conditioned
    )
    charge: list[int] | None = None
    if use_charge:
        if getattr(config, "conditioning", "none") != "charge":
            raise ValueError(
                "charge-conditioned generation requested but the checkpoint is unconditional "
                f"({checkpoint_dir}/config.json has conditioning != 'charge')"
            )
        if not reference_set:
            raise ValueError("charge-conditioned generation requires the reference set")
        from amp_challenge_2027.conditioning import reference_charge_proportions, sample_charge_bins

        proportions = reference_charge_proportions(sorted(reference_set))
        charge = sample_charge_bins(n_sequences, proportions, seed=seed)

    g = torch.Generator(device=device).manual_seed(seed)
    sequences = sample_sequences(
        model,
        n_sequences=n_sequences,
        device=device,
        max_length=length,
        generator=g,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        charge=charge,
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
    weights = np.array(
        [
            # A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
            2,
            1,
            1,
            1,
            2,
            3,
            1,
            2,
            5,
            4,
            1,
            1,
            2,
            1,
            4,
            2,
            2,
            3,
            1,
            1,
        ],
        dtype=np.float64,
    )
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
    parser.add_argument(
        "--blend-checkpoint",
        type=Path,
        default=None,
        help="secondary checkpoint blended into the library (default: auto-detect "
        "checkpoint/generator_blend — the locked 75/25 hybrid promotion; absent "
        "→ single-checkpoint generation, byte-identical to the legacy behavior)",
    )
    parser.add_argument(
        "--blend-ratio",
        type=int,
        default=3,
        help="blend interleave ratio: N sequences from the primary checkpoint "
        "per 1 from the blend checkpoint (default 3 = the locked 75/25 hybrid)",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument(
        "--sample-top-k", type=int, default=50, help="top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="nucleus sampling (0 to disable). NOTE: the L0 sweep's 8M-embedder "
        "win for 0.95 (FBD −23%) did NOT survive the 650M gate (0.274 vs 0.269) "
        "— reverted to the original default; see PHASE1_RESULTS.md",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="penalize repeated residues (>1.0, 1.0 to disable)",
    )
    parser.add_argument(
        "--charge-conditioned",
        action="store_true",
        default=False,
        help=(
            "Generate with per-sequence charge bins drawn from the reference charge "
            "distribution (requires a charge-conditioned checkpoint; auto-detected "
            "from config.json when trained with --conditioning charge)."
        ),
    )
    parser.add_argument(
        "--pool",
        type=Path,
        action="append",
        default=[],
        help=(
            "Raw candidate FASTA to rerank instead of sampling (repeatable, one per "
            "generator checkpoint). When set, no model inference runs."
        ),
    )
    parser.add_argument(
        "--pool-cap-per-source",
        type=int,
        default=0,
        help="Cap candidates taken per --pool source after a seeded shuffle (0 = unlimited).",
    )
    parser.add_argument(
        "--w-activity",
        type=float,
        default=DEFAULT_WEIGHTS["activity"],
        help="composite-score weight for the activity classifier component",
    )
    parser.add_argument(
        "--w-conformity",
        type=float,
        default=DEFAULT_WEIGHTS["conformity"],
        help="composite-score weight for the property-conformity density component",
    )
    parser.add_argument(
        "--w-precision",
        type=float,
        default=DEFAULT_WEIGHTS["precision"],
        help="composite-score weight for the embedding kNN precision proxy",
    )
    parser.add_argument(
        "--w-breadth",
        type=float,
        default=DEFAULT_WEIGHTS["breadth"],
        help="composite-score weight for panel activity breadth "
        "(requires checkpoint/reward/classifier_panel.pt; 0 = off)",
    )
    parser.add_argument(
        "--w-mdr",
        type=float,
        default=DEFAULT_WEIGHTS["mdr"],
        help="composite-score weight for MDR-genus breadth (shares the panel "
        "classifier forward pass with --w-breadth; 0 = off)",
    )
    parser.add_argument(
        "--w-safety",
        type=float,
        default=DEFAULT_WEIGHTS["safety"],
        help="composite-score weight for hemolysis SAFETY = 1 - p(risky) "
        "(requires checkpoint/reward_hemo/classifier.pt; 0 = off)",
    )
    parser.add_argument(
        "--conformity-sample",
        type=int,
        default=12000,
        help="reference sequences subsampled for the KDE density estimate (0 = all)",
    )
    parser.add_argument(
        "--precision-esm",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM-2 model for the precision proxy embeddings (8M fast, 650M fidelity)",
    )
    parser.add_argument(
        "--novelty-candidates",
        type=int,
        default=2000,
        help=(
            "score-ranked shortlist size fed to novelty screening + diversity "
            "selection (0 = screen the whole pool; much slower)"
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)  # belt-and-suspenders for any global-RNG callers

    # --- 0. Reference set (for no-overlap, novelty, and conformity) --------
    reference_set: set[str] = set()
    if ANTIBACTERIAL_FASTA.exists():
        reference_set = read_reference_set(ANTIBACTERIAL_FASTA)
        print(f"[generate] loaded {len(reference_set)} reference sequences for filtering/scoring")
    else:
        print(f"[generate] WARNING: {ANTIBACTERIAL_FASTA} missing; skipping overlap/novelty checks")

    # --- 1. Raw candidates: pool files (blend/rerank) or model sampling -----
    target = args.n_sequences
    if args.pool:
        all_sequences = load_pool_fastas(
            args.pool, cap_per_source=args.pool_cap_per_source, seed=args.seed
        )
        print(f"[generate] {len(all_sequences)} raw candidates from {len(args.pool)} pool file(s)")
    else:
        # Detect a trained model: <checkpoint>/config.json (+ weights).
        use_model = (
            _torch_available()
            and args.checkpoint.exists()
            and (args.checkpoint / "config.json").exists()
        )
        # Blend auto-detect: the repo ships the locked-hybrid secondary at a
        # stable path; its PRESENCE defines the default output (a fresh clone
        # reproduces the hybrid; without it, legacy single-checkpoint bytes).
        if args.blend_checkpoint is None:
            auto = CHECKPOINT_DIR / "generator_blend"
            args.blend_checkpoint = auto if (auto / "config.json").exists() else None
        all_sequences = _sample_candidates(args, use_model, reference_set, target)

    clean = clean_candidates(all_sequences, reference_set)
    print(
        f"[generate] {len(clean)} clean candidates "
        f"(valid + unique + no exact overlap; from {len(all_sequences)} raw)"
    )

    # --- 2. Composite scoring -----------------------------------------------
    scorer = (
        build_composite_scorer(
            sorted(reference_set),
            w_activity=args.w_activity,
            w_conformity=args.w_conformity,
            w_precision=args.w_precision,
            w_breadth=args.w_breadth,
            w_mdr=args.w_mdr,
            w_safety=args.w_safety,
            device=args.device,
            conformity_sample=args.conformity_sample,
            precision_esm_model=args.precision_esm,
            seed=args.seed,
        )
        if (
            args.w_conformity
            or args.w_activity
            or args.w_precision
            or args.w_breadth
            or args.w_mdr
            or args.w_safety
        )
        else None
    )
    combined, parts = score_candidates(scorer, clean)
    if scorer is None:
        print("[generate] no scoring components available; ranking by diversity only")

    # --- 3. Final selection ---------------------------------------------------
    result = select_library_and_top(
        clean,
        reference_set=reference_set,
        scores=combined,
        top_k=args.top_k,
        library_size=target,
        seed=args.seed,
        max_novelty_candidates=args.novelty_candidates,
    )
    print(f"[generate] selection: {len(result.library)} in library, {len(result.top)} in top")

    # --- 4. Write outputs --------------------------------------------------
    library_path = args.out_dir / "library.fasta"
    top_path = args.out_dir / "top.fasta"
    write_fasta(result.library, library_path)
    write_fasta(result.top, top_path)
    print(f"[generate] wrote {len(result.library)} sequences → {library_path}")
    print(f"[generate] wrote top {len(result.top)} sequences → {top_path}")

    if len(result.library) < args.n_sequences:
        print(
            f"[generate] WARNING: library has {len(result.library)} < {args.n_sequences}; "
            "consider increasing --n-sequences or adding --pool files.",
            file=sys.stderr,
        )


def _sample_candidates(
    args: argparse.Namespace,
    use_model: bool,
    reference_set: set[str],
    target: int,
) -> list[str]:
    """Sample raw candidates from the trained generator (or fallback sampler).

    Oversamples in rounds until the CLEAN candidate count hits the target.
    Each round checks the cheap validity/dedup/overlap count instead of running
    full selection, so rounds cost almost nothing.

    With a blend checkpoint active, BOTH streams are sampled (primary at
    ``seed``, secondary at the same round seed — its own generator state, so
    each stream reproduces the corresponding standalone run's candidate
    order), cleaned separately, and interleaved per ``--blend-ratio``. The
    clean() prefix property (order-preserving dedup) makes the interleaved
    library byte-identical to blending two standalone libraries.
    """
    blend_dir = getattr(args, "blend_checkpoint", None)
    per_b = int(getattr(args, "blend_ratio", 3))
    use_blend = (
        blend_dir is not None and use_model and (blend_dir / "config.json").exists() and per_b >= 1
    )
    if blend_dir is not None and not (blend_dir / "config.json").exists():
        print(f"[generate] blend checkpoint missing config ({blend_dir}); blending disabled")

    if not use_blend:
        return _sample_single(args, use_model, reference_set, target)

    print(
        f"[generate] blend: {per_b}:1 primary {args.checkpoint.name} + secondary {blend_dir.name}"
    )
    primary_raw = _sample_from(args.checkpoint, args, reference_set, target * 2)
    secondary_n = max(int(target * 2 * 1 / (per_b + 1)), target)
    secondary_raw = _sample_from(blend_dir, args, reference_set, secondary_n)

    from amp_challenge_2027.pipeline import clean_candidates, interleave_blend

    primary_clean = clean_candidates(primary_raw, reference_set)
    secondary_clean = clean_candidates(secondary_raw, reference_set)
    blended = interleave_blend(
        primary_clean, secondary_clean, per_primary=per_b, per_secondary=1, n=target
    )
    print(
        f"[generate] blended {len(blended)} candidates "
        f"(primary {len(primary_clean)} clean, secondary {len(secondary_clean)} clean)"
    )
    return blended


def _sample_from(
    checkpoint_dir: Path, args: argparse.Namespace, reference_set: set[str], n: int
) -> list[str]:
    """One sampling pass from ``checkpoint_dir`` using the shared decode args."""
    return generate_with_model(
        n,
        seed=args.seed,
        length=args.length,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        temperature=args.temperature,
        top_k=args.sample_top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        reference_set=reference_set or None,
        charge_conditioned=None,  # auto-detect from each checkpoint's config
    )


def _sample_single(
    args: argparse.Namespace,
    use_model: bool,
    reference_set: set[str],
    target: int,
) -> list[str]:
    all_sequences: list[str] = []
    round_idx = 0
    while True:
        round_seed = args.seed + round_idx
        if use_model:
            if round_idx == 0:
                print(f"[generate] using trained AR generator from {args.checkpoint}")
            # Generate enough to cover the shortfall plus margin.
            batch_size = int(target * 2) if round_idx == 0 else int(target * 0.5)
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
                print(f"[generate] model inference failed ({e}); falling back to seeded sampler")
                use_model = False
                batch = generate_fallback(batch_size, seed=round_seed, length=args.length)
        else:
            if round_idx == 0:
                print("[generate] no trained generator found; using deterministic seeded sampler")
            batch_size = int(target * 2) if round_idx == 0 else int(target * 0.5)
            batch = generate_fallback(batch_size, seed=round_seed, length=args.length)

        all_sequences.extend(batch)
        round_idx += 1

        clean_count = count_clean_candidates(all_sequences, reference_set)
        if clean_count >= target or round_idx > 6:
            break
        print(f"[generate] round {round_idx}: {clean_count}/{target} clean, generating more...")

    print(f"[generate] generated {len(all_sequences)} raw candidates ({round_idx} round(s))")
    return all_sequences


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
