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
    import json

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
        n_sequences=n_sequences, device=device, max_length=length, generator=g,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty,
        charge=charge,
    )
    return sequences


def _score_with_classifier(sequences: list[str], *, device: str = "cpu") -> list[float] | None:
    """Score sequences using the trained activity classifier.

    Loads the Phase-2 binary classifier from checkpoint/reward/ and returns
    activity probabilities (0=inactive, 1=active). Falls back to None if no
    classifier is found, so generate.py can use the property heuristic instead.
    """
    import json

    from amp_challenge_2027.config import REWARD_DIR

    ckpt_path = REWARD_DIR / "classifier.pt"
    config_path = REWARD_DIR / "config.json"
    if not ckpt_path.exists() or not config_path.exists():
        return None

    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    config = json.loads(config_path.read_text())
    esm_id = config["esm_model"]
    tokenizer = AutoTokenizer.from_pretrained(esm_id)
    esm = AutoModel.from_pretrained(esm_id).to(device).eval()
    hidden = esm.config.hidden_size

    class _Classifier(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.2)
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            x = self.drop(self.act(self.dense(pooled)))
            x = self.drop(self.act(self.dense(x)))
            return self.classifier(x).squeeze(-1)

    model = _Classifier(hidden)
    model.esm = esm
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    scores: list[float] = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
    return scores


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
        "--repetition-penalty", type=float, default=1.3,
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
    args = parser.parse_args()

    np.random.seed(args.seed)  # belt-and-suspenders for any global-RNG callers

    # --- 1. Generate raw candidates ----------------------------------------
    # Detect a trained model: <checkpoint>/config.json (+ weights).
    use_model = (
        _torch_available()
        and args.checkpoint.exists()
        and (args.checkpoint / "config.json").exists()
    )

    # --- 2. Load reference set (for no-overlap + novelty) ------------------
    reference_set: set[str] = set()
    if ANTIBACTERIAL_FASTA.exists():
        reference_set = read_reference_set(ANTIBACTERIAL_FASTA)
        print(f"[generate] loaded {len(reference_set)} reference sequences for novelty checks")
    else:
        print(f"[generate] WARNING: {ANTIBACTERIAL_FASTA} missing; skipping overlap/novelty checks")

    # Oversample in rounds until the CLEAN library hits the target size.
    # Each round generates 2× the shortfall, filters, and accumulates.
    # This is critical for the validator's 50k requirement — the rejection
    # rate varies with model quality and repetition penalty.
    target = args.n_sequences
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

        # Check if we have enough clean sequences.
        result = select_library_and_top(
            all_sequences,
            reference_set=reference_set,
            scores=None,
            top_k=args.top_k,
            library_size=target,
            seed=args.seed,
        )
        if len(result.library) >= target or round_idx > 6:
            break
        print(f"[generate] round {round_idx}: {len(result.library)}/{target} clean, generating more...")

    print(f"[generate] generated {len(all_sequences)} raw candidates ({round_idx} round(s))")

    # --- 3. Score with trained activity classifier --------------------------
    # Score the clean library, then build a seq→score mapping that aligns
    # with all_sequences for the final selection pass.
    seq_to_score: dict[str, float] | None = None
    try:
        lib_scores = _score_with_classifier(result.library, device=args.device)
        if lib_scores is not None:
            print(f"[generate] scored {len(lib_scores)} sequences with activity classifier")
            seq_to_score = dict(zip(result.library, lib_scores))
    except Exception as e:
        print(f"[generate] activity classifier unavailable ({e}); using fallback scorer")
        try:
            from amp_challenge_2027.reward import RewardEnsemble

            ensemble = RewardEnsemble.load_or_fallback(device="cpu")
            tag = "trained reward ensemble" if ensemble.is_trained else "fallback property scorer"
            print(f"[generate] scoring with {tag}")
            outputs = ensemble.score_batch(result.library)
            seq_to_score = {s: o.score for s, o in zip(result.library, outputs)}
        except Exception as e2:
            print(f"[generate] scoring unavailable ({e2}); ranking by diversity only")

    # --- 4. Final selection with scores ------------------------------------
    # Map scores back to all_sequences via the seq→score dict.
    if seq_to_score is not None:
        scores = [seq_to_score.get(s, 0.0) for s in all_sequences]
        result = select_library_and_top(
            all_sequences,
            reference_set=reference_set,
            scores=scores,
            top_k=args.top_k,
            library_size=target,
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
            "consider increasing --n-sequences.",
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
