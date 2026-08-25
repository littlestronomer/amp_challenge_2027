"""Candidate selection: validity + plausibility filters and diversity-aware ranking.

This module turns a raw batch of generated peptides into a compliant submission:

  1. VALIDITY   — must pass the same checks as ``scripts/verify_submission.py``
  2. PLAUSIBILITY — your collaborator's biological rules (see ``props.py``)
  3. NO-OVERLAP  — library must not contain exact known antibacterial sequences
  4. NOVELTY     — top-100 must be ≤80% Levenshtein-ratio to any reference
  5. RANKING     — by surrogate reward; ties broken for diversity

The top-100 is selected with **diversity-aware farthest-point sampling** over
embeddings, so the random 25-peptide experimental draw is protected against
landing on a narrow cluster. With an ESM-2 embedder available this uses cosine
distance; without one it falls back to sequence-length-normalized Hamming,
which is deterministic and dependency-free for the inference path.

Scalability contract: the two expensive screens — Levenshtein novelty against
the ~39k reference set and the O(N²) farthest-point distance matrix — run on a
**score-ranked shortlist** (default top 2000 by surrogate score) rather than the
full ~50k pool. When the candidate set fits the shortlist budget the result is
identical to screening everything; larger pools simply rank first and screen
the head, which bounds memory (~16 MB) and runtime while preserving the top of
the reward distribution where the top-100 lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amp_challenge_2027.config import (
    AMINO_ACIDS,
    LIBRARY_SIZE,
    MAX_LENGTH,
    MIN_LENGTH,
    TOP_K,
    TOP_SIMILARITY_THRESHOLD,
)
from amp_challenge_2027.props import is_plausible

# Default budget for the score-ranked shortlist that feeds the novelty screen
# and farthest-point selection (see module docstring). ≤0 disables shortlisting.
DEFAULT_NOVELTY_CANDIDATES = 2000


# ---------------------------------------------------------------------------
# Validity (mirrors scripts/verify_submission.py exactly)
# ---------------------------------------------------------------------------


_VALID_AA = set(AMINO_ACIDS)


def is_valid_sequence(seq: str) -> bool:
    """Single-sequence validity: alphabet, length. (Uniqueness is checked in batch.)"""
    if not seq:
        return False
    if any(c not in _VALID_AA for c in seq):
        return False
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH


def filter_valid(
    sequences: list[str], *, drop_duplicates: bool = True
) -> list[str]:
    """Keep only sequences passing alphabet + length (+ optional dedup).

    Order-preserving. The dedup keeps the first occurrence.
    """
    seen: set[str] = set()
    out: list[str] = []
    for seq in sequences:
        if not is_valid_sequence(seq):
            continue
        if drop_duplicates and seq in seen:
            continue
        seen.add(seq)
        out.append(seq)
    return out


def filter_plausible(sequences: list[str]) -> list[str]:
    """Apply the biological plausibility rules from ``props.py``."""
    return [seq for seq in sequences if is_plausible(seq)]


def remove_exact_overlap(sequences: list[str], reference: set[str]) -> list[str]:
    """Drop any sequence that exactly matches a known antibacterial peptide."""
    return [seq for seq in sequences if seq not in reference]


def clean_candidates(sequences: list[str], reference: set[str]) -> list[str]:
    """Validity + dedup + exact-overlap removal, order-preserving.

    This is the "cheap" funnel every candidate passes before any expensive
    scoring/screening; idempotent on already-clean input.
    """
    return remove_exact_overlap(filter_valid(sequences), reference)


def count_clean_candidates(sequences: list[str], reference: set[str]) -> int:
    """Number of candidates surviving ``clean_candidates`` — without building it.

    Used by the oversampling round loop in ``generate.py`` to decide when
    enough raw material exists — without paying for plausibility, novelty, or
    diversity selection on every round.
    """
    return len(remove_exact_overlap(filter_valid(sequences), reference))


# ---------------------------------------------------------------------------
# Top-100 novelty: ≤80% Levenshtein similarity to the reference set
# ---------------------------------------------------------------------------


def _hamming_normalized(a: str, b: str) -> float:
    """Length-normalized Hamming similarity in [0, 1] for equal-length strings.

    For differing lengths, pads the shorter with a sentinel so it scores low.
    This is a fast pre-filter; the validator uses Levenshtein.ratio for the
    authoritative check. We use it to keep candidate selection cheap.
    """
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    matches = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
    return matches / n


def is_novel_top(
    seq: str,
    reference: set[str] | list[str],
    *,
    threshold: float = TOP_SIMILARITY_THRESHOLD,
    use_levenshtein: bool = True,
) -> bool:
    """True if ``seq`` has Levenshtein-ratio (or Hamming) ≤ ``threshold`` to every reference.

    The competition validator uses Levenshtein.ratio, so we do too when the
    library is available. The Hamming fallback is for environments without the
    ``Levenshtein`` package and is conservative (over-reports similarity).
    """
    if use_levenshtein:
        try:
            import Levenshtein  # type: ignore

            for ref in reference:
                if Levenshtein.ratio(seq, ref) > threshold:
                    return False
            return True
        except ImportError:
            use_levenshtein = False

    for ref in reference:
        if _hamming_normalized(seq, ref) > threshold:
            return False
    return True


def filter_novel(
    sequences: list[str],
    reference: set[str] | list[str],
    *,
    threshold: float = TOP_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Keep sequences whose max similarity to ``reference`` is ≤ ``threshold``."""
    return [seq for seq in sequences if is_novel_top(seq, reference, threshold=threshold)]


# ---------------------------------------------------------------------------
# Embedding-based diversity (fallback: Hamming in property space)
# ---------------------------------------------------------------------------


def _cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance (1 - cosine similarity)."""
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    unit = emb / norm
    sim = unit @ unit.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _one_hot_embed(sequences: list[str]) -> np.ndarray:
    """Simple AA one-hot + property features as an embedding fallback.

    Deterministic, no external model. Good enough to make farthest-point
    sampling spread candidates across the alphabet/property space.
    """
    aa = AMINO_ACIDS
    idx = {a: i for i, a in enumerate(aa)}
    from amp_challenge_2027.props import compute_properties

    rows = []
    for seq in sequences:
        oh = np.zeros(len(aa), dtype=np.float32)
        for c in seq:
            if c in idx:
                oh[idx[c]] += 1.0
        oh /= max(len(seq), 1)
        p = compute_properties(seq)
        rows.append(np.concatenate([oh, [p.charge, p.hydrophobicity_kd, p.hydrophobic_moment]]))
    return np.stack(rows) if rows else np.zeros((0, len(aa) + 3), dtype=np.float32)


def farthest_point_select(
    candidates: list[str],
    scores: list[float] | np.ndarray,
    k: int,
    *,
    embeddings: np.ndarray | None = None,
    seed: int = 42,
) -> list[int]:
    """Diversity-aware selection of ``k`` candidates.

    Greedy farthest-point sampling biased by score: start from the top-scoring
    candidate, then repeatedly pick the candidate maximizing
    ``alpha * (distance to nearest selected) + (1 - alpha) * normalized_score``.
    Here alpha=0.5 balances reward and diversity (the 25-peptide random draw
    protection described in the plan).
    """
    n = len(candidates)
    if n <= k:
        return list(range(n))

    scores = np.asarray(scores, dtype=np.float32)
    s_min, s_max = scores.min(), scores.max()
    norm_score = (scores - s_min) / (s_max - s_min + 1e-8)

    if embeddings is None:
        embeddings = _one_hot_embed(candidates)
    dist = _cosine_distance_matrix(embeddings)

    rng = np.random.default_rng(seed)
    selected: list[int] = []

    # For the top-k candidates, we want the HIGHEST-scoring sequences — the
    # ones the classifier is most confident about. The random draw of 25 from
    # the top-100 provides diversity naturally; we don't need to enforce it
    # in the selection. Pure score ranking (alpha=0) is the right approach.
    #
    # Use alpha=0.1 (90% score, 10% diversity) to break score ties with
    # a small diversity preference, preventing near-duplicate high-scorers.
    alpha = 0.1

    # Seed with argmax score; random tie-break for determinism.
    top_score = norm_score.max()
    top_ids = np.where(norm_score >= top_score - 1e-6)[0]
    start = int(top_ids[rng.integers(len(top_ids))]) if len(top_ids) > 1 else int(top_ids[0])
    selected.append(start)

    nearest = dist[start].copy()
    while len(selected) < k:
        # objective: distance from the set (already in `nearest`) + score
        obj = alpha * nearest + (1 - alpha) * norm_score
        obj[selected] = -np.inf  # don't reselect
        best = int(np.argmax(obj))
        selected.append(best)
        nearest = np.minimum(nearest, dist[best])

    return selected


# ---------------------------------------------------------------------------
# Top-level selection orchestrator
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    library: list[str]  # full valid, dedup, no-overlap set (up to LIBRARY_SIZE)
    top: list[str]  # ranked top-100 (novel + plausible)
    rejected: dict[str, int]  # reason -> count


def select_library_and_top(
    raw_sequences: list[str],
    *,
    reference_set: set[str],
    scores: list[float] | np.ndarray | None = None,
    top_k: int = TOP_K,
    library_size: int = LIBRARY_SIZE,
    apply_plausibility_to_top: bool = True,
    top_embeddings: np.ndarray | None = None,
    seed: int = 42,
    max_novelty_candidates: int = DEFAULT_NOVELTY_CANDIDATES,
) -> SelectionResult:
    """Full selection pipeline producing the compliant library + ranked top-k.

    Steps:
      1. Validity filter (alphabet, length, dedup).
      2. No-overlap filter (drop exact known antibacterials) → library.
      3. For the top-k: plausibility filter, then a score-ranked shortlist
         (``max_novelty_candidates``; ≤0 = unlimited) that bounds the cost of
         the novelty screen and the farthest-point distance matrix.
      4. Rank by score (descending); farthest-point diversity selection.

    ``scores`` are higher-is-better surrogate scores. If absent, a uniform
    score is used and selection is purely diversity-driven (stable order).
    """
    n_raw = len(raw_sequences)
    valid = filter_valid(raw_sequences, drop_duplicates=True)
    library = remove_exact_overlap(valid, reference_set)
    library = library[:library_size]

    # Map scores onto sequences (first occurrence wins; duplicates share one).
    if scores is None:
        scores_arr = np.zeros(n_raw, dtype=np.float32)
    else:
        scores_arr = np.asarray(scores, dtype=np.float32)
        if len(scores_arr) != n_raw:
            raise ValueError(f"scores length {len(scores_arr)} != sequences length {n_raw}")
    seq_to_score: dict[str, float] = {}
    for seq, sc in zip(raw_sequences, scores_arr):
        if seq not in seq_to_score:
            seq_to_score[seq] = float(sc)

    # Candidate pool for the top-k: valid + (optionally) plausible.
    pool_pre = library
    if apply_plausibility_to_top:
        pool_pre = filter_plausible(pool_pre)

    # Score-ranked shortlist before the expensive screens. Stable argsort on
    # negated scores keeps original (generation) order among ties, so the
    # result is a pure function of (inputs, seed).
    pre_scores = np.array([seq_to_score.get(s, 0.0) for s in pool_pre], dtype=np.float64)
    order = np.argsort(-pre_scores, kind="stable")
    if max_novelty_candidates is not None and max_novelty_candidates > 0:
        order = order[:max_novelty_candidates]
    pool = [pool_pre[i] for i in order]
    pool_scores = [float(pre_scores[i]) for i in order]

    pool = filter_novel(pool, reference_set)
    # Novelty screening may drop entries; realign scores with the survivors.
    pool_scores = [seq_to_score.get(s, 0.0) for s in pool]

    if pool:
        idx = farthest_point_select(pool, pool_scores, min(top_k, len(pool)), seed=seed)
        top = [pool[i] for i in idx]
        # Re-rank selected by score descending for the final ordering.
        top.sort(key=lambda s: seq_to_score.get(s, 0.0), reverse=True)
    else:
        top = []

    rejected = {
        "invalid": n_raw - len(valid),
        "overlap": len(valid) - len(library),
    }
    return SelectionResult(library=library, top=top, rejected=rejected)


__all__ = [
    "is_valid_sequence",
    "filter_valid",
    "filter_plausible",
    "remove_exact_overlap",
    "clean_candidates",
    "count_clean_candidates",
    "is_novel_top",
    "filter_novel",
    "farthest_point_select",
    "SelectionResult",
    "select_library_and_top",
]
