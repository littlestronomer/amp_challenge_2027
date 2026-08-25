"""Tests for the selection pipeline.

These mirror the rules enforced by ``scripts/verify_submission.py``: alphabet
validity, length bounds, uniqueness, no exact overlap, and top-100 novelty.
They do NOT depend on torch/transformers, so they run in the minimal dev env.
"""

from __future__ import annotations

import numpy as np

from amp_challenge_2027.select import (
    count_clean_candidates,
    filter_valid,
    is_novel_top,
    is_valid_sequence,
    remove_exact_overlap,
    select_library_and_top,
)


def test_valid_alphabet():
    assert is_valid_sequence("ACDEFGHIKLMNPQRSTVWY")  # all 20 AAs, len 20
    assert is_valid_sequence("KLLKLLKLLK")  # len 10


def test_invalid_alphabet():
    assert not is_valid_sequence("BXJ")  # non-standard
    assert not is_valid_sequence("AGG-C")  # has a dash
    assert not is_valid_sequence("ACD2E")  # digit


def test_length_bounds():
    assert not is_valid_sequence("KLLL")  # len 4 < 8
    assert not is_valid_sequence("A" * 7)
    assert is_valid_sequence("A" * 8)
    assert is_valid_sequence("A" * 50)
    assert not is_valid_sequence("A" * 51)  # > 50


def test_filter_valid_dedup():
    seqs = ["ACDEFGHIK", "ACDEFGHIK", "GHIKLMNPQRS", "BADSEQ", "TOOSHORT"]
    out = filter_valid(seqs, drop_duplicates=True)
    assert out == ["ACDEFGHIK", "GHIKLMNPQRS"]


def test_remove_exact_overlap():
    seqs = ["ACDEFGHIK", "GHIKLMNPQRS"]
    ref = {"ACDEFGHIK"}
    assert remove_exact_overlap(seqs, ref) == ["GHIKLMNPQRS"]


def test_novel_top_rejects_near_identical():
    # A sequence almost identical to the reference should be rejected (>0.8 ratio).
    reference = ["KLLKLLKLLKLLKLLKLLK"]  # len 18
    near = "KLLKLLKLLKLLKLLKLLR"  # one substitution
    assert not is_novel_top(near, reference, threshold=0.8)


def test_novel_top_accepts_distant():
    reference = ["VVVVVVVVVV"]  # unrelated
    seq = "KKKKKKKK"  # also unrelated
    assert is_novel_top(seq, reference, threshold=0.8)


def test_select_library_and_top_shape():
    raw = [
        "ACDEFGHIK", "GHIKLMNPQRS", "KLLKLLKLLK", "QRSTVWYACD",
        "ACDEFGHIK",  # duplicate
        "BADSEQ00",    # invalid
    ]
    reference = {"KLLKLLKLLK"}  # one overlap target
    result = select_library_and_top(raw, reference_set=reference, top_k=2, library_size=100, seed=1)
    # Library: valid + dedup + no-overlap → 3 entries
    assert "BADSEQ00" not in result.library
    assert "ACDEFGHIK" in result.library  # appears once
    assert "KLLKLLKLLK" not in result.library  # overlap removed
    assert len(result.top) <= 2
    # Every top entry must be in the library.
    for s in result.top:
        assert s in result.library


def test_select_determinism_same_seed():
    raw = [f"{'ACDEFGHIK'[i % 9]}{chr(65 + i)}{chr(66 + i)}{chr(67 + i)}{chr(68 + i)}"
           f"{chr(69 + i)}{chr(70 + i)}{chr(71 + i)}" for i in range(20)]
    reference: set[str] = set()
    r1 = select_library_and_top(raw, reference_set=reference, top_k=5, seed=42)
    r2 = select_library_and_top(raw, reference_set=reference, top_k=5, seed=42)
    assert r1.top == r2.top
    assert r1.library == r2.library


# ---------------------------------------------------------------------------
# Scalability contract: score-ranked shortlist before novelty/FPS
# ---------------------------------------------------------------------------


def _plausible_pool(n: int) -> list[str]:
    """n distinct seeded sequences that all pass props.is_plausible."""
    from amp_challenge_2027.props import is_plausible

    rng = np.random.default_rng(11)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    # AMP-like composition: cationic + hydrophobic enriched, no extreme tails.
    weights = np.array([
        2, 1, 1, 1, 1, 2, 1, 2, 4, 4, 1, 1, 1, 1, 3, 2, 2, 3, 0.5, 0.5,
    ])
    weights /= weights.sum()
    seqs, seen = [], set()
    while len(seqs) < n:
        seq = "".join(rng.choice(alphabet, size=int(rng.integers(10, 17)), p=weights))
        if seq in seen or not is_plausible(seq):
            continue
        seen.add(seq)
        seqs.append(seq)
    return seqs


def test_count_clean_candidates():
    raw = ["KLLAKLLAKL", "BADSEQ!", "KLLAKLLAKL", "A" * 4, "GHIKLMNPQRST"]
    ref = {"GHIKLMNPQRST"}
    assert count_clean_candidates(raw, ref) == 1  # dedup keeps one KLLAKLLAKL;
    # GHIKLMNPQRST overlaps the reference and is dropped.
    assert count_clean_candidates(["KLLAKLLAKL"], set()) == 1


def test_shortlist_equivalent_when_budget_exceeds_pool():
    raw = _plausible_pool(30)
    scores = np.linspace(0, 1, len(raw))
    big = select_library_and_top(
        raw, reference_set=set(), scores=scores, top_k=10,
        library_size=100, seed=0, max_novelty_candidates=1000,
    )
    exact = select_library_and_top(
        raw, reference_set=set(), scores=scores, top_k=10,
        library_size=100, seed=0, max_novelty_candidates=0,  # unlimited
    )
    assert big.library == exact.library
    assert big.top == exact.top


def test_shortlist_restricts_top_to_high_scores():
    raw = _plausible_pool(40)
    scores = np.linspace(0.0, 1.0, len(raw))  # later indices score higher
    result = select_library_and_top(
        raw, reference_set=set(), scores=scores, top_k=5,
        library_size=200, seed=0, max_novelty_candidates=6,
    )
    shortlisted = {raw[i] for i in range(len(raw) - 6, len(raw))}
    assert result.top, "expected a non-empty top list"
    for s in result.top:
        assert s in shortlisted
