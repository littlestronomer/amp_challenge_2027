"""Tests for the property-distribution shaper (``amp_challenge_2027.shape``).

Exercises the three guarantees the shaper must uphold:
  1. Determinism — identical inputs/seed → byte-identical output.
  2. Effectiveness — shaped charge spread > naive first-N spread (closer to ref).
  3. Length contract — output is exactly ``library_size`` (when pool is large).
  4. Distribution match — shaped charge histogram is closer to the reference
     histogram than the naive first-N histogram (lower chi-square distance).

Uses synthetic UNIQUE peptides with controlled net charges so the assertions are
exact and no model/GPU is required. (The real generation path feeds the shaper
deduplicated candidates, so the fixtures respect that precondition.)
"""

from __future__ import annotations

import numpy as np
import pytest

from amp_challenge_2027.props import compute_properties
from amp_challenge_2027.shape import (
    property_stats,
    shape_library_to_reference,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures: unique peptides with an exact target net charge.
# K=+1, E=-1, neutrals (G/A/S/T/P/Q/N)=0, termini cancel → charge == (#K - #E).
# ---------------------------------------------------------------------------


def _make_unique(charge: int, n: int, *, seed: int, length: int = 12) -> list[str]:
    """Generate ``n`` unique peptides each with net charge == ``charge``."""
    rng = np.random.default_rng(seed)
    neutrals = list("GASTPQN")
    n_pos = max(charge, 0)
    n_neg = max(-charge, 0)
    length = max(length, n_pos + n_neg + 4)
    rest = length - n_pos - n_neg
    seqs: set[str] = set()
    tries = 0
    while len(seqs) < n and tries < n * 80:
        tries += 1
        chars = ["K"] * n_pos + ["E"] * n_neg + [str(rng.choice(neutrals)) for _ in range(rest)]
        rng.shuffle(chars)
        seqs.add("".join(chars))
    return sorted(seqs)


def _wide_reference() -> list[str]:
    """Reference with a deliberately wide, even charge spread (anionic→cationic)."""
    parts = [
        _make_unique(-8, 50, seed=101),
        _make_unique(-4, 60, seed=102),
        _make_unique(0, 80, seed=103),
        _make_unique(2, 80, seed=104),
        _make_unique(4, 60, seed=105),
        _make_unique(8, 40, seed=106),
    ]
    return [s for part in parts for s in part]


def _narrow_candidates() -> list[str]:
    """Candidate pool concentrated at the mode, with RARE tails (generator-like)."""
    parts = [
        _make_unique(0, 300, seed=1),
        _make_unique(2, 250, seed=2),
        _make_unique(4, 150, seed=3),
        _make_unique(-4, 25, seed=4),
        _make_unique(-8, 20, seed=5),
        _make_unique(8, 20, seed=6),
    ]
    pool = [s for part in parts for s in part]
    # Simulate generator output order (not alphabetical) with a fixed permutation
    # so "naive first-N" is representative of the pool's narrow distribution.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pool))
    return [pool[i] for i in order]


def _charges(seqs: list[str]) -> np.ndarray:
    return np.array([compute_properties(s).charge for s in seqs], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pool_is_deduped_precondition():
    """Sanity: the fixtures feed the shaper unique sequences (as the real path does)."""
    pool = _narrow_candidates()
    assert len(set(pool)) == len(pool)


def test_length_contract():
    pool = _narrow_candidates()
    ref = _wide_reference()
    out = shape_library_to_reference(pool, ref, library_size=100, seed=0)
    assert len(out) == 100
    assert len(set(out)) == 100  # unique in → unique out


def test_length_when_pool_smaller_than_target():
    """If the pool can't fill the target, return everything available."""
    pool = _make_unique(2, 3, seed=7)
    out = shape_library_to_reference(pool, _wide_reference(), library_size=100, seed=0)
    assert len(out) == 3
    assert set(out) == set(pool)


def test_empty_inputs():
    assert shape_library_to_reference([], _wide_reference(), library_size=50, seed=0) == []
    assert (
        shape_library_to_reference(_narrow_candidates(), [], library_size=50, seed=0)
        == _narrow_candidates()[:50]
    )


def test_determinism():
    """Identical inputs → identical output, regardless of seed (order-deterministic)."""
    pool = _narrow_candidates()
    ref = _wide_reference()
    out_a = shape_library_to_reference(pool, ref, library_size=100, seed=0)
    out_b = shape_library_to_reference(pool, ref, library_size=100, seed=0)
    out_c = shape_library_to_reference(pool, ref, library_size=100, seed=999)
    assert out_a == out_b
    assert out_a == out_c  # seed currently reserved/unused → same output


def test_shaped_charge_spread_exceeds_naive():
    """The whole point: shaping widens the charge distribution vs first-N."""
    pool = _narrow_candidates()
    ref = _wide_reference()
    lib_size = 120

    naive = pool[:lib_size]
    shaped = shape_library_to_reference(pool, ref, library_size=lib_size, seed=0)

    naive_std = _charges(naive).std()
    shaped_std = _charges(shaped).std()
    ref_std = _charges(ref).std()

    assert shaped_std > naive_std, (
        f"shaping should widen charge spread: shaped {shaped_std:.2f} <= naive {naive_std:.2f}"
    )
    assert abs(shaped_std - ref_std) < abs(naive_std - ref_std)


def _chi_square_to_reference(charges: np.ndarray, ref_charges: np.ndarray) -> float:
    lo, hi = int(np.floor(ref_charges.min())), int(np.ceil(ref_charges.max()))
    edges = np.arange(lo, hi + 2) - 0.5
    ref_h, _ = np.histogram(ref_charges, bins=edges, density=True)
    cand_h, _ = np.histogram(charges, bins=edges, density=True)
    return float(np.sum((cand_h - ref_h) ** 2))


def test_shaped_histogram_closer_to_reference():
    """Shaped charge histogram has lower chi-square distance to the reference."""
    pool = _narrow_candidates()
    ref = _wide_reference()
    lib_size = 150
    ref_c = _charges(ref)

    naive_dist = _chi_square_to_reference(_charges(pool[:lib_size]), ref_c)
    shaped_dist = _chi_square_to_reference(
        _charges(shape_library_to_reference(pool, ref, library_size=lib_size, seed=0)), ref_c
    )
    assert shaped_dist < naive_dist


def test_property_stats_empty():
    assert property_stats([]).n == 0


def test_property_stats_nonempty():
    s = property_stats(["KKKK", "EEEE"])  # +4, -4
    assert s.n == 2
    assert s.charge_mean == pytest.approx(0.0, abs=1e-6)
