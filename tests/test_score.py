"""Tests for the multi-objective scorer (score.py) and pipeline wiring.

Only the numpy-only conformity path is exercised end-to-end here; the
torch-backed activity/precision components degrade gracefully to "dropped" in
the minimal test environment, which the pipeline tests assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from amp_challenge_2027.config import AMINO_ACIDS
from amp_challenge_2027.pipeline import build_composite_scorer, clean_candidates
from amp_challenge_2027.score import (
    CompositeScorer,
    ConformityScorer,
    _log_densities,
    _property_matrix,
    _silverman_bandwidths,
)


def _random_seqs(rng: np.random.Generator, n: int, lo: int = 8, hi: int = 13) -> list[str]:
    return [
        "".join(rng.choice(list(AMINO_ACIDS), size=int(rng.integers(lo, hi)))) for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Property space + KDE machinery
# ---------------------------------------------------------------------------


def test_property_matrix_sane():
    seqs = ["KKKKKKKKKK", "EEEEEEEEEE", "ACDEFGHIKL"]
    m = _property_matrix(seqs)
    assert m.shape == (3, 3)
    # Poly-K is strongly cationic, poly-E anionic; charge ordering must hold.
    assert m[0, 0] > m[2, 0] > m[1, 0]
    # Hydrophobicity column finite everywhere.
    assert np.isfinite(m).all()


def test_log_densities_rank_matches_scipy_gaussian_kde():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 1))
    queries = rng.normal(size=(50, 1))
    h = _silverman_bandwidths(data)

    mine = _log_densities(queries, data, h)
    kde = scipy.gaussian_kde(data.ravel(), bw_method=float(h[0]) / data.std())
    ref = np.log(kde(queries.ravel()))

    rho = np.corrcoef(np.argsort(np.argsort(mine)), np.argsort(np.argsort(ref)))[0, 1]
    assert rho > 0.99


def test_conformity_scores_inlier_above_outlier():
    rng = np.random.default_rng(42)
    reference = _random_seqs(rng, 400)
    scorer = ConformityScorer(reference, sample=0)
    inliers = _random_seqs(rng, 30)
    outlier = ["K" * 12]  # extreme cationic tail — far outside a uniform ref
    scores = scorer.score(inliers + outlier)
    assert scores.shape == (31,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # The extreme outlier must sit below every inlier's density quantile.
    assert (scores[:-1] > scores[-1]).mean() > 0.9


def test_conformity_subsample_is_deterministic_and_bounded():
    rng = np.random.default_rng(7)
    reference = _random_seqs(rng, 500)
    s_full = ConformityScorer(reference, sample=0)
    s_a = ConformityScorer(reference, sample=100, seed=1)
    s_b = ConformityScorer(reference, sample=100, seed=1)
    queries = _random_seqs(rng, 10)
    a1, a2 = s_a.score(queries), s_b.score(queries)
    assert np.allclose(a1, a2)
    full = s_full.score(queries)
    # Same distribution family → moderate rank agreement between subsample and full.
    assert np.corrcoef(a1, full)[0, 1] > 0.5


# ---------------------------------------------------------------------------
# Composite combiner
# ---------------------------------------------------------------------------


def test_composite_zscore_weighting():
    vals_a = np.array([0.0, 1.0, 2.0, 3.0])
    vals_b = np.array([3.0, 2.0, 1.0, 0.0])

    def make(vals):
        return lambda seqs: vals

    combo = CompositeScorer([("a", 1.0, make(vals_a)), ("b", 1.0, make(vals_b))])
    combined, parts = combo.score(["x"] * 4)
    assert set(parts) == {"a", "b"}
    # Symmetric opposite components cancel exactly.
    assert np.allclose(combined, 0.0, atol=1e-6)

    tilted = CompositeScorer([("a", 3.0, make(vals_a)), ("b", 1.0, make(vals_b))])
    combined_t, _ = tilted.score(["x"] * 4)
    assert combined_t[-1] > combined_t[0]


def test_composite_zero_variance_component_ignored():
    flat = CompositeScorer(
        [
            ("a", 1.0, lambda seqs: np.ones(len(seqs))),
            ("b", 1.0, lambda seqs: np.arange(len(seqs), dtype=float)),
        ]
    )
    combined, parts = flat.score(["x", "y", "z"])
    assert "a" in parts
    assert np.isfinite(combined).all()
    assert combined[-1] > combined[0]


# ---------------------------------------------------------------------------
# Pipeline wiring (minimal-env degradation)
# ---------------------------------------------------------------------------


def test_build_composite_scorer_conformity_only_without_torch_artifacts():
    rng = np.random.default_rng(3)
    reference = _random_seqs(rng, 120)
    scorer = build_composite_scorer(reference, w_activity=0.0, w_conformity=0.5, w_precision=0.0)
    assert scorer is not None
    # In this environment the classifier artifact doesn't exist; whatever
    # happens with torch availability, at least conformity must survive.
    assert "conformity" in scorer.names
    combined, parts = scorer.score(_random_seqs(rng, 8))
    assert combined.shape == (8,)
    assert "conformity" in parts


def test_clean_candidates_filters_and_preserves_order():
    raw = ["KLLAKLLAKL", "BAD!", "KLLAKLLAKL", "A" * 4, "GHIKLMNPQRST"]
    cleaned = clean_candidates(raw, reference_set={"GHIKLMNPQRST"})
    assert cleaned == ["KLLAKLLAKL"]
