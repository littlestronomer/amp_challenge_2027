"""Reproducibility tests for the generation entry point.

Verifies the core determinism contract that the validator checks: two calls
to the fallback generator with the same seed produce identical output. The
model-backed path is covered separately once weights exist.
"""

from __future__ import annotations

from amp_challenge_2027.generate import generate_fallback


def test_fallback_generator_is_deterministic():
    a = generate_fallback(1000, seed=42, length=50)
    b = generate_fallback(1000, seed=42, length=50)
    assert a == b


def test_fallback_generator_respects_length_bounds():
    seqs = generate_fallback(500, seed=7, length=50)
    assert all(8 <= len(s) <= 50 for s in seqs)


def test_fallback_generator_alphabet():
    seqs = generate_fallback(500, seed=7, length=50)
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    for s in seqs:
        assert set(s) <= valid


def test_different_seeds_differ():
    a = generate_fallback(500, seed=1, length=50)
    b = generate_fallback(500, seed=2, length=50)
    assert a != b
