"""Hybrid-library blending: interleave semantics + entry-point blend wiring.

The locked 75/25 hybrid (PHASE1_RESULTS.md) is produced by
``interleave_blend`` — these tests pin its exact semantics (ratio blocks,
global dedup, exhaustion, prefix property that makes blended output equal
blending two standalone libraries) and the entry-point wiring (auto-detect,
missing-checkpoint fallback, default byte-compat).
"""

from __future__ import annotations

from pathlib import Path

from amp_challenge_2027 import generate as gen
from amp_challenge_2027.pipeline import interleave_blend

A = [f"A{i}" for i in range(100)]
B = [f"B{i}" for i in range(100)]


# ---------------------------------------------------------------------------
# interleave_blend semantics
# ---------------------------------------------------------------------------


def test_ratio_blocks_and_order():
    out = interleave_blend(A, B, per_primary=3, per_secondary=1, n=8)
    assert out == ["A0", "A1", "A2", "B0", "A3", "A4", "A5", "B1"]


def test_global_dedup_first_occurrence_wins():
    primary = ["A0", "A1", "A2", "SHARED", "A3"]
    secondary = ["B0", "SHARED", "B1"]
    out = interleave_blend(primary, secondary, per_primary=1, per_secondary=1, n=6)
    assert out == ["A0", "B0", "A1", "SHARED", "A2", "B1"]  # SHARED once, from primary


def test_secondary_exhaustion_primary_continues():
    out = interleave_blend(A, ["B0"], per_primary=1, per_secondary=1, n=5)
    assert out == ["A0", "B0", "A1", "A2", "A3"]


def test_primary_exhaustion_secondary_continues():
    out = interleave_blend(["A0"], B, per_primary=1, per_secondary=1, n=4)
    assert out == ["A0", "B0", "B1", "B2"]


def test_deterministic_and_capped():
    r1 = interleave_blend(A, B, per_primary=3, per_secondary=1, n=50)
    r2 = interleave_blend(A, B, per_primary=3, per_secondary=1, n=50)
    assert r1 == r2 and len(r1) == 50


def test_prefix_property_matches_two_library_blend():
    """The locked-hybrid construction: interleave two 50k-style libraries.

    interleave(clean(rawA)[:k], clean(rawB)[:m]) == prefix of the full blend
    when only prefixes are consumed — this is what makes the entry-point
    (candidate-stage blending) byte-match the probe (library-stage blending).
    """
    full = interleave_blend(A, B, per_primary=3, per_secondary=1, n=100)
    truncated = interleave_blend(A[:20], B[:20], per_primary=3, per_secondary=1, n=8)
    assert truncated == full[:8]


def test_invalid_ratio_rejected():
    import pytest

    with pytest.raises(ValueError):
        interleave_blend(A, B, per_primary=0, per_secondary=1, n=5)


# ---------------------------------------------------------------------------
# Entry-point wiring (hermetic: monkeypatched sampler)
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, checkpoint, blend, ratio=3, seed=42):
        self.checkpoint = checkpoint
        self.blend_checkpoint = blend
        self.blend_ratio = ratio
        self.seed = seed
        self.length = 50
        self.device = "cpu"
        self.temperature = 1.0
        self.sample_top_k = 50
        self.top_p = 0.9
        self.repetition_penalty = 1.3
        self.charge_conditioned = False


def test_blend_path_samples_both_and_interleaves(tmp_path, monkeypatch):
    calls = []

    AA = "ACDEFGHIKLMNPQRSTVWY"

    def fake_sample(n, **kwargs):
        checkpoint_dir = Path(kwargs["checkpoint_dir"])
        calls.append((checkpoint_dir.name, n))
        # Valid peptides: primary K-flavored, secondary R-flavored, unique per i.
        if checkpoint_dir.name == "generator":
            return [f"KKL{AA[i % 20]}{AA[(i // 20) % 20]}KLLKLL" for i in range(n)]
        return [f"RRL{AA[i % 20]}{AA[(i // 20) % 20]}KLLKLL" for i in range(n)]

    monkeypatch.setattr(gen, "generate_with_model", fake_sample)
    primary = tmp_path / "generator"
    primary.mkdir()
    secondary = tmp_path / "generator_blend"
    secondary.mkdir()
    (primary / "config.json").write_text("{}")
    (secondary / "config.json").write_text("{}")

    out = gen._sample_candidates(
        _Args(primary, secondary, ratio=3), use_model=True, reference_set=set(), target=40
    )
    names = {c[0] for c in calls}
    assert names == {"generator", "generator_blend"}
    # Both streams at standalone sizing (target*2) — the conditioned bin draw
    # is n-dependent, so byte-reproduction requires matching sample counts.
    assert all(n == 80 for _name, n in calls)
    # 3:1 blocks from the two streams (K-flavored primary, R-flavored secondary)
    assert [seq[:3] for seq in out[:4]] == ["KKL", "KKL", "KKL", "RRL"]
    assert len(out) == 40


def test_blend_missing_config_disables_blending(tmp_path, monkeypatch):
    """Secondary without config.json → legacy single-stream path (bytes)."""
    primary = tmp_path / "generator"
    primary.mkdir()
    (primary / "config.json").write_text("{}")
    missing = tmp_path / "generator_blend"  # not created

    def fake_sample(*a, **k):
        raise AssertionError("should not blend")

    monkeypatch.setattr(gen, "_sample_from", fake_sample)

    AA = "ACDEFGHIKLMNPQRSTVWY"

    def fake_fallback(n, *, seed, length):
        return [f"KLLKLLKKLLK{AA[i % 20]}" for i in range(n)]

    monkeypatch.setattr(gen, "generate_fallback", fake_fallback)
    # use_model=False exercises the single path without torch
    out = gen._sample_candidates(
        _Args(primary, missing), use_model=False, reference_set=set(), target=10
    )
    assert len(out) == 20  # legacy round-0 sizing: target*2


def test_no_blend_flag_single_stream(tmp_path, monkeypatch):
    primary = tmp_path / "generator"
    primary.mkdir()
    (primary / "config.json").write_text("{}")

    def fake_sample(*a, **k):
        raise AssertionError("should not blend")

    monkeypatch.setattr(gen, "_sample_from", fake_sample)

    AA = "ACDEFGHIKLMNPQRSTVWY"

    def fake_fallback(n, *, seed, length):
        return [f"KLLKLLKKLLK{AA[i % 20]}" for i in range(n)]

    monkeypatch.setattr(gen, "generate_fallback", fake_fallback)
    out = gen._sample_candidates(
        _Args(primary, None), use_model=False, reference_set=set(), target=10
    )
    assert len(out) == 20
