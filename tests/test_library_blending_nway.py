"""N-way fixed-weights blending: quotas, interleave order, shared handling.

``fixed_weights_blend`` is the N-component generalization of the locked-hybrid
two-pool blend. These tests pin its exact semantics: Hamilton quotas,
priority-ordered selection with shared-candidate reservation, deterministic
weight-block interleave, provenance accounting and failure behavior.
"""

from __future__ import annotations

import pytest

from amp_challenge_2027.library_blending import (
    apportioned_quotas,
    fixed_weights_blend,
    parse_ratio,
    parse_weights,
    source_quotas,
)

A = [f"A{i}" for i in range(40)]
B = [f"B{i}" for i in range(20)]
C = [f"C{i}" for i in range(20)]


# ---------------------------------------------------------------------------
# Weight parsing + quota apportionment
# ---------------------------------------------------------------------------


def test_parse_weights_reduces_and_validates():
    assert parse_weights("10:4:2") == (5, 2, 1)
    assert parse_weights("1:1") == (1, 1)
    assert parse_weights("9:3:1") == (9, 3, 1)
    for bad in ("5", "0:1", "1:0", "a:b", "1::2", "", "1:1:1:0"):
        with pytest.raises(ValueError):
            parse_weights(bad)


def test_apportioned_quotas_exact_and_summing():
    assert apportioned_quotas(50_000, (3, 1)) == (37_500, 12_500)
    # floors 34615/11538/3846 (sum 49999); remainder numerators 5/6/2 → the
    # single unit goes to the largest fractional part (component 1).
    assert apportioned_quotas(50_000, (9, 3, 1)) == (34_615, 11_539, 3_846)
    quotas = apportioned_quotas(50, (5, 2, 1))
    assert sum(quotas) == 50
    # floors 31/12/6 (sum 49); remainders 2/4/2 → the single unit goes to the
    # largest fractional part (component 1) → (31, 13, 6)
    assert quotas == (31, 13, 6)


def test_apportioned_quotas_tie_broken_by_component_order():
    assert apportioned_quotas(10, (1, 1, 1)) == (4, 3, 3)  # all remainders equal → earliest wins


def test_apportioned_quotas_matches_two_pool_on_divisible_ratios():
    for size in (50_000, 40_000):
        for ratio in ((3, 1), (7, 1), (1, 1), (87, 13)):
            assert apportioned_quotas(size, ratio) == source_quotas(size, ratio)


# ---------------------------------------------------------------------------
# fixed_weights_blend semantics
# ---------------------------------------------------------------------------


def test_disjoint_pools_block_interleave_order():
    blend = fixed_weights_blend([A, B, C], (5, 2, 1), size=20, labels=["a", "b", "c"])
    assert blend.quotas == (13, 5, 2)
    expected = (
        A[0:5] + B[0:2] + C[0:1]
        + A[5:10] + B[2:4] + C[1:2]
        + A[10:13] + B[4:5]  # partial final cycle: quotas exhausted per component
    )
    assert blend.sequences == expected
    assert blend.sources.count("a") == 13
    assert blend.sources.count("b") == 5
    assert blend.sources.count("c") == 2
    assert blend.pool_shared == (0, 0, 0)
    assert blend.selected_shared == (0, 0, 0)


def test_default_labels_and_membership_provenance():
    blend = fixed_weights_blend([A, B, C], (1, 1, 1), size=9)
    assert set(blend.labels) == {"component_0", "component_1", "component_2"}
    assert sorted(zip(blend.sequences, blend.sources)) == sorted(
        zip(blend.sequences, [lbl for seq, lbl in zip(blend.sequences, blend.sources)])
    )
    assert len(set(blend.sequences)) == 9


def test_shared_sequences_assigned_once_with_priority():
    # A's plain prefix (SH1, SH2, A0, A1, A2) would claim the two shared
    # candidates C needs; the reservation rule defers them so C meets quota.
    a = ["SH1", "SH2"] + [f"A{i}" for i in range(12)]
    b = [f"B{i}" for i in range(10)]
    c = ["SH1", "SH2", "SH3"]  # quota 3 = its entire pool
    blend = fixed_weights_blend([a, b, c], (5, 2, 3), size=10, labels=["a", "b", "c"])
    assert blend.quotas == (5, 2, 3)
    assert len(set(blend.sequences)) == 10
    selected = {label: [seq for seq, src in zip(blend.sequences, blend.sources) if src == label] for label in "abc"}
    assert selected["a"] == [f"A{i}" for i in range(5)]  # shared candidates deferred
    assert selected["c"] == ["SH1", "SH2", "SH3"]
    # C's selection contains 2 shared members (SH3 is C-exclusive)
    assert blend.selected_shared == (0, 0, 2)
    assert blend.pool_shared == (2, 0, 2)


def test_shared_between_earlier_components_stay_with_earlier():
    a = ["X0", "X1", "SH"] + [f"A{i}" for i in range(6)]
    b = ["SH", "B0", "B1", "B2", "B3"]
    blend = fixed_weights_blend([a, b], (2, 1), size=6)
    assert len(set(blend.sequences)) == 6
    assert "SH" in blend.sequences
    # No starvation: both quotas met exactly.
    assert blend.sources.count("component_0") == 4
    assert blend.sources.count("component_1") == 2


def test_deterministic_repeat_calls():
    first = fixed_weights_blend([A, B, C], (5, 2, 1), size=21)
    second = fixed_weights_blend([A, B, C], (5, 2, 1), size=21)
    assert first.sequences == second.sequences
    assert first.sources == second.sources


def test_starvation_and_infeasibility_fail_closed():
    with pytest.raises(ValueError, match="Insufficient candidates"):
        fixed_weights_blend([A[:3], B, C], (5, 2, 1), size=20)
    with pytest.raises(ValueError, match="overlap too heavily|Unable to satisfy"):
        tiny = ["S1", "S2", "S3"]
        fixed_weights_blend([tiny, tiny[:], [f"X{i}" for i in range(20)]], (1, 1, 1), size=9)
    with pytest.raises(ValueError, match="must be unique"):
        fixed_weights_blend([A + A[:1], B, C], (1, 1, 1), size=9)


def test_weight_count_and_label_validation():
    with pytest.raises(ValueError):
        fixed_weights_blend([A, B, C], (1, 1), size=9)
    with pytest.raises(ValueError):
        fixed_weights_blend([A, B], (1, 1), size=9, labels=["same", "same"])


def test_two_way_matches_legacy_quotas_on_divisible_ratio():
    ratio = parse_ratio("3:1")
    blend = fixed_weights_blend([A, B], ratio, size=40)
    assert blend.quotas == source_quotas(40, ratio)
    assert blend.sequences[:8] == A[0:3] + B[0:1] + A[3:6] + B[1:2]
