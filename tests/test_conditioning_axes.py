"""Multi-axis conditioning: descriptors, bins, axis parsing, joint sampling.

The hydrophobicity / hydrophobic-moment implementations replicate the exact
seqme→modlamp call pattern (Eisenberg scale; ``calculate_global``;
``calculate_moment(window=11, angle=100, modality="mean")``). Golden anchors
below are pinned from this implementation; the SSH parity run
(``scripts/parity_modlamp_descriptors.py``) is the authoritative comparison
against modlamp and gates any conditioned training.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from amp_challenge_2027.conditioning import (
    CONDITIONING_AXES,
    EISENBERG_SCALE,
    HMOMENT_MIN,
    HMOMENT_STEP,
    HYDRO_MIN,
    HYDRO_STEP,
    NUM_CHARGE_BINS,
    NUM_HMOMENT_BINS,
    NUM_HYDRO_BINS,
    axis_bin,
    axis_default_bin,
    axis_num_bins,
    charge_modlamp,
    condition_bins,
    hmoment_bin,
    hydrophobic_moment_modlamp,
    hydrophobicity_bin,
    hydrophobicity_modlamp,
    joint_bin_counts,
    parse_conditioning_axes,
    reference_joint_proportions,
    sample_condition_bins,
    total_variation_distance,
)

# (sequence, charge, hydrophobicity, hydrophobic_moment) — pinned anchors.
ANCHORS = [
    ("GLFDIVKKVVGALGSL", 0.996, 0.44875, 0.5350441929603722),
    ("GIGKFLHSAKKFGKAFVGEIMNS", 3.095, 0.17913043478260865, 0.5322227991233682),
    ("KKKKKKKK", 7.994, -1.5, 0.15733118084574002),
    ("FLPAIWAAAKFL", 0.996, 0.6591666666666667, 0.3412809572600638),
    ("WLRRIRKIAAHR", 5.094, -0.4958333333333333, 0.6096420262203484),
]


# ---------------------------------------------------------------------------
# Descriptors (seqme/modlamp scale)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq,charge,hydro,moment", ANCHORS)
def test_descriptor_anchors(seq, charge, hydro, moment):
    assert charge_modlamp(seq) == pytest.approx(charge, abs=1e-9)
    assert hydrophobicity_modlamp(seq) == pytest.approx(hydro, abs=1e-9)
    assert hydrophobic_moment_modlamp(seq) == pytest.approx(moment, abs=1e-9)


def test_hydrophobicity_is_mean_eisenberg():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    expected = sum(EISENBERG_SCALE[a] for a in seq) / len(seq)
    assert hydrophobicity_modlamp(seq) == pytest.approx(expected, abs=1e-12)
    assert hydrophobicity_modlamp("KKKKKKKK") == -1.5


def test_hmoment_matches_independent_trig_computation():
    """Window-11 sliding mean, independently recomputed with cmath."""
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    values = [EISENBERG_SCALE[a] for a in seq]
    wdw = 11
    theta = math.radians(100.0)
    windows = []
    for start in range(len(values) - wdw + 1):
        acc = sum(values[start + k] * cmath.exp(1j * k * theta) for k in range(wdw))
        windows.append(abs(acc) / wdw)
    assert hydrophobic_moment_modlamp(seq) == pytest.approx(sum(windows) / len(windows), abs=1e-12)


def test_hmoment_short_sequence_clamps_window():
    """len < window → single whole-sequence window (modlamp semantics)."""
    seq = "KKKKKKKK"
    theta = math.radians(100.0)
    acc = sum(EISENBERG_SCALE[a] * cmath.exp(1j * k * theta) for k, a in enumerate(seq))
    assert hydrophobic_moment_modlamp(seq, window=11) == pytest.approx(abs(acc) / len(seq), abs=1e-12)


def test_invalid_residue_fails_closed():
    for fn in (hydrophobicity_modlamp, hydrophobic_moment_modlamp):
        with pytest.raises(ValueError):
            fn("KLLZKLL")
        with pytest.raises(ValueError):
            fn("")


# ---------------------------------------------------------------------------
# Bins + axis metadata
# ---------------------------------------------------------------------------


def test_bins_round_and_clamp():
    # hydro(KKKKKKKK) = -1.5 exactly at the lower edge → bin 0
    assert hydrophobicity_bin("KKKKKKKK") == 0
    # custom tiny grid forces both clamps
    assert hydrophobicity_bin("KKKKKKKK", hydro_min=-1.0, step=0.1, num_bins=3) == 0
    assert hydrophobicity_bin("IIIIIIII", hydro_min=-1.0, step=0.1, num_bins=3) == 2
    assert hmoment_bin("GLFDIVKKVVGALGSL", hmoment_min=0.5, step=0.1, num_bins=2) == 0
    assert hmoment_bin("GLFDIVKKVVGALGSL", hmoment_min=0.0, step=0.01, num_bins=2) == 1


def test_axis_bin_dispatch_matches_direct_helpers():
    seq = "GLFDIVKKVVGALGSL"
    assert axis_bin(seq, "hydro") == hydrophobicity_bin(seq)
    assert axis_bin(seq, "hmoment") == hmoment_bin(seq)
    assert condition_bins(seq, ("hydro",)) == {"hydro": hydrophobicity_bin(seq)}
    with pytest.raises(ValueError):
        axis_bin(seq, "gravvy")


def test_axis_metadata():
    assert axis_num_bins("charge") == NUM_CHARGE_BINS
    assert axis_num_bins("hydro") == NUM_HYDRO_BINS
    assert axis_num_bins("hmoment") == NUM_HMOMENT_BINS
    # defaults live inside the bin ranges on the default edges
    for axis in CONDITIONING_AXES:
        assert 0 <= axis_default_bin(axis) < axis_num_bins(axis)
    assert axis_default_bin("hydro") == int(round(((-0.067) - HYDRO_MIN) / HYDRO_STEP))
    assert axis_default_bin("hmoment") == int(round((0.395 - HMOMENT_MIN) / HMOMENT_STEP))


def test_parse_conditioning_axes():
    assert parse_conditioning_axes("none") == ()
    assert parse_conditioning_axes("") == ()
    assert parse_conditioning_axes("charge") == ("charge",)
    assert parse_conditioning_axes("hmoment,charge") == ("charge", "hmoment")  # canonical order
    assert parse_conditioning_axes("hmoment,hydro,charge") == CONDITIONING_AXES
    for bad in ("gravvy", "charge,charge", "charge,,hydro", ",", "charge;hydro"):
        with pytest.raises(ValueError):
            parse_conditioning_axes(bad)


# ---------------------------------------------------------------------------
# Joint bin sampling (Hamilton over reference joint cells)
# ---------------------------------------------------------------------------

REF = [
    "GLFDIVKKVVGALGSL",
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "KKKKKKKK",
    "FLPAIWAAAKFL",
    "WLRRIRKIAAHR",
    "KLLKLLKKLLKAAK",
    "WWWWWWWW",
    "GLFDIVKKVVGALGSK",
]
AXES = ("charge", "hydro", "hmoment")


def _flat_index(draw):
    return draw["charge"] * NUM_HYDRO_BINS * NUM_HMOMENT_BINS + draw["hydro"] * NUM_HMOMENT_BINS + draw["hmoment"]


def test_joint_draws_match_reference_joint_exactly_when_divisible():
    n = 8 * 125  # multiple of |REF| → Hamilton reproduces proportions exactly
    draws = sample_condition_bins(n, REF, axes=AXES, seed=11)
    props = reference_joint_proportions(REF, AXES).reshape(-1)
    empirical = np.zeros_like(props)
    for draw in draws:
        empirical[_flat_index(draw)] += 1
    assert total_variation_distance(empirical, props) == pytest.approx(0.0, abs=1e-12)
    # marginals of the joint draws equal the per-axis marginals
    charge_props = np.zeros(NUM_CHARGE_BINS)
    for seq in REF:
        charge_props[axis_bin(seq, "charge")] += 1 / len(REF)
    drawn_charge = np.zeros(NUM_CHARGE_BINS)
    for draw in draws:
        drawn_charge[draw["charge"]] += 1
    assert np.allclose(drawn_charge / n, charge_props, atol=1e-12)


def test_joint_draws_hamilton_exactness_on_remainder():
    n = 13  # floors + one remainder unit
    draws = sample_condition_bins(n, REF, axes=AXES, seed=5)
    counts = joint_bin_counts(REF, AXES)
    total = int(counts.sum())
    flat = counts.reshape(-1)
    expected = np.floor(n * flat / total).astype(np.int64)
    remainder = n - int(expected.sum())
    fractional = (n * flat) % total
    order = sorted(range(flat.size), key=lambda i: (-fractional[i], i))
    for i in order[:remainder]:
        expected[i] += 1
    empirical = np.zeros_like(flat)
    for draw in draws:
        empirical[_flat_index(draw)] += 1
    assert (empirical == expected).all()


def test_joint_draws_deterministic_and_ordered_shuffle():
    first = sample_condition_bins(100, REF, axes=AXES, seed=42)
    second = sample_condition_bins(100, REF, axes=AXES, seed=42)
    assert first == second
    other = sample_condition_bins(100, REF, axes=AXES, seed=43)
    assert sorted(map(_flat_index, first)) == sorted(map(_flat_index, other))  # same multiset
    assert first != other  # different shuffle order


def test_joint_draws_validation():
    with pytest.raises(ValueError):
        sample_condition_bins(10, [], axes=AXES)
    with pytest.raises(ValueError):
        sample_condition_bins(-1, REF, axes=AXES)
    with pytest.raises(ValueError):
        sample_condition_bins(10, REF, axes=())


def test_total_validation_distance_known_values():
    target = np.array([0.5, 0.5])
    assert total_variation_distance(np.array([5, 5]), target) == pytest.approx(0.0)
    assert total_variation_distance(np.array([10, 0]), target) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        total_variation_distance(np.array([1, 2, 3]), target)
    with pytest.raises(ValueError):
        total_variation_distance(np.array([0, 0]), target)
