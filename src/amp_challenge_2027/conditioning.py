"""Charge-conditioning utilities: labels, reference distribution, sampling.

Charge conditioning makes the AR generator controllable by net charge: the
model is trained with each peptide's modlamp net charge as a conditioning bin,
and at generation time bins are drawn from the *reference* charge distribution
so the produced library reproduces it. This fixes the root cause of the narrow
generated charge distribution (pool σ 2.44 vs reference σ 3.28) that forces the
conformity/F3 tradeoff.

The charge here is the **modlamp Bjellqvist partial charge** (seqme's
ConformityScore scale) — verified bit-for-bit against modlamp 4.x. It is
implemented locally (pure Python, no dependencies) so this module is
self-contained and runs in the minimal validator environment.

Multi-axis conditioning generalizes this to the other two ConformityScore
predictors, hydrophobicity and hydrophobic moment, both on the exact
seqme/modlamp scale:

* seqme ``Hydrophobicity()`` = modlamp ``PeptideDescriptor.load_scale(
  "eisenberg").calculate_global()`` — the mean Eisenberg-scale value over the
  sequence (the 1000-residue default window clamps to the peptide length, so
  max/mean coincide with the whole-sequence mean).
* seqme ``HydrophobicMoment()`` = ``calculate_moment(window=11, angle=100,
  modality="mean")`` — the mean over sliding 11-residue windows of
  µH = |Σ h_k·e^{ikθ}| / window.

Bin edges below are provisional defaults frozen in each checkpoint's
``config.json``; the SSH parity pass (``scripts/parity_modlamp_descriptors.py``)
verifies the descriptor scale and reports reference edge-clamp fractions
before any conditioned training run.

All functions are deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Canonical binning (must match DecoderConfig.num_charge_bins / charge_min).
CHARGE_MIN = -8
CHARGE_MAX = 12
NUM_CHARGE_BINS = CHARGE_MAX - CHARGE_MIN + 1  # 21

# Bjellqvist pKa values (CRC Handbook), exactly as modlamp's _charge() uses.
# Each terminus is counted once per peptide.
_POS_PKA = {"Nterm": 9.38, "K": 10.67, "R": 12.10, "H": 6.04}
_NEG_PKA = {"Cterm": 2.15, "D": 3.71, "E": 4.15, "C": 8.14, "Y": 10.10}


def _partial_positive(pka: float, ph: float) -> float:
    """Fraction protonated (>0) of a basic group at ``ph``."""
    x = 10.0 ** (pka - ph)
    return x / (x + 1.0)


def _partial_negative(pka: float, ph: float) -> float:
    """Fraction deprotonated (>0, negative contribution) of an acidic group."""
    x = 10.0 ** (ph - pka)
    return x / (x + 1.0)


def charge_modlamp(seq: str, *, ph: float = 7.0) -> float:
    """Net charge via the Bjellqvist partial-charge method.

    Matches ``modlamp.descriptors.GlobalDescriptor.calculate_charge(ph=ph)``
    exactly (the value seqme's Charge predictor reports): sum of protonated
    fractions over basic groups minus deprotonated fractions over acidic
    groups, termini counted once each, rounded to 3 dp.
    """
    pos = sum(
        seq.count(a) * _partial_positive(pka, ph)
        for a, pka in _POS_PKA.items()
        if a != "Nterm"
    )
    pos += _partial_positive(_POS_PKA["Nterm"], ph)
    neg = sum(
        seq.count(a) * _partial_negative(pka, ph)
        for a, pka in _NEG_PKA.items()
        if a != "Cterm"
    )
    neg += _partial_negative(_NEG_PKA["Cterm"], ph)
    return round(pos - neg, 3)


def charge_bin(seq: str, *, charge_min: int = CHARGE_MIN, num_bins: int = NUM_CHARGE_BINS) -> int:
    """Integer charge bin of one sequence (modlamp charge, rounded + clamped)."""
    b = int(round(charge_modlamp(seq))) - charge_min
    return max(0, min(num_bins - 1, b))


def bin_to_charge(bin_index: int, *, charge_min: int = CHARGE_MIN) -> int:
    """Nominal (integer) net charge of a bin index."""
    return bin_index + charge_min


def reference_charge_counts(
    reference_seqs: Sequence[str],
    *,
    num_bins: int = NUM_CHARGE_BINS,
    charge_min: int = CHARGE_MIN,
) -> np.ndarray:
    """Histogram of reference charge bins, shape ``(num_bins,)`` of counts."""
    counts = np.zeros(num_bins, dtype=np.int64)
    for seq in reference_seqs:
        counts[charge_bin(seq, charge_min=charge_min, num_bins=num_bins)] += 1
    return counts


def reference_charge_proportions(
    reference_seqs: Sequence[str],
    *,
    num_bins: int = NUM_CHARGE_BINS,
    charge_min: int = CHARGE_MIN,
) -> np.ndarray:
    """Reference charge-bin proportions, shape ``(num_bins,)``, sums to 1."""
    counts = reference_charge_counts(reference_seqs, num_bins=num_bins, charge_min=charge_min)
    total = counts.sum()
    if total == 0:
        raise ValueError("reference set is empty; cannot build charge distribution")
    return counts / total


def sample_charge_bins(
    n: int,
    proportions: np.ndarray,
    *,
    seed: int = 42,
) -> list[int]:
    """Draw ``n`` charge bins matching ``proportions`` (seeded, deterministic).

    Deterministic allocation: bins are filled by floor(quota) first, then the
    remainder by largest fractional part (Hamilton apportionment), so the drawn
    counts match the target proportions as closely as possible. The seeded RNG
    only shuffles the *order* of the bins (interleaving bins for batching
    diversity); the multiset of bins is unchanged by shuffling.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    quotas = np.asarray(proportions, dtype=np.float64) * n
    counts = np.floor(quotas).astype(np.int64)
    remainder = n - int(counts.sum())
    if remainder > 0:
        # Hamilton: give extra units to bins with the largest fractional parts.
        frac = quotas - counts
        order = np.argsort(-frac, kind="stable")
        for i in range(remainder):
            counts[order[i % len(counts)]] += 1
    bins = np.repeat(np.arange(len(counts)), counts)
    rng = np.random.default_rng(seed)
    rng.shuffle(bins)
    return bins.astype(int).tolist()


# ---------------------------------------------------------------------------
# Multi-axis conditioning (ConformityScore predictor space)
# ---------------------------------------------------------------------------

# modlamp's "eisenberg" scale (modlamp/core.py load_scale), the default for
# seqme's Hydrophobicity and HydrophobicMoment predictors.
EISENBERG_SCALE = {
    "A": 0.62, "R": -2.5, "N": -0.78, "D": -0.9, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.4, "I": 1.4,
    "L": 1.1, "K": -1.5, "M": 0.64, "F": 1.2, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.1,
}

# Canonical conditioning axes and their canonical order.
CONDITIONING_AXES = ("charge", "hydro", "hmoment")

# Provisional bin edges (frozen into each checkpoint's config.json; validate
# with scripts/parity_modlamp_descriptors.py before conditioned training).
# Reference (seqme scale): hydro −0.067 ± 0.381, hmoment +0.395 ± 0.199 —
# 20 bins of 0.13 from −1.5 and 14 bins of 0.08 from 0.0 cover ±3σ with clamp.
NUM_HYDRO_BINS = 20
HYDRO_MIN = -1.5
HYDRO_STEP = 0.13
NUM_HMOMENT_BINS = 14
HMOMENT_MIN = 0.0
HMOMENT_STEP = 0.08

# Reference-mode descriptor values used as the neutral default bin for an
# unconditioned forward pass on a conditioned model (mirrors the charge
# default of +3). Reference means on the seqme scale.
HYDRO_MODE_VALUE = -0.067
HMOMENT_MODE_VALUE = 0.395


def hydrophobicity_modlamp(seq: str) -> float:
    """Mean Eisenberg-scale hydrophobicity — seqme ``Hydrophobicity()``.

    Matches modlamp ``PeptideDescriptor.load_scale("eisenberg").calculate_global()``
    for peptides shorter than the 1000-residue default window (always true
    here): one whole-sequence window, so the ``max``/``mean`` modalities and
    the window default are all the plain residue mean.
    """
    if not seq:
        raise ValueError("hydrophobicity requires a non-empty sequence")
    try:
        return sum(EISENBERG_SCALE[residue] for residue in seq) / len(seq)
    except KeyError as error:
        raise ValueError(f"non-standard amino acid {error} in sequence") from error


def hydrophobic_moment_modlamp(seq: str, *, window: int = 11, angle: float = 100.0) -> float:
    """Mean sliding-window hydrophobic moment — seqme ``HydrophobicMoment()``.

    Matches modlamp ``calculate_moment(window=11, angle=100, modality="mean")``:
    for each window of ``min(window, len)`` residues,
    µH = sqrt(Σ h·sin(kθ)² + Σ h·cos(kθ)²) / window, averaged over all windows.
    """
    if not seq:
        raise ValueError("hydrophobic moment requires a non-empty sequence")
    try:
        values = [EISENBERG_SCALE[residue] for residue in seq]
    except KeyError as error:
        raise ValueError(f"non-standard amino acid {error} in sequence") from error
    if window < 1:
        raise ValueError("window must be >= 1")
    wdw = min(window, len(values))
    theta = angle * (math.pi / 180.0)
    cos = [math.cos(k * theta) for k in range(wdw)]
    sin = [math.sin(k * theta) for k in range(wdw)]
    moments = []
    for start in range(len(values) - wdw + 1):
        chunk = values[start : start + wdw]
        vcos = sum(chunk[k] * cos[k] for k in range(wdw))
        vsin = sum(chunk[k] * sin[k] for k in range(wdw))
        moments.append(math.sqrt(vsin * vsin + vcos * vcos) / wdw)
    return sum(moments) / len(moments)


def hydrophobicity_bin(
    seq: str,
    *,
    hydro_min: float = HYDRO_MIN,
    step: float = HYDRO_STEP,
    num_bins: int = NUM_HYDRO_BINS,
) -> int:
    """Integer hydrophobicity bin (rounded offset, clamped to the edges)."""
    offset = (hydrophobicity_modlamp(seq) - hydro_min) / step
    return max(0, min(num_bins - 1, int(round(offset))))


def hmoment_bin(
    seq: str,
    *,
    hmoment_min: float = HMOMENT_MIN,
    step: float = HMOMENT_STEP,
    num_bins: int = NUM_HMOMENT_BINS,
) -> int:
    """Integer hydrophobic-moment bin (rounded offset, clamped to the edges)."""
    offset = (hydrophobic_moment_modlamp(seq) - hmoment_min) / step
    return max(0, min(num_bins - 1, int(round(offset))))


def parse_conditioning_axes(value: str) -> tuple[str, ...]:
    """Parse a conditioning spec into canonical-order unique axes.

    ``"none"``/empty → ``()``. Otherwise a comma-separated subset of
    ``CONDITIONING_AXES`` (any order, duplicates rejected), returned in the
    canonical order ``("charge", "hydro", "hmoment")`` so the config string is
    a stable function of the axis set.
    """
    if value in ("", "none"):
        return ()
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid conditioning {value!r}: empty axis")
    unknown = [part for part in parts if part not in CONDITIONING_AXES]
    if unknown:
        raise ValueError(
            f"unknown conditioning axes {unknown}; expected subsets of {list(CONDITIONING_AXES)}"
        )
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate conditioning axis in {value!r}")
    return tuple(axis for axis in CONDITIONING_AXES if axis in parts)


def axis_bin(seq: str, axis: str) -> int:
    """Per-axis conditioning bin of one sequence (default edges)."""
    if axis == "charge":
        return charge_bin(seq)
    if axis == "hydro":
        return hydrophobicity_bin(seq)
    if axis == "hmoment":
        return hmoment_bin(seq)
    raise ValueError(f"unknown conditioning axis {axis!r}")


def axis_num_bins(axis: str) -> int:
    """Default number of bins for a conditioning axis."""
    if axis == "charge":
        return NUM_CHARGE_BINS
    if axis == "hydro":
        return NUM_HYDRO_BINS
    if axis == "hmoment":
        return NUM_HMOMENT_BINS
    raise ValueError(f"unknown conditioning axis {axis!r}")


def axis_default_bin(axis: str) -> int:
    """Neutral default bin (reference-mode descriptor value) for an axis."""
    if axis == "charge":
        return max(0, min(NUM_CHARGE_BINS - 1, 3 - CHARGE_MIN))
    if axis == "hydro":
        offset = (HYDRO_MODE_VALUE - HYDRO_MIN) / HYDRO_STEP
        return max(0, min(NUM_HYDRO_BINS - 1, int(round(offset))))
    if axis == "hmoment":
        offset = (HMOMENT_MODE_VALUE - HMOMENT_MIN) / HMOMENT_STEP
        return max(0, min(NUM_HMOMENT_BINS - 1, int(round(offset))))
    raise ValueError(f"unknown conditioning axis {axis!r}")


def condition_bins(seq: str, axes: Sequence[str] = CONDITIONING_AXES) -> dict[str, int]:
    """Per-axis conditioning bins of one sequence."""
    return {axis: axis_bin(seq, axis) for axis in axes}


def joint_bin_counts(
    reference_seqs: Sequence[str],
    axes: Sequence[str] = CONDITIONING_AXES,
) -> np.ndarray:
    """Histogram of reference joint per-axis bins, shape ``(bins(axis)...)``."""
    axes = tuple(axes)
    if not axes:
        raise ValueError("at least one conditioning axis is required")
    shape = tuple(axis_num_bins(axis) for axis in axes)
    counts = np.zeros(shape, dtype=np.int64)
    for seq in reference_seqs:
        counts[tuple(axis_bin(seq, axis) for axis in axes)] += 1
    return counts


def reference_joint_proportions(
    reference_seqs: Sequence[str],
    axes: Sequence[str] = CONDITIONING_AXES,
) -> np.ndarray:
    """Reference joint-bin proportions (same shape, sums to 1)."""
    counts = joint_bin_counts(reference_seqs, axes)
    total = counts.sum()
    if total == 0:
        raise ValueError("reference set is empty; cannot build joint distribution")
    return counts / total


def sample_condition_bins(
    n: int,
    reference_seqs: Sequence[str],
    *,
    axes: Sequence[str] = CONDITIONING_AXES,
    seed: int = 42,
) -> list[dict[str, int]]:
    """Draw ``n`` joint per-axis bin dicts matching the reference JOINT distribution.

    Deterministic allocation (Hamilton apportionment over observed joint cells
    with exact integer arithmetic), so the drawn joint histogram matches the
    reference joint distribution as closely as ``n`` allows — including the
    charge/hydro/hmoment correlations the ConformityScore KDE lives in, which
    independent per-marginal draws would destroy. The seeded RNG only shuffles
    the ORDER of the drawn cells; the multiset is unchanged by shuffling.
    """
    axes = tuple(axes)
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    if not axes:
        raise ValueError("at least one conditioning axis is required")
    counts = joint_bin_counts(reference_seqs, axes)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("reference set is empty; cannot draw joint bins")
    flat = counts.reshape(-1)
    quotas = (n * flat).astype(np.int64) // total
    remainder = n - int(quotas.sum())
    if remainder:
        fractional = (n * flat).astype(np.int64) % total
        order = np.argsort(-fractional, kind="stable")
        quotas[order[:remainder]] += 1
    cells = np.repeat(np.arange(flat.size), quotas)
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    unraveled = [np.unravel_index(int(cell), counts.shape) for cell in cells]
    return [
        {axis: int(index) for axis, index in zip(axes, tuple_indices)}
        for tuple_indices in unraveled
    ]


def total_variation_distance(
    drawn_counts: np.ndarray,
    proportions: np.ndarray,
) -> float:
    """TV distance between an empirical draw histogram and target proportions.

    Both arrays share a shape; ``drawn_counts`` holds counts (normalized
    internally). Returns 0.5 * Σ |p_drawn − p_target| — 0 is a perfect match.
    """
    drawn_counts = np.asarray(drawn_counts, dtype=np.float64)
    proportions = np.asarray(proportions, dtype=np.float64)
    if drawn_counts.shape != proportions.shape:
        raise ValueError("drawn counts and proportions must share a shape")
    total = drawn_counts.sum()
    if total <= 0:
        raise ValueError("drawn counts must be non-empty")
    return float(0.5 * np.abs(drawn_counts / total - proportions).sum())


__all__ = [
    "CHARGE_MIN",
    "CHARGE_MAX",
    "NUM_CHARGE_BINS",
    "EISENBERG_SCALE",
    "CONDITIONING_AXES",
    "NUM_HYDRO_BINS",
    "HYDRO_MIN",
    "HYDRO_STEP",
    "NUM_HMOMENT_BINS",
    "HMOMENT_MIN",
    "HMOMENT_STEP",
    "charge_modlamp",
    "charge_bin",
    "bin_to_charge",
    "reference_charge_counts",
    "reference_charge_proportions",
    "sample_charge_bins",
    "hydrophobicity_modlamp",
    "hydrophobic_moment_modlamp",
    "hydrophobicity_bin",
    "hmoment_bin",
    "parse_conditioning_axes",
    "axis_bin",
    "axis_num_bins",
    "axis_default_bin",
    "condition_bins",
    "joint_bin_counts",
    "reference_joint_proportions",
    "sample_condition_bins",
    "total_variation_distance",
]
