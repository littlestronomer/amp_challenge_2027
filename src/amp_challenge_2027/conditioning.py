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

All functions are deterministic.
"""

from __future__ import annotations

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


__all__ = [
    "CHARGE_MIN",
    "CHARGE_MAX",
    "NUM_CHARGE_BINS",
    "charge_modlamp",
    "charge_bin",
    "bin_to_charge",
    "reference_charge_counts",
    "reference_charge_proportions",
    "sample_charge_bins",
]
