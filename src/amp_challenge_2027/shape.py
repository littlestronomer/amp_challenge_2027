"""Property-distribution shaping for the AMP library.

This is a deterministic post-selection step that reshapes a generated candidate
pool so its biophysical property *marginals* — net charge, hydrophobicity, and
helical hydrophobic moment — match the reference AMP set (antibacterial.fasta).
Phase-1 ``Conformity`` measures exactly this: how well the generated library's
property distribution conforms to the reference. The default generation path
keeps "the first N valid sequences", which leaves the distribution narrower than
the reference (notably the charge tails). Bucket-filling against the reference
histogram widens it toward the target, directly improving Conformity.

Design constraints (non-negotiable):
  - **Pure numpy, no new dependencies.** The validator installs only the light
    runtime (numpy + levenshtein) for ``uv run generate``, so any shaping that
    ships in the generation path must run there too. We deliberately do NOT use
    seqme's modlamp-backed predictors here (they need an optional extra).
  - **Deterministic & byte-reproducible.** ``shape_library_to_reference`` is a
    pure function of (candidates, reference, seed). The primary accept pass uses
    no RNG — it walks candidates in stable input order. ``seed`` is accepted for
    API stability but the algorithm is currently fully order-deterministic; it is
    reserved for optional top-up tie-breaking without changing the signature.
  - **Never under-fill.** If the candidate pool lacks sequences for some bins
    (the generator under-produced a tail), top up deterministically with the
    remaining candidates so the output always reaches ``library_size``. This is a
    ceiling on the gain, not a failure.

Property predictors come from ``props.compute_properties`` (Kyte-Doolittle
hydrophobicity, Eisenberg hydrophobic moment, standard-pKa net charge). These are
monotonic proxies for seqme's predictors: charge is near-identical, the other two
are correlated, so matching these marginals pulls the seqme Conformity distance
down sharply.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from amp_challenge_2027.props import compute_properties

# Property axes, in canonical order. Must stay aligned with ``_property_vector``
# and the three ``_Binning`` objects built in ``shape_library_to_reference``.
#   axis 0: net charge            (integer bins)
#   axis 1: hydrophobicity (KD)   (equal-width bins over reference range)
#   axis 2: hydrophobic moment    (equal-width bins over reference range)


def _property_vector(seq: str) -> tuple[float, float, float]:
    """The three conformity-relevant properties as a tuple."""
    p = compute_properties(seq)
    return (p.charge, p.hydrophobicity_kd, p.hydrophobic_moment)


@dataclass
class PropertyStats:
    """Summary statistics for the three property axes (for logging/diagnostics)."""

    n: int
    charge_mean: float
    charge_std: float
    hydro_mean: float
    hydro_std: float
    hmoment_mean: float
    hmoment_std: float

    def __str__(self) -> str:
        if self.n == 0:
            return "PropertyStats(n=0)"
        return (
            f"n={self.n}  "
            f"charge={self.charge_mean:+.2f}±{self.charge_std:.2f}  "
            f"hydro={self.hydro_mean:+.2f}±{self.hydro_std:.2f}  "
            f"hmoment={self.hmoment_mean:.3f}±{self.hmoment_std:.3f}"
        )


def property_stats(seqs: list[str]) -> PropertyStats:
    """Mean/std of the three property axes over ``seqs`` (empty → zeros)."""
    if not seqs:
        return PropertyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.array([_property_vector(s) for s in seqs], dtype=np.float64)
    return PropertyStats(
        n=len(seqs),
        charge_mean=float(arr[:, 0].mean()),
        charge_std=float(arr[:, 0].std()),
        hydro_mean=float(arr[:, 1].mean()),
        hydro_std=float(arr[:, 1].std()),
        hmoment_mean=float(arr[:, 2].mean()),
        hmoment_std=float(arr[:, 2].std()),
    )


# ---------------------------------------------------------------------------
# 1-D binning specs derived from the reference distribution
# ---------------------------------------------------------------------------


@dataclass
class _Binning:
    """Histogram spec + per-bin target quota for one property axis.

    ``kind == "integer"``: bin index = round(value). Used for net charge, which
    is effectively discrete (integer-valued at neutral pH once rounded). Quota is
    keyed by the integer bin and sparse (only bins present in the reference have
    nonzero quota).

    ``kind == "uniform"``: ``n_bins`` equal-width bins over ``[lo, hi]`` (the
    reference min/max). Quota is keyed by bin index in ``[0, n_bins)``.
    """

    kind: str
    lo: float
    hi: float
    n_bins: int
    quota: dict[int, float]  # bin index -> target count for the output library

    def bin_of(self, v: float) -> int:
        if self.kind == "integer":
            return int(round(v))
        if self.hi <= self.lo:
            return 0
        b = int((v - self.lo) / (self.hi - self.lo) * self.n_bins)
        return max(0, min(self.n_bins - 1, b))

    def quota_for(self, bin_index: int) -> float:
        return self.quota.get(bin_index, 0.0)


def _build_charge_binning(ref_charge: np.ndarray, library_size: int) -> _Binning:
    """Integer binning for net charge (bin = round(charge))."""
    bins = np.rint(ref_charge).astype(int)
    unique, counts = np.unique(bins, return_counts=True)
    total = max(int(ref_charge.size), 1)
    quota = {int(u): round(int(c) / total * library_size) for u, c in zip(unique, counts)}
    return _Binning(kind="integer", lo=0.0, hi=0.0, n_bins=0, quota=quota)


def _build_uniform_binning(
    ref_values: np.ndarray, library_size: int, n_bins: int
) -> _Binning:
    """Equal-width binning over the reference [min, max]."""
    lo = float(ref_values.min())
    hi = float(ref_values.max())
    if hi <= lo:
        # Degenerate range → single bin holding everything.
        return _Binning(
            kind="uniform", lo=lo, hi=hi, n_bins=1, quota={0: float(library_size)}
        )
    idx = np.clip(((ref_values - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    unique, counts = np.unique(idx, return_counts=True)
    total = max(int(ref_values.size), 1)
    quota = {int(u): round(int(c) / total * library_size) for u, c in zip(unique, counts)}
    return _Binning(kind="uniform", lo=lo, hi=hi, n_bins=n_bins, quota=quota)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def shape_library_to_reference(
    candidates: list[str],
    reference_seqs: list[str],
    *,
    library_size: int,
    seed: int = 42,
    n_bins: int = 20,
) -> list[str]:
    """Select ``library_size`` candidates whose property marginals match the reference.

    The reference defines the target histogram for each of the three properties.
    We then walk ``candidates`` in stable input order and accept a candidate iff
    it is under-quota in **all three** properties simultaneously (greedy marginal
    matching). This is a single deterministic pass — no RNG, no reordering. If
    the pool cannot satisfy every quota (the generator under-produced a tail),
    we deterministically top up with the remaining candidates in input order so
    the output always reaches ``library_size`` (when the pool is large enough).

    Args:
        candidates: already-valid, deduplicated, no-overlap sequences in a stable
            order (the order is part of the reproducibility contract). Typically
            the output of ``filter_valid`` + ``remove_exact_overlap``.
        reference_seqs: the reference AMP set (e.g. antibacterial.fasta). Defines
            the target property marginals.
        library_size: desired output size (e.g. 50_000).
        seed: reserved for future top-up tie-breaking; currently unused because
            the algorithm is fully order-deterministic. Accepted for API stability.
        n_bins: number of equal-width bins for the two continuous axes.

    Returns:
        A list of ``min(library_size, len(candidates))`` sequences. The first
        ``k <= library_size`` entries are quota-accepted (distribution-shaped);
        any remainder (when the pool under-fills some bins) is filled by input
        order. Always ≤ ``library_size`` and in deterministic order.
    """
    # Edge cases: nothing to shape from.
    if library_size <= 0 or not candidates:
        return []
    if not reference_seqs:
        # No reference → no shaping possible; preserve original first-N behaviour.
        return candidates[:library_size]
    if len(candidates) <= library_size:
        return list(candidates)

    # --- 1. Build the three target binnings from the reference ---------------
    ref_arr = np.array([_property_vector(s) for s in reference_seqs], dtype=np.float64)
    binning = (
        _build_charge_binning(ref_arr[:, 0], library_size),
        _build_uniform_binning(ref_arr[:, 1], library_size, n_bins),
        _build_uniform_binning(ref_arr[:, 2], library_size, n_bins),
    )

    # --- 2. Precompute candidate property vectors (single pass over props) ---
    cand_arr = np.array([_property_vector(s) for s in candidates], dtype=np.float64)
    cand_bins = np.stack(
        [
            np.array([binning[i].bin_of(v) for v in cand_arr[:, i]], dtype=np.int64)
            for i in range(3)
        ],
        axis=1,
    )

    # --- 3. Greedy marginal-matching accept pass ----------------------------
    # Running count per bin, per property. defaultdict so unseen bins default to
    # 0 (their quota is also 0 via ``quota_for`` → never accepted).
    counts = (defaultdict(int), defaultdict(int), defaultdict(int))
    accepted: list[str] = []
    accepted_idx: list[int] = []
    rejected_idx: list[int] = []
    for i in range(len(candidates)):
        b0, b1, b2 = int(cand_bins[i, 0]), int(cand_bins[i, 1]), int(cand_bins[i, 2])
        if (
            counts[0][b0] < binning[0].quota_for(b0)
            and counts[1][b1] < binning[1].quota_for(b1)
            and counts[2][b2] < binning[2].quota_for(b2)
        ):
            counts[0][b0] += 1
            counts[1][b1] += 1
            counts[2][b2] += 1
            accepted.append(candidates[i])
            accepted_idx.append(i)
            if len(accepted) >= library_size:
                break
        else:
            rejected_idx.append(i)

    # --- 4. Deterministic top-up (only if quotas couldn't be met) -----------
    if len(accepted) < library_size:
        for i in rejected_idx:
            accepted.append(candidates[i])
            if len(accepted) >= library_size:
                break

    return accepted[:library_size]


__all__ = [
    "PropertyStats",
    "property_stats",
    "shape_library_to_reference",
]
