"""Peptide physicochemical properties and biological plausibility rules.

This is the **biology→ML interface** that encodes your collaborator's domain
knowledge as computable functions. Every constant and threshold below is a
knob the molecular-genetics side of the team owns; the ML code treats them as
fixed rules. Phase-1 scoring explicitly rewards "property distribution
conformity" to known AMPs — these definitions make that objective concrete.

Reference scales:
  - Charge: standard sidechain pKa values at pH 7 (Lehninger).
  - Hydrophobicity: Kyte-Doolittle (1982).
  - Helical hydrophobic moment: Eisenberg, Weiss & Terwilliger (1982, 1984).

All functions are pure and dependency-free (numpy only) so they can run inside
the deterministic inference path and inside tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Biophysical scales — owned by the biology side. Tweak thresholds, not scales.
# ---------------------------------------------------------------------------

# Net charge at pH 7 (sidechain + termini). N-term +1, C-term -1.
CHARGE_AT_PH7 = {
    "K": +1.0, "R": +1.0, "H": +0.1,
    "D": -1.0, "E": -1.0,
    "C": 0.0, "M": 0.0, "S": 0.0, "T": 0.0, "N": 0.0, "Q": 0.0,
    "G": 0.0, "P": 0.0, "A": 0.0, "V": 0.0, "L": 0.0, "I": 0.0,
    "F": 0.0, "Y": 0.0, "W": 0.0,
}

# Kyte-Doolittle hydrophobicity (more positive = more hydrophobic).
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Eisenberg consensus hydrophobicity (used for the hydrophobic moment).
EISENBERG_HYDRO = {
    "I": 0.73, "V": 0.54, "L": 0.53, "F": 0.61, "C": 0.04, "M": 0.26, "A": 0.62,
    "G": 0.48, "T": 0.26, "S": -0.18, "W": 0.37, "Y": -0.09, "P": -0.07,
    "H": -0.40, "E": -0.62, "Q": -0.69, "D": -0.90, "N": -0.78, "K": -1.50, "R": -2.53,
}


# ---------------------------------------------------------------------------
# Core property functions
# ---------------------------------------------------------------------------


@dataclass
class PeptideProperties:
    """Computed biophysical descriptors for a single sequence."""

    sequence: str
    length: int
    charge: float
    hydrophobicity_kd: float  # mean Kyte-Doolittle
    hydrophobic_moment: float  # Eisenberg helical moment at 100°
    cysteine_count: int
    fraction_positive: float
    fraction_hydrophobic: float


def net_charge(seq: str, *, ph7: bool = True) -> float:
    """Net charge including termini (N +1, C -1)."""
    table = CHARGE_AT_PH7
    z = sum(table.get(aa, 0.0) for aa in seq)
    return z + 1.0 - 1.0  # termini cancel at neutral pH


def mean_hydrophobicity(seq: str) -> float:
    """Mean Kyte-Doolittle hydrophobicity."""
    if not seq:
        return 0.0
    return float(np.mean([KYTE_DOOLITTLE[a] for a in seq]))


def hydrophobic_moment(seq: str, *, angle_deg: float = 100.0) -> float:
    """Eisenberg helical hydrophobic moment (µH).

    For an α-helix (100° per residue), this measures amphipathicity: high µH
    means hydrophobic and polar residues segregate onto opposite helix faces —
    a hallmark of membrane-active AMPs. Defined in Eisenberg et al. (1982).
    """
    if len(seq) < 2:
        return 0.0
    theta = math.radians(angle_deg)
    h = [EISENBERG_HYDRO[a] for a in seq]
    sin_sum = sum(h[i] * math.sin(i * theta) for i in range(len(h)))
    cos_sum = sum(h[i] * math.cos(i * theta) for i in range(len(h)))
    return math.sqrt(sin_sum**2 + cos_sum**2) / len(h)


def compute_properties(seq: str) -> PeptideProperties:
    """Compute all descriptors for a sequence."""
    n = len(seq)
    pos = sum(1 for a in seq if a in "KR")
    hydrophobic = sum(1 for a in seq if a in "AVLIFWYM")
    return PeptideProperties(
        sequence=seq,
        length=n,
        charge=net_charge(seq),
        hydrophobicity_kd=mean_hydrophobicity(seq),
        hydrophobic_moment=hydrophobic_moment(seq),
        cysteine_count=seq.count("C"),
        fraction_positive=pos / n if n else 0.0,
        fraction_hydrophobic=hydrophobic / n if n else 0.0,
    )


# ---------------------------------------------------------------------------
# Biological plausibility rules — the collaborator's filters.
#
# Each rule returns True if the peptide PASSES (is plausible). Defaults are
# intentionally permissive; tighten them based on literature/domain judgment.
# These run during candidate selection (see select.py) before any peptide can
# consume a wet-lab slot.
# ---------------------------------------------------------------------------


# Empirical AMP property ranges (broad; refine with collaborator).
AMP_CHARGE_MIN, AMP_CHARGE_MAX = -2.0, 12.0
AMP_HYDRO_KD_MIN, AMP_HYDRO_KD_MAX = -3.0, 1.5
AMP_HMOMENT_MIN = 0.0  # informational; no hard floor by default
MAX_CONSECUTIVE_HYDROPHOBIC = 8  # longer runs risk aggregation/haemo
MAX_CYSTEINE_FRACTION = 0.25  # unpaired Cys → misfolding/disulfide chaos
# Known hemolysis-prone motifs (seed list — your collaborator expands this).
HEMOLYTIC_MOTIFS: tuple[str, ...] = ("LLLL", "FFFF", "WWWW")


def _longest_run(seq: str, members: str) -> int:
    longest = run = 0
    for a in seq:
        run = run + 1 if a in members else 0
        longest = max(longest, run)
    return longest


@dataclass
class PlausibilityVerdict:
    passed: bool
    reasons: list[str]


def check_plausibility(seq: str) -> PlausibilityVerdict:
    """Run all biological plausibility rules. Returns pass + failure reasons."""
    reasons: list[str] = []
    p = compute_properties(seq)

    if not (AMP_CHARGE_MIN <= p.charge <= AMP_CHARGE_MAX):
        reasons.append(f"charge {p.charge:.2f} outside [{AMP_CHARGE_MIN}, {AMP_CHARGE_MAX}]")
    if not (AMP_HYDRO_KD_MIN <= p.hydrophobicity_kd <= AMP_HYDRO_KD_MAX):
        reasons.append(
            f"hydrophobicity {p.hydrophobicity_kd:.2f} outside "
            f"[{AMP_HYDRO_KD_MIN}, {AMP_HYDRO_KD_MAX}]"
        )
    if p.length and p.cysteine_count / p.length > MAX_CYSTEINE_FRACTION:
        reasons.append(f"cysteine fraction too high ({p.cysteine_count}/{p.length})")
    if _longest_run(seq, "AVILFW") > MAX_CONSECUTIVE_HYDROPHOBIC:
        reasons.append(f"hydrophobic run > {MAX_CONSECUTIVE_HYDROPHOBIC}")
    for motif in HEMOLYTIC_MOTIFS:
        if motif in seq:
            reasons.append(f"hemolytic motif {motif!r}")

    return PlausibilityVerdict(passed=not reasons, reasons=reasons)


def is_plausible(seq: str) -> bool:
    """Boolean shortcut for ``check_plausibility(seq).passed``."""
    return check_plausibility(seq).passed


__all__ = [
    "PeptideProperties",
    "PeptideProperties",
    "net_charge",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "compute_properties",
    "PlausibilityVerdict",
    "check_plausibility",
    "is_plausible",
    "AMP_CHARGE_MIN",
    "AMP_CHARGE_MAX",
    "AMP_HYDRO_KD_MIN",
    "AMP_HYDRO_KD_MAX",
]
