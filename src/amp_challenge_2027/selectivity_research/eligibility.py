from __future__ import annotations

import numpy as np
from amp_challenge_2027.config import STANDARD_AMINO_ACIDS, MIN_LENGTH, MAX_LENGTH
from amp_challenge_2027.props import is_plausible


def scan(frame, reference: set[str]) -> tuple[np.ndarray, list[str]]:
    """Full-pool cheap gates. Exact top novelty is checked after solving."""
    eligible = np.zeros(len(frame), dtype=bool)
    reasons = []
    for i, sequence in enumerate(frame["sequence"].astype(str)):
        codes = []
        if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):
            codes.append("length")
        if set(sequence) - set(STANDARD_AMINO_ACIDS):
            codes.append("alphabet")
        if sequence in reference:
            codes.append("exact_reference_overlap")
        if not is_plausible(sequence):
            codes.append("custom_plausibility")
        if not np.isfinite(frame.iloc[i][["activity", "hemolysis_risk", "conformity", "precision"]].to_numpy(dtype=float)).all():
            codes.append("missing_score")
        eligible[i] = not codes
        reasons.append("eligible" if not codes else ";".join(codes))
    return eligible, reasons
