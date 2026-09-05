"""Experimental library membership selection with fixed length/charge counts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from amp_challenge_2027.conditioning import charge_bin


def stratum(sequence: str) -> tuple[int, int]:
    return len(sequence), charge_bin(sequence)


def select_stratified_library(
    baseline: list[str], pool: list[str], scores: np.ndarray, *, strength: float,
) -> list[str]:
    """Replace at most floor(strength * bin_size) members in each joint bin.

    Zero reproduces the baseline including order. Each bin retains its best
    baseline members and fills the remaining slots by score. Counts in every
    joint length/charge bin remain EXACTLY unchanged. Scores must be finite,
    aligned with the unique pool, and higher-is-better. Novelty/validity must
    be checked by the caller; this does not guarantee diversity or activity.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if len(set(baseline)) != len(baseline) or len(set(pool)) != len(pool):
        raise ValueError("baseline and pool must each be unique")
    if not set(baseline).issubset(pool):
        raise ValueError("pool must include every baseline sequence")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(pool),) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and aligned with the pool")
    if strength == 0:
        return list(baseline)
    score_by_seq = dict(zip(pool, scores))
    slots: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: dict[tuple[int, int], list[str]] = defaultdict(list)
    for i, seq in enumerate(baseline):
        slots[stratum(seq)].append(i)
    for seq in pool:
        key = stratum(seq)
        if key in slots:
            candidates[key].append(seq)
    output = list(baseline)
    for key, indices in slots.items():
        budget = int(strength * len(indices))
        if budget == 0:
            continue
        # Stable ties favor baseline members over unnecessary replacements.
        worst_slots = sorted(indices, key=lambda i: score_by_seq[baseline[i]])[:budget]
        replaceable = [baseline[i] for i in worst_slots]
        existing = {baseline[i] for i in indices}
        options = replaceable + [s for s in candidates[key] if s not in existing]
        best = sorted(options, key=lambda s: -score_by_seq[s])[:budget]
        for i, seq in zip(worst_slots, best):
            output[i] = seq
    return output
