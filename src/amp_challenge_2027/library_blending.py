"""Deterministic two-pool blends with exact, auditable source quotas."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd


def parse_ratio(value: str) -> tuple[int, int]:
    """Parse positive integer primary:secondary weights, reducing equivalents."""
    try:
        primary, secondary = map(int, value.split(":"))
    except (ValueError, TypeError) as error:
        raise ValueError("ratio must be positive integer weights, e.g. 7:1, 3:1 or 1:1") from error
    if primary < 1 or secondary < 1:
        raise ValueError("both ratio weights must be positive")
    divisor = gcd(primary, secondary)
    return primary // divisor, secondary // divisor


def source_quotas(size: int, ratio: tuple[int, int]) -> tuple[int, int]:
    primary, secondary = ratio
    if size < 1 or primary < 1 or secondary < 1:
        raise ValueError("library size and both ratio weights must be positive")
    numerator = size * primary
    denominator = primary + secondary
    if numerator % denominator:
        raise ValueError(f"Ratio {primary}:{secondary} cannot give exact integer counts at size {size}")
    primary_count = numerator // denominator
    return primary_count, size - primary_count


@dataclass
class FixedBlend:
    sequences: list[str]
    sources: list[str]
    primary_count: int
    secondary_count: int
    shared_pool_count: int
    selected_shared_count: int


def fixed_ratio_blend(
    primary: list[str], secondary: list[str], *, size: int, ratio: tuple[int, int],
) -> FixedBlend:
    """Select exact quotas, then interleave their selected sequences in blocks.

    Pools must already be valid, unique and reference-free. Shared sequences
    are assigned once: favor the primary prefix, reserving shared candidates
    for the secondary only when needed to meet its quota. This avoids both
    duplicate-induced ratio drift and avoidable exhaustion of the secondary.
    Source labels describe selection provenance, not exclusive model support.
    """
    n_primary, n_secondary = source_quotas(size, ratio)
    p_set, s_set = set(primary), set(secondary)
    if len(p_set) != len(primary) or len(s_set) != len(secondary):
        raise ValueError("Each component pool must be unique")
    if len(primary) < n_primary or len(secondary) < n_secondary or len(p_set | s_set) < size:
        raise ValueError(f"Insufficient unique candidates for exact quotas {n_primary}+{n_secondary}")
    shared = p_set & s_set
    reserve_count = max(0, n_secondary - len(s_set - p_set))
    reserved = set([seq for seq in secondary if seq in shared][:reserve_count])
    selected_primary = [seq for seq in primary if seq not in reserved][:n_primary]
    selected_primary_set = set(selected_primary)
    selected_secondary = [seq for seq in secondary if seq not in selected_primary_set][:n_secondary]
    if len(selected_primary) != n_primary or len(selected_secondary) != n_secondary:
        raise ValueError("Unable to satisfy exact source quotas")
    sequences, sources = [], []
    p_weight, s_weight = ratio
    p_index = s_index = 0
    while p_index < n_primary or s_index < n_secondary:
        block = selected_primary[p_index:p_index + p_weight]
        sequences.extend(block)
        sources.extend(["primary"] * len(block))
        p_index += len(block)
        block = selected_secondary[s_index:s_index + s_weight]
        sequences.extend(block)
        sources.extend(["secondary"] * len(block))
        s_index += len(block)
    return FixedBlend(
        sequences, sources, n_primary, n_secondary, len(shared),
        sum(seq in shared for seq in sequences),
    )
