"""Deterministic pool blends with exact, auditable source quotas.

Two engines:

* :func:`fixed_ratio_blend` — the original two-pool exact-quota interleave
  (locked-hybrid ratio sweeps; semantics frozen, cached results depend on it).
* :func:`fixed_weights_blend` — the N-way generalization: ``5:2:1``-style
  integer weights over N component pools with largest-remainder (Hamilton)
  quotas, priority-ordered shared-sequence assignment and a
  weight-proportional block interleave.

All functions are deterministic: identical inputs produce byte-identical
libraries. Shared sequences between component pools are assigned exactly once.
"""

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


# ---------------------------------------------------------------------------
# N-way generalization
# ---------------------------------------------------------------------------


def parse_weights(value: str) -> tuple[int, ...]:
    """Parse ``":"``-joined positive integer weights, reducing equivalents.

    ``"5:2:1"`` → ``(5, 2, 1)``; ``"10:4:2"`` → ``(5, 2, 1)``. At least two
    components are required (single-pool "blends" are not a blend).
    """
    try:
        parts = tuple(int(part) for part in value.split(":"))
    except (ValueError, TypeError) as error:
        raise ValueError(
            "weights must be positive integers joined by ':', e.g. 5:2:1"
        ) from error
    if len(parts) < 2 or any(weight < 1 for weight in parts):
        raise ValueError("at least two positive integer weights are required")
    divisor = gcd(*parts)
    return tuple(weight // divisor for weight in parts)


def apportioned_quotas(size: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    """Exact integer quotas summing to ``size`` (largest-remainder / Hamilton).

    Integer arithmetic throughout: fractional parts are compared via the
    remainders ``size * w_i % total`` over the common denominator ``total``,
    with ties broken by component order. For weights that divide ``size``
    exactly this coincides with the exact fractional quotas used by
    :func:`source_quotas`.
    """
    if size < 1 or not weights or any(weight < 1 for weight in weights):
        raise ValueError("size and all weights must be positive")
    total = sum(weights)
    quotas = [size * weight // total for weight in weights]
    remainder = size - sum(quotas)
    if remainder:
        fractional = [size * weight % total for weight in weights]
        order = sorted(range(len(weights)), key=lambda i: (-fractional[i], i))
        for i in order[:remainder]:
            quotas[i] += 1
    return tuple(quotas)


@dataclass
class FixedWeightsBlend:
    """Result of :func:`fixed_weights_blend` with full provenance."""

    sequences: list[str]
    sources: list[str]
    labels: list[str]
    weights: tuple[int, ...]
    quotas: tuple[int, ...]
    pool_sizes: tuple[int, ...]
    pool_shared: tuple[int, ...]
    selected_shared: tuple[int, ...]


def fixed_weights_blend(
    pools: list[list[str]],
    weights: tuple[int, ...],
    *,
    size: int,
    labels: list[str] | None = None,
) -> FixedWeightsBlend:
    """Blend N pools at exact Hamilton quotas, interleaved in weight blocks.

    Selection semantics (deterministic, auditable):

    * Components are processed in list order = priority order. Component ``i``
      selects the prefix of its own pool order among sequences not yet claimed.
    * Shared-with-later candidates are deferred to the back of a component's
      eligible list only when taking the plain prefix would leave some later
      component unable to meet its quota (the N-way form of the two-pool
      "reserve shared candidates for the secondary when needed" rule).
    * If quotas still cannot be met, the blend fails: no silent rebalancing.

    The interleave emits ``weight_i`` sequences per component per cycle in
    component order, so a 2-component blend reproduces the 3:1-style block
    pattern of the locked hybrid (quota selection differs from
    :func:`fixed_ratio_blend` only in overlap-starved cases).
    """
    n = len(pools)
    if n < 2 or len(weights) != n:
        raise ValueError("fixed_weights_blend needs >= 2 pools with matching weights")
    if size < 1:
        raise ValueError("size must be positive")
    quotas = apportioned_quotas(size, weights)
    if labels is None:
        labels = [f"component_{i}" for i in range(n)]
    if len(labels) != n or len(set(labels)) != n:
        raise ValueError("labels must be unique and match the pool count")
    pool_sets = [set(pool) for pool in pools]
    for i, (pool, pool_set, quota) in enumerate(zip(pools, pool_sets, quotas)):
        if len(pool_set) != len(pool):
            raise ValueError(f"Each component pool must be unique (component {labels[i]})")
        if len(pool) < quota:
            raise ValueError(
                f"Insufficient candidates for exact quota in component {labels[i]}: "
                f"{len(pool)} available < {quota}"
            )

    claimed: set[str] = set()
    selected: list[list[str]] = []
    for i in range(n):
        later_union: set[str] = set()
        for j in range(i + 1, n):
            later_union |= pool_sets[j]
        eligible = [seq for seq in pools[i] if seq not in claimed]
        if len(eligible) < quotas[i]:
            raise ValueError(
                f"Unable to satisfy exact quota for component {labels[i]} given prior claims"
            )
        take = eligible[: quotas[i]]

        def downstream_ok(chosen: set[str]) -> bool:
            return all(
                sum(1 for seq in pools[j] if seq not in claimed and seq not in chosen) >= quotas[j]
                for j in range(i + 1, n)
            )

        if not downstream_ok(set(take)):
            # Defer shared-with-later candidates so earlier components do not
            # consume sequences later components depend on (stable order).
            eligible = (
                [seq for seq in eligible if seq not in later_union]
                + [seq for seq in eligible if seq in later_union]
            )
            take = eligible[: quotas[i]]
            if not downstream_ok(set(take)):
                raise ValueError(
                    "Cannot satisfy exact quotas for all components; pools overlap too heavily"
                )
        selected.append(take)
        claimed.update(take)

    sequences: list[str] = []
    sources: list[str] = []
    cursor = [0] * n
    while any(cursor[i] < quotas[i] for i in range(n)):
        for i in range(n):
            block = selected[i][cursor[i] : cursor[i] + weights[i]]
            sequences.extend(block)
            sources.extend([labels[i]] * len(block))
            cursor[i] += len(block)
    if len(sequences) != size or len(set(sequences)) != size:
        raise AssertionError("Blend invariant violated: size or uniqueness")

    def _shared_counts(seq_lists: list[list[str]]) -> tuple[int, ...]:
        others: list[set[str]] = []
        for i in range(n):
            union_without = set()
            for j in range(n):
                if j != i:
                    union_without |= pool_sets[j]
            others.append(union_without)
        return tuple(sum(seq in others[i] for seq in seq_lists[i]) for i in range(n))

    return FixedWeightsBlend(
        sequences=sequences,
        sources=sources,
        labels=list(labels),
        weights=tuple(weights),
        quotas=quotas,
        pool_sizes=tuple(len(pool) for pool in pools),
        pool_shared=_shared_counts(pools),
        selected_shared=_shared_counts(selected),
    )
