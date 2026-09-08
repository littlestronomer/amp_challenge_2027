"""CPU-only diagnostics; deliberately separate from production ranking defaults."""

from __future__ import annotations

import numpy as np


def fit_normalization(parts: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, values in parts.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid normalization input: {name}")
        sd = float(values.std())
        result[name] = {"mean": float(values.mean()), "std": sd,
                        "denominator": sd if sd > 1e-8 else 1.0, "n": len(values)}
    return result


def normalized_score(parts: dict, weights: dict, normalization: dict) -> tuple[np.ndarray, dict]:
    """Match CompositeScorer arithmetic, but allow an explicitly frozen scale."""
    if set(parts) != set(weights) or set(parts) != set(normalization):
        raise ValueError("Normalization components differ")
    total_weight = sum(weights.values())
    if not np.isfinite(list(weights.values())).all() or min(weights.values()) < 0 or total_weight <= 0:
        raise ValueError("Invalid component weights")
    size = len(next(iter(parts.values())))
    combined = np.zeros(size, dtype=np.float64)
    contributions = {}
    for name, weight in weights.items():
        values = np.asarray(parts[name], dtype=np.float64)
        item = normalization[name]
        mean, denominator = item["mean"], item["denominator"]
        if (values.shape != (size,) or not np.isfinite(values).all()
                or not np.isfinite([mean, denominator]).all() or denominator <= 0):
            raise ValueError(f"Invalid normalization: {name}")
        contribution = weight * ((values - mean) / denominator)
        combined += contribution
        contributions[name] = contribution / total_weight
    return (combined / total_weight).astype(np.float32), contributions


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite one-dimensional distribution")
    keys = ["mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "p99",
            "duplicate_fraction", "tied_observation_fraction", "fraction_at_one", "near_zero_variance"]
    if not len(values):
        return {"n": 0, **dict.fromkeys(keys)}
    _, counts = np.unique(values, return_counts=True)
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            **{f"p{p}": float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90, 99)},
            "duplicate_fraction": float(1 - len(counts) / len(values)),
            "tied_observation_fraction": float(counts[counts > 1].sum() / len(values)),
            "fraction_at_one": float(np.mean(values == 1)),
            "near_zero_variance": bool(values.std() <= 1e-8)}


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman correlation with average ranks; constant/empty inputs are undefined."""
    if len(a) < 2 or len(b) != len(a):
        return None

    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def selection_stages(sequences: list[str], scores: np.ndarray, reference: set[str], *,
                     top_k: int, seed: int, shortlist: int) -> dict[str, np.ndarray]:
    """Expose the production funnel using its unchanged filter/FPS primitives."""
    import Levenshtein  # noqa: F401 -- never permit the selector's fallback

    from amp_challenge_2027.select import (
        clean_candidates,
        farthest_point_select,
        filter_novel,
        filter_plausible,
    )

    scores = np.asarray(scores, dtype=np.float32)
    if (clean_candidates(sequences, reference) != sequences or scores.shape != (len(sequences),)
            or not np.isfinite(scores).all() or top_k <= 0 or shortlist <= 0):
        raise ValueError("Expected a clean ordered library and finite aligned scores")
    indices = {seq: i for i, seq in enumerate(sequences)}
    plausible = np.array([indices[seq] for seq in filter_plausible(sequences)], dtype=int)
    short = plausible[np.argsort(-scores[plausible].astype(np.float64), kind="stable")[:shortlist]]
    novel = np.array([indices[seq] for seq in filter_novel([sequences[i] for i in short], reference)], dtype=int)
    if len(novel) < top_k:
        raise ValueError(f"Underfilled selection: {len(novel)} novel candidates for top-{top_k}")
    picked = farthest_point_select([sequences[i] for i in novel], scores[novel], top_k, seed=seed)
    top = novel[picked]
    top = top[np.argsort(-scores[top], kind="stable")]
    return {"library": np.arange(len(sequences)), "plausible": plausible,
            "shortlist": short, "novel_shortlist": novel, "top": top}


def paired_deltas(rows: list[dict], group: str) -> list[dict]:
    """Compare cases within a policy, or policies within a case; never pool cells as seeds."""
    output = []
    for value in sorted({row[group] for row in rows}):
        for seed in sorted({row["seed"] for row in rows}):
            varying, names = ("case", ("hybrid", "p3_s1")) if group == "policy" else ("policy", ("P0", "P1"))
            pair = [row for row in rows if row[group] == value and row["seed"] == seed]
            if len(pair) != 2 or {row[varying] for row in pair} != set(names):
                raise ValueError("Missing or duplicate paired cells")
            a, b = (next(row for row in pair if row[varying] == name) for name in names)
            delta = {group: value, "seed": seed, "comparison": f"{names[1]}_minus_{names[0]}",
                     "top_overlap_count": len(set(a["top_sequences"]) & set(b["top_sequences"]))}
            for name, number in a.items():
                if name in {"seed", "top_size"} or isinstance(number, bool):
                    continue
                if isinstance(number, (float, int)) and isinstance(b.get(name), (float, int)):
                    delta[name] = b[name] - number
            output.append(delta)
    return output
