"""Shared candidate pipeline: pool loading, cleaning, composite scoring.

One place owns the steps shared by ``generate.py``, ``scripts/blend_libraries.py``,
and ``scripts/sweep_selection.py`` so the entry point, offline blending, and the
sweep harness cannot drift apart:

  1. ``load_pool_fastas``   — read raw candidate FASTAs (one per generator
     checkpoint) into a single deterministic pool with optional per-source caps.
  2. ``clean_candidates``    — validity + dedup + no-overlap (the cheap filters).
  3. ``build_composite_scorer`` — wire the activity / conformity / precision
     components from ``score.py`` that are actually available in this
     environment; weights renormalize over surviving components.
  4. ``score_candidates``    — combined score vector aligned with input order.

Everything is deterministic given (inputs, seed); heavy deps stay optional.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amp_challenge_2027.config import ESM2_MODEL
from amp_challenge_2027.data import iter_fasta
from amp_challenge_2027.score import (
    ActivityScorer,
    CompositeScorer,
    ConformityScorer,
    PanelScorer,
    PrecisionProxyScorer,
)
from amp_challenge_2027.select import clean_candidates as _clean

# Submission selection recipe (adopted 2026-08-29 from the panel-weight sweep,
# sweep_results/panel-weights, mix m1): breadth/mdr dominate the top-100
# composition while old-instrument activity stays within ~0.1 of the legacy
# anchor. Components missing from the environment drop and weights
# renormalize, so this degrades gracefully on the minimal validator env.
DEFAULT_WEIGHTS = {
    "activity": 0.5,
    "conformity": 0.25,
    "precision": 0.25,
    "breadth": 1.0,
    "mdr": 1.0,
}


def load_pool_fastas(
    paths: list[Path | str], *, cap_per_source: int = 0, seed: int = 42
) -> list[str]:
    """Concatenate candidate FASTAs in CLI order into one raw pool.

    Each source is deduplicated internally after a seeded shuffle (so caps take
    a random head, not the first N lines), then truncated to
    ``cap_per_source`` (0 = unlimited). Cross-source duplicates are NOT removed
    here — ``clean_candidates`` handles that downstream.
    """
    pool: list[str] = []
    for i, path in enumerate(paths):
        path = Path(path)
        seqs = [seq for _, seq in iter_fasta(path)]
        rng = np.random.default_rng(seed + i)
        rng.shuffle(seqs)
        if cap_per_source and cap_per_source > 0:
            seqs = seqs[:cap_per_source]
        print(f"[pipeline] pool {path.name}: {len(seqs)} candidates")
        pool.extend(seqs)
    return pool


def clean_candidates(sequences: list[str], reference_set: set[str]) -> list[str]:
    """Validity + dedup + exact-overlap removal (order-preserving)."""
    return _clean(sequences, reference_set)


def _torch_ready() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def build_composite_scorer(
    reference_seqs: list[str],
    *,
    w_activity: float = DEFAULT_WEIGHTS["activity"],
    w_conformity: float = DEFAULT_WEIGHTS["conformity"],
    w_precision: float = DEFAULT_WEIGHTS["precision"],
    w_breadth: float = DEFAULT_WEIGHTS["breadth"],
    w_mdr: float = DEFAULT_WEIGHTS["mdr"],
    device: str = "cpu",
    conformity_sample: int = 12000,
    precision_esm_model: str = ESM2_MODEL,
    seed: int = 42,
) -> CompositeScorer | None:
    """Assemble every requested component that this environment can run.

    - conformity needs only numpy + a reference set.
    - activity loads ``checkpoint/reward/classifier.pt`` (None → dropped).
    - breadth/mdr load ``checkpoint/reward/classifier_panel.pt`` (None → both
      dropped); both weights share ONE PanelScorer forward pass per candidate.
    - precision needs torch + transformers (dropped otherwise).

    Returns None only when NO component survives, in which case callers fall
    back to unscored (pure-diversity) selection.
    """
    components = []
    if w_conformity and reference_seqs:
        components.append(
            (
                "conformity",
                float(w_conformity),
                ConformityScorer(reference_seqs, sample=conformity_sample, seed=seed).score,
            )
        )
    if w_activity:
        activity = ActivityScorer.load(device=device)
        if activity is not None:
            components.append(("activity", float(w_activity), activity.score))
        else:
            print("[pipeline] no checkpoint/reward/classifier.pt; activity component dropped")
    if w_breadth or w_mdr:
        panel = PanelScorer.load(device=device)
        if panel is not None:
            if w_breadth:
                components.append(("breadth", float(w_breadth), panel.breadth))
            if w_mdr:
                components.append(("mdr", float(w_mdr), panel.mdr_breadth))
        else:
            print(
                "[pipeline] no checkpoint/reward/classifier_panel.pt; "
                "breadth/mdr components dropped"
            )
    if w_precision and reference_seqs:
        if _torch_ready():
            precision = PrecisionProxyScorer(
                reference_seqs, esm_model=precision_esm_model, device=device
            )
            components.append(("precision", float(w_precision), precision.score))
        else:
            print("[pipeline] torch/transformers unavailable; precision component dropped")
    if not components:
        return None
    return CompositeScorer([(n, w, c) for n, w, c in components])


def score_candidates(
    scorer: CompositeScorer, sequences: list[str]
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Combined score for ``sequences``; prints per-component summary stats.

    Returns ``(combined, parts)``; ``combined`` is None when ``scorer`` is None.
    """
    if scorer is None or not sequences:
        return None, {}
    combined, parts = scorer.score(sequences)
    stats = ", ".join(
        f"{name}: μ={vals.mean():.3f} σ={vals.std():.3f} [{vals.min():.3f},{vals.max():.3f}]"
        for name, vals in parts.items()
    )
    print(f"[pipeline] scored {len(sequences)} candidates ({stats})")
    return combined, parts


__all__ = [
    "DEFAULT_WEIGHTS",
    "load_pool_fastas",
    "clean_candidates",
    "build_composite_scorer",
    "score_candidates",
]
