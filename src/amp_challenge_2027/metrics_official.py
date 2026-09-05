"""Official Phase-1 protocol metrics via seqme, built for reuse.

``build_metric_list`` constructs the four competition metric families
(sequence-level, distributional, property conformity, authenticity) exactly as
described in the proposal §1.5. It is shared by ``scripts/eval_official.py``
(one-shot CLI scoring) and ``scripts/sweep_selection.py`` (many cells, one
shared embedder), so both can never disagree about protocol details.

seqme is optional: import this module without it; only ``build_metric_list``
needs it at call time. With ``cheap=True`` only dependency-free sequence-level
metrics are returned (no embedder required), which keeps local smoke tests and
CPU-only sanity runs fast.
"""

from __future__ import annotations

from typing import Any


def build_metric_list(
    reference: list[str],
    embedder: Any | None = None,
    *,
    cheap: bool = False,
    strict: bool = False,
) -> list[tuple[str, Any]]:
    """Build (name, metric) pairs replicating the Phase-1 protocol.

    Requires ``embedder`` unless ``cheap=True``. Metrics whose constructor is
    unavailable in the installed seqme version are skipped with a note rather
    than failing the whole evaluation, unless ``strict=True`` requires the
    complete local comparison protocol.
    """
    import seqme as sm

    metrics: list[tuple[str, Any]] = []

    # Family 1: Sequence-level (dependency-free)
    metrics.append(("Uniqueness", sm.metrics.Uniqueness()))
    metrics.append(("Novelty", sm.metrics.Novelty(reference=reference)))
    metrics.append(("Diversity", sm.metrics.Diversity()))
    metrics.append(("Length", sm.metrics.Length()))
    try:
        metrics.append(
            ("NGramJaccard", sm.metrics.NGramJaccardSimilarity(reference=reference, n=3))
        )
    except Exception as e:
        print(f"[metrics] skip NGramJaccard: {e}")

    if cheap:
        return metrics

    assert embedder is not None, "embedder required for non-cheap metric families"

    # Property predictors for ConformityScore (modlamp-backed).
    predictors = []
    for name, cls in [
        ("charge", sm.models.Charge),
        ("hydrophobicity", sm.models.Hydrophobicity),
        ("hydrophobic_moment", sm.models.HydrophobicMoment),
    ]:
        try:
            predictors.append(cls())
        except Exception as e:
            print(f"[metrics] skip predictor {name}: {e}")

    # Family 2: Distributional (embedding-space)
    try:
        metrics.append(("FBD", sm.metrics.FBD(reference=reference, embedder=embedder)))
    except Exception as e:
        print(f"[metrics] skip FBD: {e}")
    try:
        metrics.append(("MMD", sm.metrics.MMD(reference=reference, embedder=embedder)))
    except Exception as e:
        print(f"[metrics] skip MMD: {e}")
    # FKEA: the seqme benchmark's extra embedding-space distribution metric
    # (organizers' own tutorial includes it). Informational until the official
    # suite is known; identical constructor args to the tutorial.
    try:
        metrics.append(("FKEA", sm.metrics.FKEA(embedder=embedder, bandwidth=1.0, strict=False)))
    except Exception as e:
        print(f"[metrics] skip FKEA: {e}")
    # Precision and Recall are separate classes in seqme.
    for cls_name in ("Precision", "Recall"):
        cls = getattr(sm.metrics, cls_name, None)
        if cls is not None:
            try:
                metrics.append(
                    (
                        cls_name,
                        cls(n_neighbors=5, reference=reference, embedder=embedder, strict=False),
                    )
                )
            except Exception as e:
                print(f"[metrics] skip {cls_name}: {e}")

    # Family 3: Property conformity
    if predictors:
        try:
            metrics.append(
                (
                    "ConformityScore",
                    sm.metrics.ConformityScore(reference=reference, predictors=predictors),
                )
            )
        except Exception as e:
            print(f"[metrics] skip ConformityScore: {e}")
    # Second conformity readout matching the seqme tutorial's predictor set
    # ([amphiphilicity, charge]) — the organizers' own default configuration.
    # Informational twin of the 3-predictor score above.
    try:
        tutorial_predictors = [sm.models.Amphiphilicity(), sm.models.Charge()]
        metrics.append(
            (
                "ConformityScore(amp+charge)",
                sm.metrics.ConformityScore(
                    reference=reference, predictors=tutorial_predictors,
                    name="Conformity score (amp+charge)",
                ),
            )
        )
    except Exception as e:
        print(f"[metrics] skip tutorial conformity: {e}")

    # Family 4: Authenticity
    try:
        metrics.append(("AuthPct", sm.metrics.AuthPct(train_set=reference, embedder=embedder)))
    except Exception as e:
        print(f"[metrics] skip AuthPct: {e}")

    if strict:
        required = {
            "Uniqueness", "Novelty", "Diversity", "Length", "NGramJaccard",
            "FBD", "MMD", "FKEA", "Precision", "Recall", "ConformityScore",
            "ConformityScore(amp+charge)", "AuthPct",
        }
        missing = required - {name for name, _ in metrics}
        if missing or len(predictors) != 3:
            raise RuntimeError(
                f"Incomplete evaluation: missing {sorted(missing)}, "
                f"property predictors {len(predictors)}/3"
            )
    return metrics


__all__ = ["build_metric_list"]
