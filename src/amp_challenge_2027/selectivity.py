"""Pure helpers for the frozen-pool hemolysis sensitivity study."""

from __future__ import annotations

import hashlib

import numpy as np

LAMBDA = 0.25
FROZEN_CRITERIA = {
    "every_seed_risk_mean_decrease_min": 0.05,
    "every_seed_risk_p75_increase_max": 0.0,
    "activity_mean_drop_max": 0.03,
    "panel_mean_probability_drop_max": 0.03,
    "each_genus_probability_mean_drop_max": 0.03,
    "breadth_mean_drop_max": 0.02,
    "mdr_mean_drop_max": 0.02,
    "pairwise_distance_mean_drop_max": 0.02,
}


def validate_protocol(protocol: dict) -> None:
    expected = {
        "kind": "selectivity_tradeoff_v1", "source": "sweep_results/epoch58-top100-v1",
        "source_case": "hybrid", "generation_seeds": [42, 43, 44], "ranking_seed": 42,
        "top_size": 100, "shortlist": 2000, "novelty_threshold": 0.8,
        "policies": ["C0", "R1"], "risk_penalty": LAMBDA,
        "novelty_comparison": "inclusive; reject only values greater than threshold",
        "risk_score": "production HemoScorer.p_risky; positive means risky under candidate label definition",
        "risk_normalization": "mean and population standard deviation over all hybrid seed-42 library risk predictions; freeze for seeds 43 and 44",
        "score_definition": "S1=float32(float64(S0)-0.25*(float64(p_risky)-anchor_mean)/anchor_sd); S0 is unchanged frozen five-component combined score",
        "decision": "passing is candidate for independent evaluation only; no automatic promotion",
        "limits": [
            "This is a surrogate sensitivity analysis, not a safety claim or assay result.",
            "The three generation seeds are descriptive and are not independent model replicates.",
            "Do not tune the weight or expand the search after viewing this comparison.",
        ],
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Selectivity protocol differs from the frozen source/selection contract")
    if protocol.get("criteria") != FROZEN_CRITERIA:
        raise ValueError("Predeclared selectivity criteria changed")


def sequence_digest(sequences: list[str]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_risk(values, size: int, *, allow_missing: bool = False) -> np.ndarray:
    risk = np.asarray(values, dtype=np.float32)
    if risk.shape != (size,):
        raise ValueError(f"Risk array must have shape ({size},)")
    finite = np.isfinite(risk)
    if (not allow_missing and not finite.all()) or (risk[finite] < 0).any() or (risk[finite] > 1).any():
        raise ValueError("Risk values must be finite probabilities in [0, 1]")
    return risk


def adjusted_score(base, risk, anchor: dict, *, weight: float = LAMBDA) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    risk = validate_risk(risk, len(base))
    if not np.isfinite(base).all() or not np.isfinite(weight) or weight < 0:
        raise ValueError("Invalid base score or risk penalty")
    if weight == 0:
        return base.copy()
    mean, denominator = float(anchor["mean"]), float(anchor["denominator"])
    if not np.isfinite([mean, denominator]).all() or denominator <= 0:
        raise ValueError("Invalid frozen risk normalization")
    return (base.astype(np.float64) - weight * (risk.astype(np.float64) - mean) / denominator).astype(np.float32)


def paired_policy_deltas(rows: list[dict]) -> list[dict]:
    result = []
    for seed in (42, 43, 44):
        pair = [row for row in rows if row["seed"] == seed]
        if len(pair) != 2 or {row["policy"] for row in pair} != {"C0", "R1"}:
            raise ValueError(f"Expected one C0/R1 pair for seed {seed}")
        incumbent = next(row for row in pair if row["policy"] == "C0")
        risk_aware = next(row for row in pair if row["policy"] == "R1")
        delta = {"seed": seed, "comparison": "R1_minus_C0",
                 "top_overlap_count": len(set(incumbent["top_sequences"]) & set(risk_aware["top_sequences"]))}
        same_membership = set(incumbent["top_sequences"]) == set(risk_aware["top_sequences"])
        rank_changes = sum(a != b for a, b in zip(incumbent["top_sequences"], risk_aware["top_sequences"]))
        delta.update(membership_changed_count=len(incumbent["top_sequences"]) - delta["top_overlap_count"],
                     same_membership=same_membership, rank_order_changed_positions=rank_changes,
                     order_only_rank_changes=rank_changes if same_membership else None)
        for key, value in incumbent.items():
            if key in {"seed", "policy", "top_sequences", "top_size"} or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and isinstance(risk_aware.get(key), (int, float)):
                delta[key] = risk_aware[key] - value
        result.append(delta)
    return result


def criteria_for(deltas: list[dict]) -> dict:
    checks = {}
    tolerances = {"activity_mean": FROZEN_CRITERIA["activity_mean_drop_max"],
                  "panel_mean_probability": FROZEN_CRITERIA["panel_mean_probability_drop_max"],
                  "breadth_mean": FROZEN_CRITERIA["breadth_mean_drop_max"],
                  "mdr_mean": FROZEN_CRITERIA["mdr_mean_drop_max"],
                  "top_mean_pairwise_distance": FROZEN_CRITERIA["pairwise_distance_mean_drop_max"]}
    for seed_row in deltas:
        seed = str(seed_row["seed"])
        genus_checks = {f"{key}_drop_within_{FROZEN_CRITERIA['each_genus_probability_mean_drop_max']}":
                        value >= -FROZEN_CRITERIA["each_genus_probability_mean_drop_max"]
                        for key, value in seed_row.items() if key.startswith("p_active:")}
        checks[seed] = {
            "risk_mean_decrease_at_least_0_05": seed_row.get("hemo_risk_mean", 0) <= -FROZEN_CRITERIA["every_seed_risk_mean_decrease_min"],
            "risk_p75_not_increased": seed_row.get("hemo_risk_p75", 0) <= FROZEN_CRITERIA["every_seed_risk_p75_increase_max"],
            **{f"{name}_drop_within_{limit}": seed_row.get(name, 0) >= -limit
               for name, limit in tolerances.items()},
            **genus_checks,
        }
    passed = bool(checks) and all(all(seed.values()) for seed in checks.values())
    return {"kind": "selectivity_exploratory_criteria_v1", "checks_by_seed": checks,
            "all_seeds_pass": passed,
            "interpretation": "candidate_for_independent_evaluation" if passed else "retain_incumbent"}
