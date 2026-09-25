"""Descriptive ablations inferred from recorded experiment metadata, not labels."""

from __future__ import annotations

import json
import math
from collections import defaultdict

METRICS = (
    "raw_activity_mean", "raw_risk_mean", "raw_panel_mean",
    "raw_pairwise_distance_256", "joint_unique_yield_per_1000",
)
SCOPE = "surrogate_not_biological_or_unconfounded_causal_evidence"


def ablation_contrasts(rows: list[dict]) -> list[dict]:
    """Compare unambiguous variants on shared seeds; delta is candidate minus base.

    Missing metadata never implies an unconditional control or a warm start.
    Architecture contrasts additionally require identical requests and arms.
    Multiple labels with the same experimental signature are ambiguous and are
    skipped. A request contrast must use the very same checkpoint for each pair.
    """
    groups = defaultdict(list)
    required = {"label", "seed", "training_arm", "architecture_preset", "conditions"}
    for row in rows:
        if not required <= row.keys() or not isinstance(row["conditions"], list):
            continue
        arm, preset = row["training_arm"], row["architecture_preset"]
        if arm not in {"frozen", "unconditional", "selective", "conditional"}:
            continue
        if preset not in {None, "A", "B", "C", "D"} or not row["label"]:
            continue
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            continue
        request = json.dumps(row["conditions"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        groups[(arm, preset, request)].append(row)
    groups = {
        key: {row["seed"]: row for row in values}
        for key, values in groups.items()
        if len({row["label"] for row in values}) == 1
        and len({row["seed"] for row in values}) == len(values)
    }
    results = []

    def add(kind, baseline_key, candidate_key, *, same_checkpoint=False):
        baseline, candidate = groups.get(baseline_key, {}), groups.get(candidate_key, {})
        pairs = []
        for seed in sorted(set(baseline) & set(candidate)):
            a, b = baseline[seed], candidate[seed]
            if same_checkpoint and (
                not a.get("checkpoint_sha256")
                or a["checkpoint_sha256"] != b.get("checkpoint_sha256")
            ):
                continue
            deltas = {}
            for metric in METRICS:
                left, right = a.get(metric), b.get(metric)
                if (isinstance(left, (float, int)) and not isinstance(left, bool)
                        and isinstance(right, (float, int)) and not isinstance(right, bool)
                        and math.isfinite(left) and math.isfinite(right)):
                    deltas[metric] = float(right - left)
            if deltas:
                pairs.append({"seed": seed, "deltas": deltas})
        if not pairs:
            return
        means, counts = {}, {}
        for metric in METRICS:
            values = [pair["deltas"][metric] for pair in pairs if metric in pair["deltas"]]
            if values:
                means[metric], counts[metric] = math.fsum(values) / len(values), len(values)
        results.append({
            "kind": kind, "baseline_label": next(iter(baseline.values()))["label"],
            "candidate_label": next(iter(candidate.values()))["label"],
            "seeds": [pair["seed"] for pair in pairs], "per_seed": pairs,
            "mean_deltas": means, "metric_pair_counts": counts,
            "delta_direction": "candidate_minus_baseline", "same_checkpoint_required": same_checkpoint,
            "conditions": json.loads(candidate_key[2]), "scope": SCOPE,
            "promotion": False,
        })

    for kind, baseline_arm, candidate_arm in (
        ("data_and_continued_training", "frozen", "unconditional"),
        ("selective_training", "unconditional", "selective"),
        ("conditioning_training_at_null_request", "unconditional", "conditional"),
    ):
        add(kind, (baseline_arm, None, "[]"), (candidate_arm, None, "[]"))
    for arm, preset, request in sorted(groups, key=repr):
        if arm == "conditional" and request != "[]":
            add("condition_request", (arm, preset, "[]"), (arm, preset, request), same_checkpoint=True)
    architecture_pairs = (
        ("corrected_attnres_dense", "A", "B"),
        ("moe_standard", "A", "C"),
        ("corrected_attnres_moe", "C", "D"),
        ("moe_attnres", "B", "D"),
    )
    for arm, request in sorted({(arm, request) for arm, preset, request in groups if preset is not None}):
        for kind, baseline_preset, candidate_preset in architecture_pairs:
            add(kind, (arm, baseline_preset, request), (arm, candidate_preset, request))
    return results
