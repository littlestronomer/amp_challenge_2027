import numpy as np
import pytest

from amp_challenge_2027.selectivity import (
    LAMBDA,
    adjusted_score,
    criteria_for,
    paired_policy_deltas,
    sequence_digest,
    validate_protocol,
    validate_risk,
)


def _protocol():
    import json
    from pathlib import Path
    return json.loads(Path("experiments/selectivity_tradeoff_v1.json").read_text())


def test_frozen_protocol_rejects_changed_weight_or_criteria():
    protocol = _protocol()
    validate_protocol(protocol)
    protocol["risk_penalty"] = 0.5
    with pytest.raises(ValueError, match="frozen"):
        validate_protocol(protocol)


def test_ordered_sequence_digest_and_risk_validation():
    assert sequence_digest(["ACD", "EFG"]) != sequence_digest(["EFG", "ACD"])
    np.testing.assert_array_equal(validate_risk([0.0, 1.0], 2), np.array([0.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_risk([0.2, np.nan], 2)
    with pytest.raises(ValueError):
        validate_risk([1.1], 1)


def test_risk_penalty_direction_lambda_zero_and_frozen_anchor():
    base = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    anchor = {"mean": 0.5, "denominator": 0.25}
    risk = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    adjusted = adjusted_score(base, risk, anchor)
    assert adjusted[0] > adjusted[1] > adjusted[2]
    exact = adjusted_score(base, risk, anchor, weight=0)
    np.testing.assert_array_equal(exact, base)
    seed43 = adjusted_score(base, np.array([0.25, 0.5, 0.75]), anchor)
    assert seed43[0] > seed43[2]  # same seed-42 anchor applies unchanged
    assert LAMBDA == 0.25


def test_paired_criteria_are_per_seed_and_include_genus_probabilities():
    rows = []
    for seed in (42, 43, 44):
        for policy, risk, panel in (("C0", 0.7, 0.8), ("R1", 0.6, 0.79)):
            rows.append({"policy": policy, "seed": seed, "hemo_risk_mean": risk,
                         "hemo_risk_p75": risk, "activity_mean": 0.8,
                         "panel_mean_probability": panel, "breadth_mean": 0.5,
                         "mdr_mean": 0.3, "top_mean_pairwise_distance": 0.4,
                         "p_active:genus_mean": panel, "top_sequences": []})
    deltas = paired_policy_deltas(rows)
    result = criteria_for(deltas)
    assert result["all_seeds_pass"]
    rows[-1]["hemo_risk_mean"] = 0.7
    assert not criteria_for(paired_policy_deltas(rows))["all_seeds_pass"]
