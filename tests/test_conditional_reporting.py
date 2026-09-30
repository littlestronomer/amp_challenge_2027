import copy

import pytest

from amp_challenge_2027.conditional_reporting import SCOPE, ablation_contrasts


def row(label, arm, seed=42, preset=None, conditions=None, checkpoint="weights", risk=.5):
    return dict(label=label, seed=seed, training_arm=arm, architecture_preset=preset,
                conditions=[] if conditions is None else conditions, checkpoint_sha256=checkpoint,
                raw_activity_mean=.8, raw_risk_mean=risk, raw_panel_mean=.7,
                raw_pairwise_distance_256=.65, joint_unique_yield_per_1000=12.0)


def test_warm_contrasts_use_metadata_and_report_signed_paired_deltas():
    rows = [row(label, arm, seed, risk=risk)
            for label, arm, risk in (("arbitrary0", "frozen", .8), ("arbitrary1", "unconditional", .7),
                                     ("arbitrary2", "selective", .6), ("arbitrary3", "conditional", .5))
            for seed in (42, 43, 44)]
    contrasts = ablation_contrasts(rows)
    assert {item["kind"] for item in contrasts} == {
        "data_and_continued_training", "selective_training", "conditioning_training_at_null_request",
    }
    for item in contrasts:
        expected = -.2 if item["kind"] == "conditioning_training_at_null_request" else -.1
        assert item["seeds"] == [42, 43, 44]
        assert item["mean_deltas"]["raw_risk_mean"] == pytest.approx(expected)
        assert item["metric_pair_counts"]["raw_risk_mean"] == 3
        assert item["scope"] == SCOPE and item["promotion"] is False


def test_request_effect_requires_same_checkpoint_for_every_pair():
    conditions = [{"endpoint": "MIC", "value_upper": 16}]
    rows = [row("null", "conditional", seed, checkpoint=f"weights{seed}") for seed in (42, 43)]
    rows += [row("request", "conditional", seed, checkpoint=weights, conditions=conditions, risk=.2)
             for seed, weights in ((42, "weights42"), (43, "differentweights"))]
    contrast = ablation_contrasts(rows)[0]
    assert contrast["kind"] == "condition_request"
    assert contrast["seeds"] == [42]
    assert contrast["same_checkpoint_required"]
    assert contrast["conditions"] == conditions


def test_architecture_factorial_requires_same_requests_and_training_arms():
    conditions = [{"endpoint": "MIC", "value_upper": 16}]
    rows = [row("unrelated_label" + p, "conditional", preset=p, conditions=conditions) for p in "ABCD"]
    contrasts = ablation_contrasts(rows)
    assert {item["kind"] for item in contrasts} == {
        "corrected_attnres_dense", "moe_standard", "corrected_attnres_moe", "moe_attnres",
    }
    rows[-1]["conditions"] = [{"endpoint": "MIC", "value_upper": 4}]
    assert {item["kind"] for item in ablation_contrasts(rows)} == {
        "corrected_attnres_dense", "moe_standard",
    }
    rows[1]["training_arm"] = "unconditional"
    assert [item["kind"] for item in ablation_contrasts(rows)] == ["moe_standard"]


def test_missing_metadata_and_ambiguous_variants_are_skipped():
    assert ablation_contrasts([{"label": "frozen", "seed": 42}, {"label": "conditional", "seed": 42}]) == []
    rows = [row("base", "frozen"), row("candidate", "unconditional")]
    assert len(ablation_contrasts(rows)) == 1
    rows.append(row("another_candidate", "unconditional", seed=43))
    assert ablation_contrasts(rows) == []


def test_missing_metric_does_not_become_zero_and_inputs_are_unchanged():
    rows = [row("base", "frozen", seed=seed) for seed in (42, 43)]
    rows += [row("candidate", "unconditional", seed=seed, risk=.3) for seed in (42, 43)]
    rows[-1]["raw_risk_mean"] = None
    before = copy.deepcopy(rows)
    item = ablation_contrasts(rows)[0]
    assert item["mean_deltas"]["raw_risk_mean"] == pytest.approx(-.2)
    assert item["metric_pair_counts"]["raw_risk_mean"] == 1
    assert "raw_risk_mean" not in item["per_seed"][1]["deltas"]
    assert rows == before
