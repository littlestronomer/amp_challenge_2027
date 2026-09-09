from types import SimpleNamespace

import numpy as np
import pytest
from generator_improvement import raw_metrics, teaching_ids


def test_raw_yield_counts_repeats_failures_and_reference_in_denominator():
    sequences = ["AAAAAAAA", "AAAAAAAA", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE"]
    result = raw_metrics(sequences, [.9, .9, .9, .9, .1], [.1]*5,
                         [.1, .1, .1, .8, .1], {"CCCCCCCC"})
    assert result["raw_reward_yield_per_1000"] == 400
    assert result["raw_evaluation_yield_per_1000"] == 200
    assert result["raw_joint_yield_per_1000"] == 200
    assert result["raw_unique_novel_fraction"] == 3/5
    with pytest.raises(ValueError):
        raw_metrics(sequences, [np.nan]*5, [.1]*5, [.1]*5, set())


def test_teaching_no_relaxation_near_duplicate_or_reference():
    seqs = ["AAAAAAAA", "AAAAAAAC", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE"]
    ids = teaching_ids(seqs, [.9]*5, [.1, .2, .1, .8, .1], {"EEEEEEEE"}, 5)
    assert ids == [0, 2]
    assert teaching_ids(seqs, [.1]*5, [.1]*5, set(), 5) == []


def test_raft_kl_guard_rolls_back_and_stops(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    import generator_improvement as improvement
    from pilot_grpo import strict_runtime

    from amp_challenge_2027.model import DecoderConfig, build_model

    torch.set_num_threads(1)
    strict_runtime(42)
    model, _ = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2))
    model.eval()
    frozen = copy.deepcopy(model).requires_grad_(False).eval()
    original_kl = improvement.prefix_kl
    calls = 0

    def injected_post_update_kl(*args):
        nonlocal calls
        calls += 1
        return original_kl(*args) + (1. if calls > 2 else 0.)

    monkeypatch.setattr(improvement, "prefix_kl", injected_post_update_kl)
    args = SimpleNamespace(seed=42, device="cpu", raft_replay=4, lr=1e-6, out=tmp_path,
                           steps=2, raft_draws=8, raft_retain=4, raft_min_retain=1,
                           activity_floor=.6, risk_ceiling=.5, raft_epochs=1, kl_limit=.05)
    _, draws, stopped, status = improvement.train_raft(model, frozen,
        lambda seqs: (np.full(len(seqs), .9), np.full(len(seqs), .1)), set(), args)
    assert stopped and draws == 8 and status["updates"] == 0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), frozen.parameters(), strict=True))
