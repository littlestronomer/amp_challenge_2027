import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiment_utils import mark_files, sha256, write_json
from test_scale_validation import pilot_sources as _pilot_sources

pilot_sources = _pilot_sources


def test_forward_kl_direction_padding_and_teacher_detachment():
    torch = pytest.importorskip("torch")
    from amp_challenge_2027.opd import forward_kl

    teacher = torch.tensor([[[.8, .2], [.5, .5]]]).log().requires_grad_()
    logits = torch.tensor([[[0., 0.], [8., -8.]]], requires_grad=True)
    tokens = torch.tensor([[1, 4, 0]])
    loss = forward_kl(teacher, logits.log_softmax(-1), tokens)
    expected = .8*np.log(.8/.5)+.2*np.log(.2/.5)
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-7)
    loss.backward()
    assert teacher.grad is None
    assert torch.all(logits.grad[:, 1] == 0)
    assert logits.grad[0, 0, 0] < 0


def test_coverage_gradient_matches_exact_categorical_objective():
    torch = pytest.importorskip("torch")
    from amp_challenge_2027.opd import coverage_cost, coverage_loss

    logits = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(0)
    phi = np.array([[0., .2], [1., .7]])
    target = np.array([.3, .4])
    exact = ((probabilities @ torch.tensor(phi)-torch.tensor(target))**2).sum()
    expected = torch.autograd.grad(exact, logits)[0]
    estimate = torch.zeros(2, dtype=torch.float64)
    for a in range(2):
        for b in range(2):
            costs, _ = coverage_cost(phi[[a, b]], target)
            # Toy actions 0,1 shifted to4,5 to avoid PAD; no length normalization.
            tokens = torch.tensor([[1, a+4], [1, b+4]])
            lp = torch.cat((torch.zeros(4, dtype=torch.float64), logits.log_softmax(0)))
            loss = coverage_loss(lp.expand(2, 1, 6), tokens, torch.tensor(costs))
            grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            estimate += (probabilities[a]*probabilities[b]).detach()*grad
    assert torch.allclose(estimate, expected, atol=1e-7)
    with pytest.raises(ValueError):
        coverage_cost([[float("nan")]], [0.])


class FakeEncoder:
    def encode(self, sequences, batch_size):
        # Stable sequence-dependent embeddings, no weights or network needed.
        return np.array([[len(s), sum(map(ord, s)) % 37, *([1.]*478)] for s in sequences])


def test_coverage_features_fixed_and_finite():
    from amp_challenge_2027.opd import CoverageFeatures

    sequences = ["AAAAAAAA", "CCCCCCCCCC", "ACDEFGHIKLMNPQ"]
    first = CoverageFeatures(FakeEncoder(), sequences, 42)
    second = CoverageFeatures(FakeEncoder(), sequences, 42)
    np.testing.assert_array_equal(first.target, second.target)
    assert first.transform(sequences).shape == (3, 149)
    assert np.isfinite(first.target).all()


@pytest.mark.parametrize("variant", ["specialist", "anchored", "coverage"])
def test_optimize_frozen_teachers_repeatable_and_no_reward_calls(tmp_path, variant):
    torch = pytest.importorskip("torch")
    from pilot_grpo import strict_runtime
    from train_opd import optimize

    from amp_challenge_2027.model import DecoderConfig, build_model

    torch.set_num_threads(1)
    strict_runtime(42)
    initial, _ = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2))
    teacher = copy.deepcopy(initial)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(.01*torch.randn_like(p))
    baseline = copy.deepcopy(initial)
    frozen = {k: v.clone() for k, v in teacher.state_dict().items()}
    results = []
    for name in ("first", "repeat"):
        out = tmp_path / name
        out.mkdir()
        args = SimpleNamespace(out=out, variant=variant, seed=42, device="cpu", steps=2, batch_size=2,
                               lr=1e-5, anchor_weight=1., kl_limit=.05, coverage_draws=32, dual_lr=.5, coverage_target=.01)
        student = copy.deepcopy(initial)
        status = optimize(student, teacher, baseline, args, encoder=FakeEncoder())
        assert status["accepted_updates"] == 2
        assert status["student_draws"] == 4
        assert status["total_training_draws"] == 4+32+(4 if variant != "specialist" else 0)+(32 if variant == "coverage" else 0)
        assert status["reward_head_calls_during_training"] == status["evaluation_head_calls_during_training"] == 0
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        assert all(p.grad is None and not p.requires_grad for p in baseline.parameters())
        assert all(torch.equal(v, frozen[k]) for k, v in teacher.state_dict().items())
        assert all(torch.equal(v, initial.state_dict()[k]) for k, v in baseline.state_dict().items())
        assert any(not torch.equal(v, initial.state_dict()[k]) for k, v in student.state_dict().items())
        results.append((student.state_dict(), (out / "history.csv").read_bytes(), (out / "training_samples.csv").read_bytes()))
    assert results[0][1:] == results[1][1:]
    assert all(torch.equal(v, results[1][0][k]) for k, v in results[0][0].items())


def test_kl_stop_rolls_back(tmp_path):
    torch = pytest.importorskip("torch")
    from pilot_grpo import strict_runtime
    from train_opd import optimize

    from amp_challenge_2027.model import DecoderConfig, build_model

    strict_runtime(42)
    initial, _ = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2))
    teacher = copy.deepcopy(initial)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(torch.randn_like(p))
    student = copy.deepcopy(initial)
    args = SimpleNamespace(out=tmp_path, variant="specialist", seed=42, device="cpu", steps=2, batch_size=2,
                           lr=.1, kl_limit=1e-10)
    status = optimize(student, teacher, copy.deepcopy(initial), args)
    assert status["kl_stopped"] and status["accepted_updates"] == 0
    assert all(torch.equal(v, initial.state_dict()[k]) for k, v in student.state_dict().items())


def test_preflight_is_read_only_and_protects_sources(pilot_sources, capsys):
    pytest.importorskip("torch")
    from train_opd import main

    out = pilot_sources / "opd-specialist-seed42-v1"
    args = ["--pilot-root", str(pilot_sources), "--variant", "specialist", "--out", str(out), "--list"]
    main(args)
    assert not out.exists()
    assert "anchored_opd_v1" in capsys.readouterr().out
    with pytest.raises(ValueError, match="separate"):
        main([*args, "--out", str(pilot_sources / "initial/nested")])
    with pytest.raises(ValueError, match="Invalid"):
        main([*args, "--lr", "nan"])


def test_legacy_configuration_defaults_and_metadata(tmp_path):
    pytest.importorskip("torch")
    from train_opd import configuration_compatibility

    from amp_challenge_2027.model import DecoderConfig

    old = DecoderConfig(residual="block_attnres").to_dict()
    for key in ("conditioning", "num_charge_bins", "charge_min"):
        old.pop(key)
    old["training_note"] = "ignored by model loader"
    teacher = DecoderConfig.from_dict(old).to_dict()
    baseline_path, teacher_path = tmp_path / "baseline.json", tmp_path / "teacher.json"
    write_json(baseline_path, old)
    write_json(teacher_path, teacher)
    result = configuration_compatibility(baseline_path, teacher_path)
    assert result["effective_config"] == teacher
    assert set(result["raw_differences"]) == {"conditioning", "num_charge_bins", "charge_min", "training_note"}


@pytest.mark.parametrize("field,value", [("hidden_size", 768), ("num_heads", 3), ("conditioning", "charge"),
                                        ("eos_token_id", 3), ("residual", "attnres"), ("charge_min", -9)])
def test_effective_configuration_mismatch_is_not_bypassed(tmp_path, field, value):
    pytest.importorskip("torch")
    from train_opd import configuration_compatibility

    baseline, teacher = tmp_path / "baseline.json", tmp_path / "teacher.json"
    write_json(baseline, {})
    write_json(teacher, {field: value})
    with pytest.raises(ValueError, match=field):
        configuration_compatibility(baseline, teacher)


def test_cli_accepts_resaved_legacy_teacher_without_writing(pilot_sources, capsys):
    pytest.importorskip("torch")
    from train_opd import main

    from amp_challenge_2027.model import DecoderConfig

    teacher = pilot_sources / "grpo-generator-seed42-v1"
    write_json(teacher / "policy/config.json", DecoderConfig().to_dict())
    proof = json.loads((teacher / "complete.json").read_text())
    mark_files(teacher, "complete.json", list(proof["files"]))
    out = pilot_sources / "opd-specialist-seed42-v1"
    main(["--pilot-root", str(pilot_sources), "--out", str(out), "--variant", "specialist", "--list"])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["configuration_compatibility"]["effective_config"]["conditioning"] == "none"
    assert not out.exists()


def test_opd_source_inventory_rejects_changed_settings(pilot_sources, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("torch")
    import evaluate_opd
    from train_opd import main

    # Build CLI-realistic manifests without loading dummy model weights.
    for seed in (42, 43, 44):
        for variant in evaluate_opd.VARIANTS:
            path = pilot_sources / f"opd-{variant}-seed{seed}-v1"
            captured = []
            monkeypatch.setattr("builtins.print", lambda value: captured.append(value))
            main(["--pilot-root", str(pilot_sources), "--out", str(path), "--variant", variant, "--seed", str(seed), "--list"])
            run = json.loads(captured[0])
            run["runtime"] = {"device": "cpu"}
            (path / "policy").mkdir(parents=True)
            (path / "policy/model.pt").write_bytes(b"student")
            write_json(path / "policy/config.json", {})
            write_json(path / "run.json", run)
            write_json(path / "status.json", {"kl_stopped": False})
            mark_files(path, "complete.json", ["run.json", "status.json", "policy/model.pt", "policy/config.json"])
    pinned = evaluate_opd.inputs(pilot_sources)
    assert len(pinned["cells"]) == 15
    assert pinned["cells"]["teacher/seed44"]["sampling_seed"] == 60044
    path = pilot_sources / "opd-coverage-seed44-v1"
    run = json.loads((path / "run.json").read_text())
    run["args"]["lr"] = 2e-6
    write_json(path / "run.json", run)
    mark_files(path, "complete.json", ["run.json", "status.json", "policy/model.pt", "policy/config.json"])
    with pytest.raises(ValueError, match="settings"):
        evaluate_opd.inputs(pilot_sources)


def test_fresh_sampling_audit_report_and_tamper_check(tmp_path):
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    from evaluate_opd import METHODS, audit, report
    from pilot_grpo import strict_runtime
    from validate_generator_scale import generate_cell

    from amp_challenge_2027.model import DecoderConfig, build_model, save_model

    torch.set_num_threads(1)
    strict_runtime(42)
    model, cfg = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2))
    checkpoint = tmp_path / "checkpoint"
    save_model(model, checkpoint, config=cfg)
    reference = tmp_path / "ref.fasta"
    reference.write_text(">ref\nAAAAAAAA\n")
    cells = {f"{method}/seed42": {"checkpoint": str(checkpoint), "sampling_seed": 60042} for method in METHODS}
    recipe = {"draws": 32, "scales": [32], "device": "cpu", "sources": {
        "reference": str(reference), "cells": cells, "training_status": {},
        "common": {"inputs": {str(reference): sha256(reference)}}}}

    class FakeScorers:
        def score(self, sequences):
            return np.tile([.9, .1, .2], (len(sequences), 1))

    root = tmp_path / "evaluation"
    for cell in cells:
        generate_cell(root / cell, recipe, cell, FakeScorers())
        audit(root / cell, recipe)
        audit(root / cell, recipe)
    report(root, recipe)
    paired = pd.read_csv(root / "report/paired_growth.csv")
    assert len(paired) == 6 and (paired.raw_joint_yield_per_1000 == 0).all()
    matched = pd.read_csv(root / "report/matched_frontier.csv")
    assert not matched.both_complete.any()  # No silent promotion of32 draws to100.
    assert matched.activity_delta.isna().all()
    assert not json.loads((root / "report/report.json").read_text())["deployment_approved"]
    (root / "coverage/seed42/pool.csv").write_text("corrupt")
    with pytest.raises(ValueError):
        report(root, recipe)
