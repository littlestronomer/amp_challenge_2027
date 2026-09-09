import json

import numpy as np
import pytest
from experiment_utils import mark_files, sha256, write_json
from scale_validation import checked, library_ids, reference_violation, sources


def test_ordered_library_and_explicit_shortfall():
    seqs = ["AAAAAAAA", "CCCCCCCC", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE"]
    assert library_ids(seqs, {"AAAAAAAA"}, size=2) == [1, 3]
    assert library_ids(seqs, {"AAAAAAAA"}, size=50000) == [1, 3, 4]
    with pytest.raises(ValueError, match="Invalid"):
        library_ids(["XXXXXXX"], set())


def test_reference_similarity_boundary_matches_validator():
    assert not reference_violation("AAAAAAAAAA", {"AAAAAAAACC"})  # exactly .8
    assert reference_violation("AAAAAAAAAA", {"AAAAAAAAAC"})
    assert not reference_violation("AAAAAAAAAA", {"A"*50})


@pytest.fixture
def pilot_sources(tmp_path):
    from experiment_utils import write_summary
    from summarize_grpo_pilot import ARMS, METRICS, RAW_METRICS

    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.pt").write_bytes(b"original weights")
    write_json(initial / "config.json", {})
    ref = tmp_path / "reference.fasta"
    ref.write_text(">ref\nAAAAAAAA\n")
    inputs = {str(p.resolve()): sha256(p) for p in (initial / "model.pt", initial / "config.json", ref)}
    for method in ("raft", "grpo"):
        for seed in (42, 43, 44):
            root = tmp_path / f"{method}-generator-seed{seed}-v1"
            (root / "policy").mkdir(parents=True)
            (root / "policy/model.pt").write_bytes(f"{method}{seed}".encode())
            write_json(root / "policy/config.json", {})
            run = {"kind": "generator_improvement_pilot_v1",
                   "args": {"method": method, "seed": seed, "out": str(root),
                            "checkpoint": str(initial), "reference": str(ref), "evaluator": str(tmp_path / "eval")},
                   "inputs": inputs, "configs": {}, "revision": "a"*40, "code": {"source_sha256": "code"},
                   "runtime": {"warn_only": False, "attention": "math_only"},
                   "evaluator_marker": {}, "objective": method, "sampling": "shared",
                   "kl_beta": .05, "clip": .2, "update_epochs": 2}
            write_json(root / "run.json", run)
            write_json(root / "status.json", {"training_draws": 5120, "kl_stopped": False})
            rows = [{"arm": a.replace("grpo_selection", method+"_selection"),
                     "draws": 5124 if a.endswith("matched_total") else 4,
                     "selection_complete": True, **{k: .5 for k in METRICS+RAW_METRICS}} for a in sorted(ARMS)]
            write_summary(root / "summary.csv", rows)
            mark_files(root, "complete.json", ["run.json", "summary.csv", "status.json", "policy/model.pt", "policy/config.json"])
    return tmp_path


def test_sources_pin_every_endpoint_and_detect_changes(pilot_sources):
    result = sources(pilot_sources)
    assert len(result["cells"]) == 9
    assert result["cells"]["baseline/seed42"]["sampling_seed"] == 40042
    changed = pilot_sources / "raft-generator-seed43-v1/policy/model.pt"
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError):
        sources(pilot_sources)


def test_cli_preflight_is_read_only(pilot_sources, capsys):
    pytest.importorskip("pandas")
    from validate_generator_scale import main

    output = pilot_sources / "scale-output"
    main(["sample", "--pilot-root", str(pilot_sources), "--out", str(output), "--list"])
    assert not output.exists()
    assert "40042" in capsys.readouterr().out
    with pytest.raises(ValueError, match="multiple"):
        main(["sample", "--pilot-root", str(pilot_sources), "--out", str(output), "--draws", "2049", "--list"])


def test_checked_rejects_unlisted_required_and_parent_paths(tmp_path):
    write_json(tmp_path / "complete.json", {"files": {"../outside": "fake"}})
    with pytest.raises(ValueError):
        checked(tmp_path)
    write_json(tmp_path / "complete.json", {"files": {}})
    with pytest.raises(ValueError):
        checked(tmp_path, required=("pool.csv",))


def test_growth_and_frontier_do_not_use_evaluator_for_selection():
    pd = pytest.importorskip("pandas")
    from scale_validation import frontier, growth, separated_ids

    frame = pd.DataFrame({"sequence": ["AAAAAAAA", "AAAAAAAC", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE", "FFFFFFFF"],
                          "activity": [.9]*6, "risk": [.1]*6, "evaluation_risk": [.1]*6})
    rows, lengths = growth(frame, {"FFFFFFFF"}, [6])
    assert rows[0]["joint_pass_unique"] == 5
    assert rows[0]["joint_pass_separated_08"] == 4
    assert sum(r["bin_draws"] for r in lengths) == 6
    reference = {"AAAAAAAG"}  # first two candidates fail the >.8 novelty guard
    first, selected = frontier(frame, reference, shortlist=6, top=3, targets=(.6, .99))
    assert all(r["complete"] for r in first)
    assert all(r["sequence"] not in {"AAAAAAAA", "AAAAAAAC"} for r in selected)
    changed = frame.copy()
    changed["evaluation_risk"] = [.9]*6
    other, other_selected = frontier(changed, reference, shortlist=6, top=3, targets=(.6, .99))
    assert [r["sequence"] for r in selected] == [r["sequence"] for r in other_selected]
    assert other[0]["evaluation_risk"] > first[0]["evaluation_risk"]
    short, _ = frontier(frame, reference, shortlist=1, top=3)
    assert not any(r["complete"] for r in short)
    assert len(separated_ids(frame, target_distance=.99)) == 5


def test_generate_audit_report_resume_and_tampering(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    from pilot_grpo import strict_runtime
    from validate_generator_scale import audit_cell, generate_cell, report

    from amp_challenge_2027.model import DecoderConfig, build_model, save_model

    torch.set_num_threads(1)
    strict_runtime(42)
    model, cfg = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2))
    checkpoint = tmp_path / "checkpoint"
    save_model(model, checkpoint, config=cfg)
    ref = tmp_path / "ref.fasta"
    ref.write_text(">ref\nAAAAAAAA\n")

    class FakeScorers:
        def score(self, seqs):
            return np.tile([.9, .1, .2], (len(seqs), 1))

    # Keep real token sampling/model loading, but only64 draws for CPU integration.
    recipe = {"draws": 64, "scales": [32, 64], "device": "cpu", "sources": {
        "reference": str(ref), "common": {"inputs": {str(ref): sha256(ref)}},
        "cells": {"baseline/seed42": {"checkpoint": str(checkpoint), "sampling_seed": 40042}}}}
    dest = tmp_path / "result/baseline/seed42"
    generate_cell(dest, recipe, "baseline/seed42", FakeScorers())
    before = (dest / "pool.csv").read_bytes()
    assert len(pd.read_csv(dest / "pool.csv")) == 64
    status = json.loads((dest / "status.json").read_text())
    assert status["library_shortfall"] > 0 and not status["library_complete"]
    assert not (dest / "library.fasta").exists()
    generate_cell(dest, recipe, "baseline/seed42", FakeScorers())
    assert (dest / "pool.csv").read_bytes() == before
    repeat = tmp_path / "repeat/baseline/seed42"
    generate_cell(repeat, recipe, "baseline/seed42", FakeScorers())
    assert (repeat / "pool.csv").read_bytes() == before
    audit_cell(dest, recipe)
    audit_cell(dest, recipe)
    report(tmp_path / "result", recipe)
    result = json.loads((tmp_path / "result/report/report.json").read_text())
    assert result["complete_libraries"] == 0 and result["official_evaluations"] == 0
    partial = tmp_path / "partial/baseline/seed42"
    partial.mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete sampling cell"):
        generate_cell(partial, recipe, "baseline/seed42", FakeScorers())
    (dest / "pool.csv").write_text("corrupt")
    with pytest.raises(ValueError):
        audit_cell(dest, recipe)


def test_library_target_is_exactly_50000():
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    def sequence(n):
        return "AAAA" + "".join(alphabet[(n // 20**i) % 20] for i in range(4))
    seqs = [sequence(n) for n in range(50002)]
    ids = library_ids([seqs[0], *seqs], {seqs[0]})
    assert len(ids) == 50000 and ids[0] == 2 and ids[-1] == 50001


def test_seeded_official_evaluation_and_cache(tmp_path, monkeypatch):
    from pathlib import Path

    import experiment_utils

    directory = tmp_path / "library"
    directory.mkdir()
    (directory / "library.fasta").write_text(">s\nAAAAAAAA\n")
    reference = tmp_path / "reference.fasta"
    reference.write_text(">s\nCCCCCCCC\n")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--out")+1]).write_text("metric,value\nFBD,0.1\n")
        write_json(Path(command[command.index("--json-out")+1]), {"FBD": .1})

    monkeypatch.setattr(experiment_utils.subprocess, "run", fake_run)
    for _ in range(2):
        result = experiment_utils.evaluate_library(directory, reference=reference, esm_model="650M", device="cpu", seed=2027)
        assert result["FBD"] == .1
    assert len(commands) == 1 and commands[0][-2:] == ["--seed", "2027"]
    with pytest.raises(ValueError, match="recipe changed"):
        experiment_utils.evaluate_library(directory, reference=reference, esm_model="650M", device="cpu", seed=42)


def test_frozen_scorer_shapes_and_calibration(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pandas")
    import validate_generator_scale as runner

    import amp_challenge_2027.reward_benchmark as benchmark

    inputs, configs = {}, {}
    for name, attr, bias, temperature in (("activity", "REWARD_DIR", 2., 2.),
                                         ("hemolysis", "REWARD_HEMO_DIR", -2., 1.)):
        directory = tmp_path / name
        directory.mkdir()
        head = benchmark.build_head(480, 1, "mlp")
        for parameter in head.parameters():
            parameter.data.zero_()
        head.classifier.bias.data.fill_(bias)
        path = directory / "classifier.pt"
        torch.save(head.state_dict(), path)
        inputs[str(path)] = sha256(path)
        configs[name] = {"esm_model": "fake", "temperature": temperature}
        monkeypatch.setattr(runner, attr, directory)
    evaluator = tmp_path / "evaluator"
    for seed, bias in ((42, -1.), (43, 0.), (44, 1.)):
        directory = evaluator / f"hemolysis/original/seed{seed}"
        directory.mkdir(parents=True)
        head.classifier.bias.data.fill_(bias)
        path = directory / "head.pt"
        torch.save(head.state_dict(), path)
        inputs[str(path)] = sha256(path)
    write_json(evaluator / "hemolysis/original/calibration.json", {"temperature": 2.})

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, seqs, batch_size):
            return np.zeros((len(seqs), 480), dtype=np.float32)

    monkeypatch.setattr(benchmark, "FrozenEncoder", Encoder)
    recipe = {"sources": {"common": {"configs": configs, "revision": "a"*40, "inputs": inputs},
                           "evaluator": str(evaluator)}}
    scorer = runner.Scorers(recipe, "cpu")
    scores = scorer.score(["AAAAAAAA"]*513)
    assert scores.shape == (513, 3)
    assert np.allclose(scores[:, 0], 1/(1+np.exp(-1)))
    assert np.allclose(scores[:, 1], 1/(1+np.exp(2)))
    assert np.allclose(scores[:, 2], .5)
    assert all(not p.requires_grad for h in scorer.heads+scorer.evaluators for p in h.parameters())
