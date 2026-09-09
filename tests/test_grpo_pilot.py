import json

import numpy as np
import pytest
from pilot_grpo import rewards, select


def test_reward_and_selection_fail_closed():
    assert rewards(["A", "B"], np.array([.8, .2]), np.array([.1, .1]), set())[0] > 0
    with pytest.raises(ValueError):
        rewards(["A"], np.array([np.nan]), np.array([0.]), set())
    assert select(["A", "A", "B"], [.9, .9, .4], [.1, .1, .0], set(), 100) == [0]
    assert select(["A"], [.9], [.1], {"A"}, 100) == []


def test_policy_loss_and_sampler():
    torch = pytest.importorskip("torch")
    from amp_challenge_2027.config import BOS_ID, EOS_ID, PAD_ID
    from amp_challenge_2027.grpo import advantages, distributions, objective, rollout
    from amp_challenge_2027.model import DecoderConfig, build_model
    from amp_challenge_2027.tokenizer import RESIDUE_TO_ID

    torch.set_num_threads(1)
    model, _ = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2, dropout=.1))
    model.eval()
    tokens, sequences = rollout(model, 4, torch.Generator().manual_seed(4), "cpu", maximum=10)
    assert all(8 <= len(s) <= 10 and set(s) <= set(RESIDUE_TO_ID) for s in sequences)
    assert all(r[0] == BOS_ID and r.count(EOS_ID) == 1 for r in tokens.tolist())
    lp = distributions(model, tokens, maximum=10)
    assert torch.allclose(lp.exp().sum(-1), torch.ones_like(lp[..., 0]))
    adv = advantages(torch.tensor([0., 1., 2., 2.]), 2)
    assert torch.equal(adv, torch.tensor([-1., 1., 0., 0.]))
    old = lp.detach().clone()
    loss, kl = objective(lp, old, old, tokens, adv)
    assert abs(kl.item()) < 1e-6
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    mask = tokens[:, 1:] != PAD_ID
    assert torch.isfinite(lp.gather(-1, tokens[:, 1:, None]).squeeze(-1)[mask]).all()


def test_strict_runtime_overrides_warning_only_seed_helper():
    torch = pytest.importorskip("torch")
    from pilot_grpo import assert_strict_runtime, strict_runtime

    from amp_challenge_2027.training import enable_determinism

    enable_determinism(42)
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    runtime = strict_runtime(42)
    assert runtime["warn_only"] is False
    assert_strict_runtime()
    enable_determinism(42)
    with pytest.raises(RuntimeError, match="settings changed"):
        assert_strict_runtime()
    strict_runtime(42)


def test_offline_end_to_end_budget_and_provenance(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import pilot_grpo as pilot
    from experiment_utils import mark_files, sha256, write_json

    from amp_challenge_2027.model import DecoderConfig, build_model, save_model
    from amp_challenge_2027.reward_benchmark import build_head

    torch.set_num_threads(1)
    dirs = {n: tmp_path / n for n in ("activity", "hemolysis", "generator", "evaluator")}
    monkeypatch.setattr(pilot, "REWARD_DIR", dirs["activity"])
    monkeypatch.setattr(pilot, "REWARD_HEMO_DIR", dirs["hemolysis"])
    parity = {"tasks": {}}
    cfg = {"esm_model": "fake", "temperature": 1., "task": "binary", "unfreeze_layers": 0}
    for name in ("activity", "hemolysis"):
        dirs[name].mkdir()
        head = build_head(480, 1, "mlp")
        for p in head.parameters():
            p.data.zero_()
        head.dense.weight.data[0, 0] = 1.
        head.classifier.weight.data[0, 0] = .1
        head.classifier.bias.data.fill_(2 if name == "activity" else 0)
        torch.save(head.state_dict(), dirs[name] / "classifier.pt")
        write_json(dirs[name] / "classifier_config.json", cfg)
        parity["tasks"][name] = {"probe_verified": True, "config": cfg, "head_sha256": sha256(dirs[name] / "classifier.pt"), "revision": "a"*40}
    write_json(tmp_path / "parity.json", parity)
    model, config = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=2))
    save_model(model, dirs["generator"], config=config)
    evaluator = dirs["evaluator"]
    inventory = {}
    for seed in (42, 43, 44):
        cell = f"hemolysis/original/seed{seed}"
        dest = evaluator / cell
        dest.mkdir(parents=True)
        torch.save(head.state_dict(), dest / "head.pt")
        mark_files(dest, "complete.json", ["head.pt"])
        inventory[cell] = sha256(dest / "complete.json")
    dest = evaluator / "hemolysis/original"
    write_json(dest / "calibration.json", {"temperature": 1.})
    mark_files(dest, "complete.json", ["calibration.json"])
    inventory["hemolysis/original"] = sha256(dest / "complete.json")
    write_json(evaluator / "model_inventory.json", inventory)
    write_json(evaluator / "run.json", {"kind": "paired_label_ablation_training_v1", "manifest": {
        "backbone": "fake", "revision": "a"*40, "protocol": {"architectures": ["mlp"], "training_seeds": [42, 43, 44]}}})
    mark_files(evaluator, "complete.json", ["run.json", "model_inventory.json"])
    (tmp_path / "reference.fasta").write_text(">r\nAAAAAAAA\n")
    import amp_challenge_2027.reward_benchmark as benchmark

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, sequences, batch_size):
            result = np.zeros((len(sequences), 480), dtype=np.float32)
            result[:, 0] = [sum(map(ord, s)) / 1000. for s in sequences]
            return result

    monkeypatch.setattr(benchmark, "FrozenEncoder", Encoder)
    # Keep output separate from all input parents, including parity parent.
    parity_dir = tmp_path / "parity"
    parity_dir.mkdir()
    (tmp_path / "parity.json").rename(parity_dir / "report.json")
    args = ["--checkpoint", str(dirs["generator"]), "--parity", str(parity_dir / "report.json"),
            "--evaluator", str(evaluator), "--reference", str(tmp_path / "reference.fasta"),
            "--out", str(tmp_path / "out"), "--device", "cpu", "--steps", "1", "--groups", "1",
            "--group-size", "2", "--eval-draws", "4", "--top", "2"]
    pilot.main(args + ["--list"])
    assert not (tmp_path / "out").exists()
    pilot.main(args)
    status = json.loads((tmp_path / "out/status.json").read_text())
    assert status["training_draws"] == 2
    assert not status["deployment_approved"]
    import csv
    with (tmp_path / "out/summary.csv").open() as f:
        summary = {r["arm"]: r for r in csv.DictReader(f)}
    assert int(summary["baseline_selection_matched_total"]["draws"]) == 6
    assert int(summary["grpo_selection"]["draws"]) + status["training_draws"] == 6
    assert (tmp_path / "out/complete.json").exists()
    with pytest.raises(ValueError, match="new output"):
        pilot.main(args)
    # An independent same-seed execution must produce identical scored pools
    # and policy tensors, not merely the same deterministic-settings flags.
    repeat_args = args.copy()
    repeat_args[repeat_args.index("--out") + 1] = str(tmp_path / "repeat")
    pilot.main(repeat_args)
    assert (tmp_path / "repeat/summary.csv").read_bytes() == (tmp_path / "out/summary.csv").read_bytes()
    assert (tmp_path / "repeat/training_samples.csv").read_bytes() == (tmp_path / "out/training_samples.csv").read_bytes()
    first = torch.load(tmp_path / "out/policy/model.pt", weights_only=True)
    second = torch.load(tmp_path / "repeat/policy/model.pt", weights_only=True)
    assert all(torch.equal(first[k], second[k]) for k in first)
    initial = torch.load(dirs["generator"] / "model.pt", weights_only=True)
    assert any(not torch.equal(first[k], initial[k]) for k in first)
    from summarize_grpo_pilot import main as summarize

    run_paths = [str(tmp_path / "out")]
    for seed in (43, 44):
        seed_args = args.copy()
        seed_args[seed_args.index("--out") + 1] = str(tmp_path / f"seed{seed}")
        pilot.main(seed_args + ["--seed", str(seed)])
        run_paths.append(str(tmp_path / f"seed{seed}"))
    summarize(["--runs", *run_paths, "--out", str(tmp_path / "comparison")])
    assert (tmp_path / "comparison/paired_summary.csv").exists()
    # A repeated seed is not a replication set.
    with pytest.raises(ValueError, match="seeds42"):
        summarize(["--runs", run_paths[0], run_paths[0], run_paths[2], "--out", str(tmp_path / "bad")])
    # Even a correctly hashed old pilot cannot be mixed into deterministic v2.
    from summarize_grpo_pilot import read_run
    manifest_path = tmp_path / "repeat/run.json"
    old_run = json.loads(manifest_path.read_text())
    old_run["kind"] = "grpo_pilot_v1"
    write_json(manifest_path, old_run)
    completion = json.loads((tmp_path / "repeat/complete.json").read_text())
    completion["files"]["run.json"] = sha256(manifest_path)
    write_json(tmp_path / "repeat/complete.json", completion)
    with pytest.raises(ValueError, match="v2"):
        read_run(tmp_path / "repeat")
    changed_path = tmp_path / "seed44/run.json"
    changed_run = json.loads(changed_path.read_text())
    changed_run["args"]["lr"] *= 2
    write_json(changed_path, changed_run)
    completion_path = tmp_path / "seed44/complete.json"
    completion = json.loads(completion_path.read_text())
    completion["files"]["run.json"] = sha256(changed_path)
    write_json(completion_path, completion)
    with pytest.raises(ValueError, match="identical"):
        summarize(["--runs", *run_paths, "--out", str(tmp_path / "mixed-settings")])
