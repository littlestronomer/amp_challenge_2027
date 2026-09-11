import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiment_utils import mark_files, sha256, write_json


def test_invalid_draws_stay_in_denominator():
    pd = pytest.importorskip("pandas")
    from benchmark_external import diagnose

    frame = pd.DataFrame({"sequence": ["AAAAAAAA", "CCCCCCCC", "", "XX", "AAAAAAAA"],
                          "activity": [.9, .9, np.nan, np.nan, .9],
                          "risk": [.1, .1, np.nan, np.nan, .1],
                          "evaluation_risk": [.2, .2, np.nan, np.nan, .2]})
    rows, bins = diagnose(frame, set(), [5])
    assert rows[0]["raw_joint_yield_per_1000"] == pytest.approx(400.)
    assert rows[0]["invalid_draws"] == 2
    assert rows[0]["valid_evaluation_risk_mean"] == pytest.approx(.2)
    assert sum(b["raw_share"] for b in bins) == pytest.approx(.6)


def test_hydramp_adapter_exact_draws_no_sorting_or_filtering(tmp_path, monkeypatch):
    import external_raw_worker as worker

    calls = []
    class FakeGenerator:
        def __init__(self, **kwargs):
            assert kwargs["softmax"] is True

        def unconstrained_generation(self, **kwargs):
            calls.append(kwargs)
            return ["", "AAAAAAAA"][:kwargs["n_target"]]

    code = tmp_path / "inference.py"
    code.write_text("# fake pinned module")
    module = SimpleNamespace(HydrAMPGenerator=FakeGenerator, __file__=str(code))
    package = SimpleNamespace(inference=module)
    monkeypatch.setitem(sys.modules, "amp", SimpleNamespace(inference=package))
    monkeypatch.setitem(sys.modules, "amp.inference", package)
    monkeypatch.setitem(sys.modules, "amp.inference.inference", module)
    monkeypatch.setattr(worker.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        read_text=lambda p: json.dumps({"vcs_info": {"commit_id": "6590d2f4c2963f25d30669052a4c4a857e0e7279"}})))
    sequences, runtime = worker.sample_hydramp(tmp_path, 3, 60042, 2)
    assert sequences == ["", "AAAAAAAA", ""]
    assert [c["seed"] for c in calls] == [worker.batch_seed(60042, 0), worker.batch_seed(60042, 1)]
    assert all(not c["filter_out"] and not c["properties"] and c["n_attempts"] == 1 for c in calls)
    assert runtime["device"] == "cpu"


def test_batch_seed_schedules_are_disjoint_and_repeatable():
    from external_raw_worker import batch_seed

    schedules = [{batch_seed(seed, i) for i in range(6250)} for seed in (60042, 60043, 60044)]
    assert len(set.union(*schedules)) == 18750
    assert all(0 <= s < 2**32 for schedule in schedules for s in schedule)
    assert batch_seed(60042, 255) == batch_seed(60042, 255)
    for seed, index in [(-1, 0), (65536, 0), (42, -1), (42, 65536)]:
        with pytest.raises(ValueError):
            batch_seed(seed, index)


def test_overlap_detects_old_shifted_batch_stream():
    from benchmark_external import cross_seed_overlap

    draws = [str(i) for i in range(96)]
    row = cross_seed_overlap({42: draws[:64], 43: draws[32:]})[0]
    assert row["shared_unique"] == 32
    assert row["shift_one_batch_equal"]
    assert row["same_position_count"] == 0


def test_inventory_rejects_pointer_wrong_revision_and_dirty_source(tmp_path, monkeypatch):
    import benchmark_external as runner

    (tmp_path / "checkpoint").mkdir()
    model = tmp_path / "checkpoint/model.pt"
    weights = b"real weights"
    model.write_bytes(weights)
    expected = sha256(model)
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{expected}\nsize 12\n".encode()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").write_text("dummy")
    state = {"commit": runner.KITS["ampdiffusion"]["commit"], "dirty": b""}
    def fake_git(root, *args):
        if args[0] == "rev-parse":
            return state["commit"].encode()
        if args[0] == "diff":
            return state["dirty"]
        if args[0] == "show":
            return pointer
        if "--others" in args:
            return b""
        return b"checkpoint/model.pt\0"
    monkeypatch.setattr(runner, "git", fake_git)
    assert runner.inventory(tmp_path, "ampdiffusion")["files"]["checkpoint/model.pt"] == expected
    model.write_bytes(pointer)
    with pytest.raises(ValueError, match="LFS"):
        runner.inventory(tmp_path, "ampdiffusion")
    state["commit"] = "wrong"
    with pytest.raises(ValueError, match="commit"):
        runner.inventory(tmp_path, "ampdiffusion")
    state["commit"] = runner.KITS["ampdiffusion"]["commit"]
    state["dirty"] = b"src/changed.py"
    with pytest.raises(ValueError, match="Modified"):
        runner.inventory(tmp_path, "ampdiffusion")


def test_diffusion_worker_exact_budget_and_seeded_replay(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import external_raw_worker as worker

    class Model(torch.nn.Module):
        def sample(self, batch_size, design_len):
            return torch.randint(0, 2, (batch_size, design_len))

    def decode(model, batch, indices, length):
        return ["".join("AC"[int(x)] for x in row[:length]) for row in batch]

    source = tmp_path / "src/generate.py"
    source.parent.mkdir()
    source.write_text("# pinned fake")
    cache = tmp_path / "hub/checkpoints"
    cache.mkdir(parents=True)
    (cache / "esm2_t6_8M_UR50D.pt").write_bytes(b"decoder")
    module = SimpleNamespace(__file__=str(source), _decode=decode,
                             set_seed=torch.manual_seed,
                             load_model=lambda p, d: (Model(), Model(), None, []))
    monkeypatch.setitem(sys.modules, "ampdiffusion_starter_kit", SimpleNamespace(generate=module))
    monkeypatch.setitem(sys.modules, "ampdiffusion_starter_kit.generate", module)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache.parent))
    first, runtime = worker.sample_diffusion(tmp_path, 33, 60042, 32, "cpu")
    second, _ = worker.sample_diffusion(tmp_path, 33, 60042, 32, "cpu")
    assert first == second and len(first) == 33
    assert all(10 <= len(s) <= 40 for s in first)
    assert len(runtime["decoder_assets"]) == 1


def test_full_offline_pipeline_and_tamper_detection(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    import benchmark_external as runner

    reference = tmp_path / "ref.fasta"
    reference.write_text(">ref\nFFFFFFFF\n")
    recipe = {"draws": 32, "scales": [32], "device": "cpu", "code": {"test": True},
              "sources": {"reference": str(reference), "common": {"inputs": {str(reference): sha256(reference)}}},
              "controls": {}, "kits": {m: {"python": sys.executable, "root": str(tmp_path)} for m in runner.KITS}}
    root = tmp_path / "results"
    frame = pd.DataFrame({"sequence": ["AAAAAAAA", "CCCCCCCC"]*16,
                          "activity": [.9]*32, "risk": [.1]*32, "evaluation_risk": [.2]*32})
    for method in runner.METHODS:
        for seed in (42, 43, 44):
            cell = f"{method}/seed{seed}"
            source = tmp_path / "controls" / cell
            source.mkdir(parents=True)
            frame.to_csv(source / "pool.csv", index=False)
            write_json(source / "run.json", {})
            mark_files(source, "complete.json", ["pool.csv", "run.json"])
            recipe["controls"][cell] = {"root": str(source), "marker": sha256(source / "complete.json")}
    monkeypatch.setattr(runner, "verify_pins", lambda r: None)
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        path = command[command.index("--out")+1]
        write_json(__import__("pathlib").Path(path), {"sequences": ["", "AAAAAAAA", "CCCCCCCC", "XX"]*8,
                        "runtime": {}, "packages": [], "python": "test"})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    class Scorers:
        def score(self, sequences):
            assert all(runner.valid(s) for s in sequences)
            return np.tile([.9, .1, .2], (len(sequences), 1))
    for method in runner.KITS:
        for seed in (42, 43, 44):
            dest = root / "raw" / method / f"seed{seed}"
            runner.raw_sample(dest, recipe, method, seed)
            runner.raw_sample(dest, recipe, method, seed)  # verified skip
            runner.score(dest, recipe, Scorers())
            runner.score(dest, recipe, Scorers())
    assert len(calls) == 6
    for method in (*runner.METHODS, *runner.KITS):
        for seed in (42, 43, 44):
            origin = runner.checked_origin(recipe, method, seed, root)
            if method in runner.METHODS:
                pool = frame
            else:
                pool = pd.read_csv(root / "raw" / method / f"seed{seed}/scored/pool.csv", keep_default_na=False)
            runner.audit_cell(root / "audit" / method / f"seed{seed}", pool, recipe, origin)
    runner.report(root, recipe)
    summary = pd.read_csv(root / "report/growth.csv")
    assert set(summary[summary.method == "hydramp_raw"].invalid_draws) == {16}
    assert set(summary.raw_joint_yield_per_1000) == {62.5}
    matched = pd.read_csv(root / "report/matched_frontier.csv")
    assert not matched.matched.any()
    assert matched.activity_delta.isna().all()
    (root / "raw/hydramp_raw/seed42/raw.json").write_text("tampered")
    with pytest.raises(ValueError):
        runner.report(root, recipe)


def test_all_invalid_shortfall(tmp_path):
    pd = pytest.importorskip("pandas")
    from benchmark_external import audit_cell

    ref = tmp_path / "ref.fa"
    ref.write_text(">r\nAAAAAAAA\n")
    frame = pd.DataFrame({"sequence": [""], "activity": [""], "risk": [""], "evaluation_risk": [""]})
    recipe = {"draws": 1, "scales": [1], "sources": {"reference": str(ref)}}
    audit_cell(tmp_path / "audit", frame, recipe, {})
    status = json.loads((tmp_path / "audit/status.json").read_text())
    assert status["library_size"] == 0 and status["invalid_draws"] == 1
