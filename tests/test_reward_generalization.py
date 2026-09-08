"""No downloaded weights or real experiment data: synthetic leakage/training checks."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import benchmark_reward_generalization as benchmark
import numpy as np
import prepare_reward_generalization as prepare
import pytest
from experiment_utils import mark_files, sha256, write_json, write_summary

from amp_challenge_2027.config import PANEL_GENERA
from amp_challenge_2027.generalization import (
    arrays,
    assign_families,
    audit_boundary,
    calibrate_temperature,
    curate_labels,
    family_bootstrap,
    family_components,
    macro_rank,
)
from amp_challenge_2027.reward_audit import prediction_report
from amp_challenge_2027.reward_benchmark import FrozenEncoder, build_head, fit_head, predict_head


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_binary_conflicts_and_repeats_not_row_weighted(tmp_path):
    path = tmp_path / "labels.csv"
    write_summary(path, [
        {"sequence": "klaklaklak", "organism": "E. coli", "label": "active"},
        {"sequence": "KLAKLAKLAK", "organism": "S. aureus", "label": "inactive"},
        {"sequence": "LLLLAAAAKK", "organism": "E. coli", "label": "1"},
        {"sequence": "LLLLAAAAKK", "organism": "E. coli", "label": "true"},
        {"sequence": "AAAAAAAAAA", "organism": "E. coli", "label": "intermediate"},
        {"sequence": "AAAAAXAAAA", "organism": "E. coli", "label": "0"},
        {"sequence": "A" * 51, "organism": "E. coli", "label": "0"},
    ])
    records, events, bridges, info = curate_labels(path, "activity")
    assert records == [{"sequence": "LLLLAAAAKK", "targets": [1], "mask": [1]}]
    assert "KLAKLAKLAK" in bridges and "AAAAAAAAAA" in bridges and "AAAAAXAAAA" not in bridges
    assert sum(e["reason"] == "conflicting_output_masked" for e in events) == 2
    assert info["counts"]["conflicting_sequence_output_groups"] == 1
    assert info["counts"]["duplicate_evidence_rows"] == 2


def test_panel_conflicts_mask_only_affected_genus_and_are_order_invariant(tmp_path):
    rows = [{"sequence": "KLAKLAKLAK", "organism": g, "label": label} for g, label in
            [("E. coli", "active"), ("E. coli", "inactive"), ("S. aureus", "inactive"), ("unknown", "active")]]
    path = tmp_path / "panel.csv"
    write_summary(path, rows)
    a, _, _, _ = curate_labels(path, "panel")
    write_summary(path, rows[::-1])
    b, _, _, _ = curate_labels(path, "panel")
    assert a == b
    assert a[0]["mask"][PANEL_GENERA.index("E. coli")] == 0
    assert a[0]["mask"][PANEL_GENERA.index("S. aureus")] == 1
    assert a[0]["targets"][PANEL_GENERA.index("S. aureus")] == 0


def test_transitive_families_close_leader_cluster_hole():
    pytest.importorskip("Levenshtein")
    # A--B and B--C have .8 similarity; A--C only .6. Must be ONE family.
    a, b, c = "AAAAAAAAAA", "AAAAAAAACC", "AAAAAACCCC"
    groups = family_components([a, b, c, "KKKKKKKKKK"], .8)
    assert groups[a] == groups[b] == groups[c] != groups["KKKKKKKKKK"]
    with pytest.raises(ValueError, match="violations"):
        audit_boundary({a: "train", b: "test", c: "test"}, .8)
    assert audit_boundary({a: "train", b: "train", c: "train", "KKKKKKKKKK": "test"}, .8)["violations"] == 0


def test_graph_matches_exhaustive_connectivity_and_boundary():
    ratio = pytest.importorskip("Levenshtein").ratio
    rng = np.random.default_rng(22)
    seqs = sorted({"".join(rng.choice(list("ACDE"), size=rng.integers(8, 21))) for _ in range(90)})
    groups = family_components(seqs, .8)
    for i, a in enumerate(seqs):
        for b in seqs[i + 1:]:
            if ratio(a, b) >= .8:
                assert groups[a] == groups[b]
    assignment = assign_families(groups, 2027)
    assert assignment == assign_families(dict(reversed(list(groups.items()))), 2027)
    audit_boundary(assignment, .8)
    assert max(ratio(a, b) for a in seqs for b in seqs if assignment[a] != assignment[b]) < .8


def test_giant_component_fails_instead_of_breaking_family():
    with pytest.raises(ValueError, match="four families"):
        assign_families({"A": 1, "B": 1, "C": 2}, 42)


def test_calibration_masks_and_never_changes_ranking():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32)
    logits = np.array([[5, 1000], [-1, -1000], [-4, 999], [-5, -999]], dtype=np.float64)
    a = calibrate_temperature(logits, y, mask)
    logits[:, 1] *= -100
    b = calibrate_temperature(logits, y, mask)
    assert a == b and a["outputs"] == [0]
    assert macro_rank(logits, y, mask) == macro_rank(logits / a["temperature"], y, mask)
    empty = calibrate_temperature(logits, y, np.zeros_like(mask))
    assert empty["temperature"] == 1 and empty["status"].startswith("not_fitted")


def test_family_bootstrap_paired_zero_and_undefined_support():
    y = np.tile([[0], [1]], (12, 1)).astype(float)
    logits, mask = y * 2 - 1, np.ones_like(y)
    families = np.repeat(np.arange(12), 2).tolist()
    ci = family_bootstrap(logits, y, mask, families, replicates=120, comparator=logits)
    assert ci["valid_replicates"] == 120
    assert ci["percentile_95_lower"] == ci["percentile_95_upper"] == 0
    assert family_bootstrap(logits, y, mask, [0] * len(y))["status"] == "insufficient_support"
    assert family_bootstrap(logits, np.ones_like(y), mask, families)["status"] == "insufficient_support"


def fixture_prepared(tmp_path):
    pytest.importorskip("Levenshtein")
    protocol = json.loads((prepare.REPO_ROOT / "experiments/reward_generalization_v1.json").read_text())
    protocol.update({"epochs": 2, "patience": 1, "bootstrap_replicates": 12})
    rng = np.random.default_rng(10)
    sequences = sorted({"".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=16)) for _ in range(160)})
    reconstruction = tmp_path / "reconstruction"
    write_json(reconstruction / "run.json", {"kind": "reward_validation_reconstruction_v1"})
    write_summary(reconstruction / "results.csv", [{"test_fixture": True}])
    mark_files(reconstruction, "complete.json", ["results.csv"])
    for task in ("activity", "panel", "hemolysis"):
        path = tmp_path / f"{task}.csv"
        write_summary(path, [{"sequence": s, "organism": genus, "label": (i + j) % 2}
                            for i, s in enumerate(sequences)
                            for j, genus in enumerate(PANEL_GENERA if task == "panel" else ["E. coli"])])
        protocol["tasks"][task] = {"data": str(path), "sha256": sha256(path)}
        write_json(reconstruction / task / "backbone.json", {"model": prepare.BACKBONE, "resolved_revision": "a" * 40})
        mark_files(reconstruction / task, "complete.json", ["backbone.json"])
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    argv = ["--protocol", str(path), "--reconstruction", str(reconstruction), "--out", str(tmp_path / "prepared")]
    prepare.main(argv)
    return argv


def test_prepare_shared_immutable_splits_resume_and_provenance(tmp_path):
    argv = fixture_prepared(tmp_path)
    root = tmp_path / "prepared"
    before = {p: sha256(p) for p in root.rglob("*") if p.is_file()}
    recipe, data = benchmark.load_prepared(root)
    assert len(data["activity"]["records"]) == 160
    assert data["activity"]["records"] == data["hemolysis"]["records"]
    assert [r["split"] for r in data["panel"]["records"]] == [r["split"] for r in data["activity"]["records"]]
    assert json.loads((root / "preflight.json").read_text())["eligible_for_training"]
    prepare.main(argv)
    assert {p: sha256(p) for p in before} == before
    assert recipe["revision"] == "a" * 40
    (tmp_path / "activity.csv").write_text("changed")
    with pytest.raises(ValueError, match="Pinned input changed"):
        benchmark.load_prepared(root)


def test_preparation_rejects_protocol_drift_and_damaged_cache(tmp_path):
    argv = fixture_prepared(tmp_path)
    (tmp_path / "prepared/dataset.json").write_text("{}")
    with pytest.raises(ValueError):
        prepare.main(argv)
    with pytest.raises(ValueError):
        benchmark.load_prepared(tmp_path / "prepared")


def test_final_test_requires_explicit_unlock_before_any_writes(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.main(["test", "--prepared", str(tmp_path / "missing"), "--training", str(tmp_path / "training"),
                        "--out", str(tmp_path / "test")])
    assert not (tmp_path / "test").exists()


def test_real_heads_shared_dense_round_trip_and_masked_training():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 8)).astype(np.float32)
    y = np.c_[np.arange(30) % 2, (np.arange(30) + 1) % 2].astype(np.float32)
    m = np.ones_like(y)
    m[::3, 1] = 0
    cfg = {"learning_rate": .001, "weight_decay": .01, "epochs": 3, "patience": 2, "head_batch_size": 8}
    for arch in ("linear", "mlp"):
        state, history, val = fit_head(x[:20], y[:20], m[:20], x[20:], y[20:], m[20:],
                                       protocol=cfg, seed=42, architecture=arch, device="cpu")
        assert len(history) >= 1 and np.isfinite(val).all()
        assert np.array_equal(val, predict_head(state, x[20:], 2, arch, device="cpu"))
        a, _, _ = fit_head(x[:20], y[:20], m[:20], x[20:], y[20:], m[20:],
                           protocol=cfg, seed=42, architecture=arch, device="cpu")
        assert all(torch.equal(state[k], a[k]) for k in state)
    head = build_head(8, 2, "mlp").eval()
    xx = torch.tensor(x)
    assert torch.equal(head(xx), head.classifier(head.act(head.dense(head.act(head.dense(xx))))))


def test_frozen_encoder_pins_revision_and_pools_special_tokens_without_download(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    calls = []

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(hidden_size=480, _commit_hash="a" * 40)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 480) + self.bias)

    def tokenizer(sequences, **kwargs):
        assert kwargs["truncation"] is False
        width = max(map(len, sequences)) + 2
        ids = torch.zeros((len(sequences), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(sequences):
            ids[i, :len(s) + 2] = torch.tensor([1] + [3] * len(s) + [2])
            mask[i, :len(s) + 2] = 1
        return {"input_ids": ids, "attention_mask": mask}

    model = TinyBackbone()

    def model_factory(name, **kwargs):
        calls.append((name, kwargs))
        return model

    def tokenizer_factory(name, **kwargs):
        calls.append((name, kwargs))
        return tokenizer

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model_factory)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_factory)
    encoder = FrozenEncoder(prepare.BACKBONE, "a" * 40, device="cpu")
    actual = encoder.encode(["AAAAAAAA", "CCCCCCCCC"], 2)
    assert actual.shape == (2, 480) and actual.dtype == np.float32
    assert np.allclose(actual[:, 0], [27 / 10, 30 / 11])
    assert not model.training and not model.bias.requires_grad
    assert calls == [(prepare.BACKBONE, {"revision": "a" * 40})] * 2
    with pytest.raises(ValueError, match="truncated"):
        encoder.encode(["A" * 51], 2)
    model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="commit"):
        FrozenEncoder(prepare.BACKBONE, "a" * 40, device="cpu")


def test_full_pipeline_no_test_access_during_fit_and_all_seeds_sealed(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    fixture_prepared(tmp_path)
    prepared = tmp_path / "prepared"
    _, data = benchmark.load_prepared(prepared)
    test_seqs = {r["sequence"] for r in data["activity"]["records"] if r["split"] == "test"}
    encoded, fit_calls = [], []

    class FakeEncoder:
        def __init__(self, model_id, revision, *, device):
            self.info = {"model": model_id, "revision": revision}

        def encode(self, sequences, batch_size):
            encoded.extend(sequences)
            return np.array([[sum(map(ord, s)) / 1000, *([.1] * 479)] for s in sequences], dtype=np.float32)

    original = benchmark.fit_head

    def checked_fit(tx, ty, tm, vx, vy, vm, **kwargs):
        # The fitter's API has no calibration/test arrays or paths.
        fit_calls.append((len(tx), len(vx), kwargs["seed"], kwargs["architecture"]))
        assert len(tx) == len(arrays(data["activity"]["records"], "train")[2])
        assert len(vx) == len(arrays(data["activity"]["records"], "validation")[2])
        return original(tx, ty, tm, vx, vy, vm, **kwargs)

    monkeypatch.setattr(benchmark, "FrozenEncoder", FakeEncoder)
    monkeypatch.setattr(benchmark, "fit_head", checked_fit)
    train = ["train", "--prepared", str(prepared), "--out", str(tmp_path / "training"), "--device", "cpu"]
    benchmark.main(train + ["--list"])
    assert not encoded and not (tmp_path / "training").exists()
    test = ["test", "--prepared", str(prepared), "--training", str(tmp_path / "training"),
            "--out", str(tmp_path / "test"), "--device", "cpu", "--unlock-test"]

    def interrupted_fit(*args, **kwargs):
        if len(fit_calls) == 1:
            raise RuntimeError("simulated interruption")
        return checked_fit(*args, **kwargs)

    monkeypatch.setattr(benchmark, "fit_head", interrupted_fit)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        benchmark.main(train)
    first_head = tmp_path / "training/activity/linear/seed42/head.pt"
    first_hash = sha256(first_head)
    with pytest.raises(FileNotFoundError):
        benchmark.main(test)
    assert not (tmp_path / "test").exists()
    monkeypatch.setattr(benchmark, "fit_head", checked_fit)
    benchmark.main(train)
    assert sha256(first_head) == first_hash
    assert len(fit_calls) == 18 and not test_seqs.intersection(encoded)
    assert not (tmp_path / "training/embeddings/test").exists()
    before = {p: sha256(p) for p in (tmp_path / "training").rglob("*") if p.is_file()}
    benchmark.main(train)
    assert len(fit_calls) == 18
    identity = benchmark.code_identity()
    with monkeypatch.context() as changed:
        changed.setattr(benchmark, "code_identity", lambda: {**identity, "commit": "different"})
        with pytest.raises(ValueError, match="code/runtime"):
            benchmark.main(test)
    assert not (tmp_path / "test").exists()
    actual_predict = benchmark.predict_head
    prediction_calls = []

    def interrupted_predict(*args, **kwargs):
        if len(prediction_calls) == 3:
            raise RuntimeError("interrupted test")
        prediction_calls.append(True)
        return actual_predict(*args, **kwargs)

    monkeypatch.setattr(benchmark, "predict_head", interrupted_predict)
    with pytest.raises(RuntimeError, match="interrupted test"):
        benchmark.main(test)
    first_test = tmp_path / "test/activity/linear/predictions.npz"
    prediction_hash = sha256(first_test)
    monkeypatch.setattr(benchmark, "predict_head", actual_predict)
    benchmark.main(test)
    assert sha256(first_test) == prediction_hash
    assert test_seqs <= set(encoded) and len(fit_calls) == 18
    assert {p: sha256(p) for p in before} == before
    results = read_csv(tmp_path / "test/results.csv")
    assert len(results) == 6
    assert all(r["macro_auroc"] for r in results)
    files = {p: sha256(p) for p in (tmp_path / "test").rglob("*") if p.is_file()}
    benchmark.main(test)
    assert {p: sha256(p) for p in files} == files
    # A changed head cannot be hidden by an otherwise complete test output.
    (tmp_path / "training/activity/mlp/seed42/head.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        benchmark.main(test)


@pytest.mark.parametrize("field,value", [("similarity_threshold", 0), ("training_seeds", [42, 42, 43]),
                                          ("fractions", [.8, .2]), ("learning_rate", float("nan"))])
def test_invalid_protocol_rejected(field, value):
    cfg = json.loads((prepare.REPO_ROOT / "experiments/reward_generalization_v1.json").read_text())
    cfg[field] = value
    with pytest.raises(ValueError):
        prepare.validate_protocol(cfg)


def test_masked_output_metrics_do_not_invent_chance_performance():
    y = np.array([[1, 0], [0, 0]], dtype=float)
    mask = np.array([[1, 0], [1, 0]], dtype=float)
    summary, rows, _ = prediction_report(y, y, mask, y, mask, temperature=1., outputs=["a", "b"])
    assert summary["stored_temperature"]["outputs_with_auroc"] == 1
    assert next(r for r in rows if r["output"] == "b")["auroc"] is None
