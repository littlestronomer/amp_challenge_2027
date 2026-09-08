"""Provisional reconstruction: exact inputs, no guessed provenance, no downloads."""

from __future__ import annotations

import json
from types import SimpleNamespace

import eval_reward_reconstruction as evaluate
import numpy as np
import pytest
from audit_reward_artifacts import tensor_fingerprint
from experiment_utils import mark_files, sha256, write_json, write_summary
from train_reward_classifier import load_activity_data, load_panel_data

from amp_challenge_2027.reward_audit import (
    binary_report,
    cross_split_similarity,
    discrimination,
    prediction_report,
    reconstruct_split,
    reliability,
    sigmoid,
)


def fixture_inputs(tmp_path, monkeypatch):
    specs, artifacts, records = {}, {}, {}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for name, member, seed, panel in (("activity", 2, 44, False), ("hemolysis", 0, 42, False), ("panel", 1, 43, True)):
        root, training = tmp_path / f"deployed_{name}", tmp_path / f"training_{name}"
        stem, task = ("classifier_panel", "panel") if panel else ("classifier", "binary")
        cfg = {"esm_model": evaluate.baseline.BACKBONE, "checkpoint_format": "head-only", "unfreeze_layers": 0,
               "task": task, "temperature": .95, "val_auroc": .7}
        if panel:
            cfg["genera"] = evaluate.baseline.PANEL_GENERA
        write_json(root / f"{stem}_config.json", cfg)
        head = root / f"{stem}.pt"
        head.write_bytes(f"test-only-{name}".encode())
        member_path = training / f"member{member}" / f"{stem}.pt"
        member_path.parent.mkdir(parents=True)
        member_path.write_bytes(head.read_bytes())
        write_json(training / "config.json", cfg)
        write_json(training / "members.json", [{"member": member, "seed": seed, "temperature": .95, "val_auroc": .7}])
        records.update({str(p): sha256(p) for p in (training / "config.json", training / "members.json")})
        data = tmp_path / f"{name}.csv"
        rows = []
        for i in range(20):
            seq = f"K{alphabet[i]}LAGRLKAA"
            for j, genus in enumerate(evaluate.baseline.PANEL_GENERA if panel else [""]):
                rows.append({"sequence": seq, "organism": genus, "label": "active" if (i + j) % 2 else "inactive"})
        write_summary(data, rows)
        specs[name] = {"data": str(data), "data_sha256": sha256(data), "training_run": str(training),
                       "member": member, "seed": seed, "task": task, "temperature": .95,
                       "recorded_val_auroc": .7, "head_sha256": sha256(head), "positive_label": "test-positive"}
        artifacts[name] = {"issues": [], "status": "provenance_incomplete", "directory": str(root), "stem": stem,
                           "files": {p.name: sha256(p) for p in (head, root / f"{stem}_config.json")},
                           "tensor_sha256": "unused-with-fake-predictor",
                           "matches": [{"path": str(member_path), "sha256": sha256(member_path), "tensor_identical": True}]}
    source, audit = tmp_path / "source", tmp_path / "audit"
    write_json(source / "run.json", {"kind": "paired_top100_v1", "classifiers": artifacts})
    write_json(source / "backbones.json", {name: "fake-revision" for name in specs})
    np.save(source / "reference_embeddings.npy", np.zeros((1, 4)))
    mark_files(source, "reference_cache.json", ["backbones.json", "reference_embeddings.npy"])
    write_json(audit / "inventory.json", {"artifacts": artifacts, "records": records})
    write_json(audit / "run.json", {"kind": "reward_artifact_inventory_v1", "inputs": {str(source / "run.json"): sha256(source / "run.json")}})
    mark_files(audit, "complete.json", ["inventory.json"])
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"kind": "reward_reconstruction_protocol_v1", "historical_validation_verified": False,
                          "assumptions": ["Test fixture; not a historical validation set"], "tasks": specs})
    calls = {"init": 0, "predict": 0}

    class FakePredictor:
        def __init__(self, cell, *, device):
            calls["init"] += 1
            self.outputs = len(cell["outputs"])
            self.info = {"resolved_revision": cell["revision"]}

        def predict(self, sequences, batch_size):
            calls["predict"] += 1
            return np.tile(np.linspace(-1, 1, len(sequences))[:, None], (1, self.outputs)).astype(np.float32)

    monkeypatch.setattr(evaluate, "PinnedPredictor", FakePredictor)
    return ["--protocol", str(protocol), "--artifact-audit", str(audit), "--source", str(source),
            "--out", str(tmp_path / "out"), "--device", "cpu", "--accept-reconstruction"], calls


def test_full_run_masks_provenance_resume_and_input_preservation(tmp_path, monkeypatch):
    argv, calls = fixture_inputs(tmp_path, monkeypatch)
    before = {p: sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    evaluate.main([a for a in argv if a != "--accept-reconstruction"] + ["--list"])
    assert not (tmp_path / "out").exists() and calls["init"] == 0
    evaluate.main(argv)
    assert calls == {"init": 3, "predict": 3}
    for task in ("hemolysis", "activity", "panel"):
        metrics = json.loads((tmp_path / f"out/{task}/metrics.json").read_text())
        assert metrics["summary"]["historical_validation_verified"] is False
        assert metrics["summary"]["independent_test"] is False
        assert metrics["modes"]["uncalibrated"]["macro_auroc"] == metrics["modes"]["stored_temperature"]["macro_auroc"]
        assert (tmp_path / f"out/{task}/split_assignments.csv").is_file()
    assert {p: sha256(p) for p in before} == before
    results = (tmp_path / "out/results.csv").read_bytes()
    evaluate.main(argv)
    assert (tmp_path / "out/results.csv").read_bytes() == results
    assert calls == {"init": 3, "predict": 3}


def test_acknowledgement_required_without_writing(tmp_path, monkeypatch):
    argv, calls = fixture_inputs(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        evaluate.main([a for a in argv if a != "--accept-reconstruction"])
    assert calls["init"] == 0 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("damage", ["csv", "head", "member", "seed", "temperature", "revision", "historical_claim"])
def test_pinned_inputs_fail_closed_before_inference(tmp_path, monkeypatch, damage):
    argv, calls = fixture_inputs(tmp_path, monkeypatch)
    if damage in {"csv", "head", "member"}:
        path = {"csv": tmp_path / "hemolysis.csv", "head": tmp_path / "deployed_hemolysis/classifier.pt",
                "member": tmp_path / "training_hemolysis/member0/classifier.pt"}[damage]
        path.write_bytes(path.read_bytes() + b"changed")
    elif damage == "revision":
        write_json(tmp_path / "source/backbones.json", {"hemolysis": "new-revision"})
    else:
        path = tmp_path / "protocol.json"
        protocol = json.loads(path.read_text())
        if damage == "historical_claim":
            protocol["historical_validation_verified"] = True
        else:
            protocol["tasks"]["hemolysis"][damage] = 999
        write_json(path, protocol)
    with pytest.raises(ValueError):
        evaluate.main(argv)
    assert calls["init"] == 0 and not (tmp_path / "out").exists()


def test_failed_reporting_resumes_saved_predictions(tmp_path, monkeypatch):
    argv, calls = fixture_inputs(tmp_path, monkeypatch)
    original = evaluate.cross_split_similarity
    monkeypatch.setattr(evaluate, "cross_split_similarity", lambda *_a: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate.main(argv)
    assert (tmp_path / "out/hemolysis/inference.json").exists()
    assert calls["predict"] == 1
    monkeypatch.setattr(evaluate, "cross_split_similarity", original)
    evaluate.main(argv)
    assert calls["predict"] == 3


def test_changed_run_or_damaged_complete_output_rejected(tmp_path, monkeypatch):
    argv, _ = fixture_inputs(tmp_path, monkeypatch)
    evaluate.main(argv)
    with pytest.raises(ValueError, match="inputs changed"):
        evaluate.main(argv + ["--batch-size", "32"])
    (tmp_path / "out/hemolysis/validation_predictions.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate.main(argv)


def test_input_change_during_loading_and_nested_output_rejected(tmp_path, monkeypatch):
    argv, calls = fixture_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="separate"):
        evaluate.main(argv + ["--out", str(tmp_path / "training_activity/new-audit")])
    original = evaluate.load_activity_data

    def changing_file(path):
        records = original(path)
        path.write_text(path.read_text() + "\n")
        return records

    monkeypatch.setattr(evaluate, "load_activity_data", changing_file)
    with pytest.raises(ValueError, match="changed"):
        evaluate.main(argv)
    assert calls["init"] == 0 and not (tmp_path / "out").exists()


def test_split_matches_legacy_random_algorithms_and_retains_duplicates(tmp_path, monkeypatch):
    fixture_inputs(tmp_path, monkeypatch)
    for name, panel, seed in (("activity", False, 44), ("panel", True, 43), ("hemolysis", False, 42)):
        records = (load_panel_data if panel else load_activity_data)(tmp_path / f"{name}.csv")
        if not panel:
            records += [dict(records[0])]  # Do not silently dedup old row-level training inputs.
        train, val = reconstruct_split(records, panel=panel, seed=seed)
        rng = np.random.default_rng(seed)
        if panel:
            shuffled = list(records)
            rng.shuffle(shuffled)
            expected_train, expected_val = shuffled[:int(.8 * len(shuffled))], shuffled[int(.8 * len(shuffled)):]
        else:
            active = [r for r in records if r["label"] == 1]
            inactive = [r for r in records if r["label"] == 0]
            rng.shuffle(active)
            rng.shuffle(inactive)
            na, ni = int(.8 * len(active)), int(.8 * len(inactive))
            expected_train, expected_val = active[:na] + inactive[:ni], active[na:] + inactive[ni:]
            rng.shuffle(expected_train)
            rng.shuffle(expected_val)
        assert [id(records[i]) for i in train] == list(map(id, expected_train))
        assert [id(records[i]) for i in val] == list(map(id, expected_val))
        assert set(train).isdisjoint(val) and sorted([*train, *val]) == list(range(len(records)))


def test_metrics_known_values_ties_extremes_and_empty_strata():
    y, scores = np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])
    assert discrimination(y, scores)["auroc"] == pytest.approx(.75)
    assert discrimination(y, scores)["average_precision"] == pytest.approx(5 / 6)
    assert discrimination(y, np.ones(4))["auroc"] == .5
    assert discrimination(y, np.ones(4))["average_precision"] == .5
    for labels in (np.array([]), np.ones(4), np.zeros(4)):
        report = discrimination(labels, np.zeros(len(labels)))
        assert report["auroc"] is report["average_precision"] is None
    extreme, _ = binary_report(np.array([1, 0]), np.array([-1000., 1000.]), temperature=.5, training_labels=y)
    assert extreme["log_loss"] == 2000
    assert np.isfinite(list(sigmoid(np.array([-1000, 1000])))).all()
    zero, _ = binary_report(y, np.zeros(4), temperature=.95, training_labels=y)
    assert zero["brier"] == zero["training_prevalence_baseline_brier"] == .25
    assert zero["log_loss"] == pytest.approx(np.log(2))


def test_metrics_match_sklearn_with_ties():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(13)
    for _ in range(5):
        y, scores = rng.integers(0, 2, 100), rng.integers(-5, 5, 100)
        values = discrimination(y, scores)
        assert values["auroc"] == pytest.approx(sklearn.roc_auc_score(y, scores))
        assert values["average_precision"] == pytest.approx(sklearn.average_precision_score(y, scores))


def test_masks_exclude_unobserved_and_single_class_rank_metrics():
    logits = np.array([[-1., 3., 4.], [1., -2., 5.]])
    targets = np.array([[0, 1, 0], [1, 1, 0]])
    mask = np.array([[1, 1, 0], [1, 1, 0]])
    metrics, rows, bins = prediction_report(logits, targets, mask, targets, mask, temperature=.8, outputs=["both", "single", "empty"])
    assert metrics["stored_temperature"]["macro_auroc"] == 1
    assert metrics["stored_temperature"]["outputs_with_auroc"] == 1
    assert next(r for r in rows if r["output"] == "single")["auroc"] is None
    assert next(r for r in rows if r["output"] == "empty")["brier"] is None
    assert sum(r["n"] for r in bins) == 8  # Four observed labels, two probability modes.
    with pytest.raises(ValueError):
        prediction_report(logits * np.nan, targets, mask, targets, mask, temperature=.8, outputs=["a", "b", "c"])


def test_reliability_boundaries_and_explicit_similarity_overlap():
    rows = reliability(np.array([0, 1, 1]), np.array([0., .5, 1.]))
    assert rows[0]["n"] == rows[5]["n"] == rows[9]["n"] == 1
    assert rows[1]["mean_probability"] is None
    audit, nearest = cross_split_similarity(["KALAGRLKAA", "KGLAGRLKAA"], ["KALAGRLKAA", "KALAGRLKAA", "DEDEDEDE"])
    assert audit["validation_rows_with_exact_training_sequence"] == 2
    assert audit["validation_duplicate_rows"] == 1
    assert nearest[:2] == [1, 1]


@pytest.mark.parametrize("outputs", [1, 10])
def test_real_production_forward_with_pinned_fake_backbone(tmp_path, monkeypatch, outputs):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from amp_challenge_2027.score import ActivityScorer, PanelScorer

    state = {"dense.weight": torch.eye(480), "dense.bias": torch.zeros(480),
             "classifier.weight": torch.ones(outputs, 480) / 480, "classifier.bias": torch.zeros(outputs)}
    torch.save(state, tmp_path / "classifier.pt")
    requested = []

    class FakeESM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=480, _commit_hash="fixed-revision")

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 480))

    class FakeTokenizer:
        def __call__(self, sequences, **kwargs):
            ids = torch.ones(len(sequences), 4, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids}

    def load_model(name, **kwargs):
        requested.append(kwargs["revision"])
        return FakeESM()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name, **kw: FakeTokenizer())
    cell = {"artifact": {"directory": str(tmp_path), "stem": "classifier", "tensor_sha256": tensor_fingerprint(state)},
            "config": {"esm_model": evaluate.baseline.BACKBONE}, "revision": "fixed-revision",
            "outputs": ["active"] if outputs == 1 else evaluate.baseline.PANEL_GENERA}
    predictor = evaluate.PinnedPredictor(cell, device="cpu")
    seqs = ["KALAGRLKAA", "DEDEDEDE"]
    logits = predictor.predict(seqs, 1)
    if outputs == 1:
        deployed = ActivityScorer(predictor.model, predictor.tokenizer, "cpu", temperature=.95)
        np.testing.assert_allclose(sigmoid(logits[:, 0] / .95), deployed.score(seqs), rtol=1e-6)
    else:
        deployed = PanelScorer(predictor.model, predictor.tokenizer, "cpu", evaluate.baseline.PANEL_GENERA,
                               frozenset(evaluate.baseline.MDR_PANEL_GENERA), temperature=.95)
        np.testing.assert_allclose(sigmoid(logits / .95), deployed._probs(seqs), rtol=1e-6)
    assert requested == ["fixed-revision"]
    cell["revision"] = "wrong-revision"
    with pytest.raises(ValueError, match="revision/dimension"):
        evaluate.PinnedPredictor(cell, device="cpu")
