"""CPU-only tests for immutable paired ranking and strict audit failures."""

from __future__ import annotations

import json
from types import SimpleNamespace

import compare_top100 as compare
import numpy as np
import pytest
from experiment_utils import mark_files, sha256, write_json

from amp_challenge_2027.data import iter_fasta, write_fasta
from amp_challenge_2027.pipeline import DEFAULT_WEIGHTS


def fixture_run(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "LIBRARY_SIZE", 12)
    monkeypatch.setattr(compare, "TOP_K", 4)
    reference = tmp_path / "reference.fasta"
    write_fasta(["DEDEDEDE"], reference)
    roots = {name: tmp_path / name for name in ("checkpoint-screen", "checkpoint-confirm", "blend-screen", "blend-confirm")}
    for name, root in roots.items():
        seeds = [42] if name.endswith("screen") else [43, 44]
        blend = name.startswith("blend")
        case = "p3_s1" if blend else "hybrid"
        run = {"kind": "fixed_ratio_blend_v1" if blend else "checkpoint_sweep_v1", "seeds": seeds,
               "library_size": 12, "reference_sha256": sha256(reference),
               "secondary_recipe": {"reference_sha256": sha256(reference)},
               "ratios": [[3, 1]], "primary_case": "ckpt_epoch58", "cases": [{"name": "hybrid", "secondary": "conditioned"}]}
        write_json(root / "run.json", run)
        for seed in seeds:
            cell = root / case / f"seed{seed}"
            aa = "ACDEFGHIKLMNPQRSTVWY"
            sequences = [f"K{aa[i]}LA{aa[(i+seed)%20]}GRLKAA" for i in range(12)]
            if blend:
                sequences[-1] = "KRKAGLAGKRA"
            write_fasta(sequences, cell / "library.fasta")
            write_json(cell / "metrics.json", {"FBD": 0.2})
            (cell / "metrics.csv").write_text("FBD\n0.2\n")
            mark_files(cell, "generation.json", ["library.fasta"])
            mark_files(cell, "evaluation.json", ["library.fasta", "metrics.json", "metrics.csv"])
    reward, hemo = tmp_path / "reward", tmp_path / "hemo"
    for directory, stem, task in ((reward, "classifier", "binary"), (reward, "classifier_panel", "panel"), (hemo, "classifier", "binary")):
        cfg = {"esm_model": compare.BACKBONE, "task": task, "unfreeze_layers": 0,
               "checkpoint_format": "head-only", "temperature": 0.9}
        if task == "panel":
            cfg["genera"] = compare.PANEL_GENERA
        write_json(directory / f"{stem}_config.json", cfg)
        (directory / f"{stem}.pt").write_bytes(b"test-only head, never loaded")
    monkeypatch.setattr(compare, "REWARD_DIR", reward)
    monkeypatch.setattr(compare, "REWARD_HEMO_DIR", hemo)
    calls = {"init": 0, "score": 0, "risk": 0}

    class FakeScorers:
        def __init__(self, *_args):
            calls["init"] += 1

        def score_library(self, sequences):
            calls["score"] += 1
            return score_fixture(len(sequences))

        def risk(self, top):
            calls["risk"] += 1
            return np.linspace(0.1, 0.4, len(top))

    monkeypatch.setattr(compare, "AuditScorers", FakeScorers)
    argv = ["--out", str(tmp_path / "out"), "--reference", str(reference), "--device", "cpu"]
    for name, root in roots.items():
        argv += [f"--{name}", str(root)]
    return argv, roots, calls


def score_fixture(size=12):
    return {"activity": np.linspace(0.2, 0.9, size), "conformity": np.linspace(0.1, 0.6, size),
            "precision": np.linspace(0.3, 0.8, size),
            "panel": np.linspace(0.1, 0.9, size * len(compare.PANEL_GENERA)).reshape(size, -1)}


def test_weights_match_current_production_but_exclude_hemo():
    assert compare.WEIGHTS == {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "safety"}
    assert "hemolysis" not in compare.WEIGHTS


def test_six_cells_resume_without_rescoring_and_preserve_source_bytes(tmp_path, monkeypatch):
    argv, roots, calls = fixture_run(tmp_path, monkeypatch)
    before = {p: sha256(p) for root in roots.values() for p in root.rglob("*") if p.is_file()}
    compare.main(argv + ["--list"])
    assert not (tmp_path / "out").exists()
    assert calls["init"] == 0
    compare.main(argv)
    assert calls == {"init": 1, "score": 6, "risk": 6}
    for root in roots.values():
        for path in root.glob("*/seed*/library.fasta"):
            destination = tmp_path / "out" / path.parent.parent.name / path.parent.name
            assert sha256(destination / "library.fasta") == sha256(path)
            top = [seq for _, seq in iter_fasta(destination / "top.fasta")]
            library = [seq for _, seq in iter_fasta(path)]
            assert len(set(top)) == 4 and set(top) <= set(library)
    assert {p: sha256(p) for p in before} == before
    first = (tmp_path / "out/results.csv").read_bytes()
    compare.main(argv)
    assert calls == {"init": 1, "score": 6, "risk": 6}
    assert (tmp_path / "out/results.csv").read_bytes() == first
    assert len((tmp_path / "out/paired_deltas.csv").read_text().splitlines()) == 4


@pytest.mark.parametrize("field,value", [("unfreeze_layers", 4), ("temperature", 0), ("temperature", float("nan")), ("task", "panel")])
def test_bad_classifier_metadata_fails_before_model_load(tmp_path, monkeypatch, field, value):
    argv, _, calls = fixture_run(tmp_path, monkeypatch)
    path = tmp_path / "reward/classifier_config.json"
    cfg = json.loads(path.read_text())
    cfg[field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="frozen head"):
        compare.main(argv)
    assert calls["init"] == 0


def test_missing_per_artifact_config_never_uses_shared_fallback(tmp_path, monkeypatch):
    argv, _, calls = fixture_run(tmp_path, monkeypatch)
    path = tmp_path / "reward/classifier_config.json"
    path.rename(path.with_name("config.json"))
    with pytest.raises(FileNotFoundError):
        compare.main(argv)
    assert calls["init"] == 0


def test_source_tampering_and_output_inside_source_rejected(tmp_path, monkeypatch):
    argv, roots, calls = fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="separate"):
        compare.main(argv + ["--out", str(roots["checkpoint-screen"] / "nested")])
    library = roots["checkpoint-screen"] / "hybrid/seed42/library.fasta"
    library.write_text(library.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        compare.main(argv)
    assert calls["init"] == 0


def test_incomplete_evaluation_marker_rejected(tmp_path, monkeypatch):
    argv, roots, _ = fixture_run(tmp_path, monkeypatch)
    mark_files(roots["checkpoint-screen"] / "hybrid/seed42", "evaluation.json", ["library.fasta"])
    with pytest.raises(ValueError, match="Invalid completion marker"):
        compare.main(argv)


def test_cached_scores_tampering_and_changed_recipe_rejected(tmp_path, monkeypatch):
    argv, _, calls = fixture_run(tmp_path, monkeypatch)
    compare.main(argv)
    with pytest.raises(ValueError, match="inputs changed"):
        compare.main(argv + ["--seeds", "43", "44"])
    (tmp_path / "out/hybrid/seed42/scores.npz").write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="changed"):
        compare.main(argv)
    assert calls["score"] == 6


def test_failed_risk_stage_resumes_saved_library_scores(tmp_path, monkeypatch):
    argv, _, calls = fixture_run(tmp_path, monkeypatch)
    original = compare.AuditScorers.risk
    monkeypatch.setattr(compare.AuditScorers, "risk", lambda self, top: np.full(len(top), np.nan))
    with pytest.raises(ValueError, match="hemolysis"):
        compare.main(argv)
    assert calls["score"] == 1
    assert not (tmp_path / "out/hybrid/seed42/complete.json").exists()
    monkeypatch.setattr(compare.AuditScorers, "risk", original)
    compare.main(argv)
    assert calls["score"] == 6  # interrupted first cell was not scored again


@pytest.mark.parametrize("damage", ["missing", "shape", "nan", "probability"])
def test_invalid_scores_fail_closed(damage):
    scores = score_fixture()
    if damage == "missing":
        del scores["activity"]
    elif damage == "shape":
        scores["panel"] = scores["panel"][:, :-1]
    elif damage == "nan":
        scores["conformity"][0] = np.nan
    else:
        scores["activity"][0] = 1.1
    with pytest.raises(ValueError):
        compare.validate_scores(scores, 12)


def test_audit_rejects_underfilled_or_nonnovel_top(monkeypatch):
    monkeypatch.setattr(compare, "TOP_K", 2)
    top = ["KALAGRLKAA", "KGLAGRLKAA"]
    with pytest.raises(ValueError, match="Incomplete"):
        compare.audit_top(top[:1], ["DEDEDEDE"], np.array([0.1]))
    with pytest.raises(ValueError, match="novelty"):
        compare.audit_top(top, top, np.array([0.1, 0.2]))


def test_pairwise_distance_risk_and_paired_delta_directions(monkeypatch):
    monkeypatch.setattr(compare, "TOP_K", 2)
    stats, _ = compare.audit_top(["KALAGRLKAA", "KGLAGRLKAA"], ["DEDEDEDE"], np.array([0.1, 0.3]))
    assert stats["hemo_risk_mean"] == pytest.approx(0.2)
    assert stats["top_mean_pairwise_distance"] == pytest.approx(0.1)
    rows = [{"case": "hybrid", "seed": 42, "top_sequences": ["A", "B"], "hemo_risk_mean": 0.2},
            {"case": "p3_s1", "seed": 42, "top_sequences": ["B", "C"], "hemo_risk_mean": 0.3}]
    delta = compare.paired_summary(rows)[0]
    assert delta["top_overlap_count"] == 1
    assert delta["hemo_risk_mean"] == pytest.approx(0.1)


def test_complete_frozen_head_validation():
    torch = pytest.importorskip("torch")
    state = {"dense.weight": torch.zeros(480, 480), "dense.bias": torch.zeros(480),
             "classifier.weight": torch.zeros(10, 480), "classifier.bias": torch.zeros(10)}
    compare.validate_head(state, 10)
    with pytest.raises(ValueError, match="keys"):
        compare.validate_head({k: v for k, v in state.items() if k != "dense.bias"}, 10)
    with pytest.raises(ValueError, match="tensor"):
        compare.validate_head(state, 1)
    state["dense.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="tensor"):
        compare.validate_head(state, 10)


def test_classifier_forward_ignores_unused_pooler():
    torch = pytest.importorskip("torch")
    from amp_challenge_2027.score import _build_activity_module

    class FakeESM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pooler = torch.nn.Linear(4, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 4),
                                   pooler_output=self.pooler(torch.ones(input_ids.shape[0], 4)))

    Classifier, _, _ = _build_activity_module()
    model = Classifier(4)
    model.esm = FakeESM()
    model.eval()
    ids = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        before = model(ids, ids)
        model.esm.pooler.weight.fill_(999)
        model.esm.pooler.bias.fill_(-999)
        torch.testing.assert_close(model(ids, ids), before, rtol=0, atol=0)


def test_real_head_loaders_and_scoring_with_tiny_local_backbones(tmp_path, monkeypatch):
    """Exercise production loading/forward/cache paths; no downloads or trained models."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from amp_challenge_2027 import score as score_module

    original_scorers = compare.AuditScorers
    fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(compare, "AuditScorers", original_scorers)
    monkeypatch.setattr(score_module, "REWARD_DIR", tmp_path / "reward")
    monkeypatch.setattr(score_module, "REWARD_HEMO_DIR", tmp_path / "hemo")
    for root, stem, count in ((tmp_path / "reward", "classifier", 1),
                              (tmp_path / "reward", "classifier_panel", len(compare.PANEL_GENERA)),
                              (tmp_path / "hemo", "classifier", 1)):
        state = {"dense.weight": torch.zeros(480, 480), "dense.bias": torch.zeros(480),
                 "classifier.weight": torch.zeros(count, 480), "classifier.bias": torch.zeros(count)}
        torch.save(state, root / f"{stem}.pt")
    revision = {"value": "test-only-backbone-revision"}

    class FakeESM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pooler = torch.nn.Linear(480, 480)
            self.config = SimpleNamespace(hidden_size=480, _commit_hash=revision["value"])

        def forward(self, input_ids, attention_mask):
            hidden = input_ids.float().unsqueeze(-1) + torch.arange(480).float() / 480
            return SimpleNamespace(last_hidden_state=hidden)

    class FakeTokenizer:
        def __call__(self, sequences, **_kwargs):
            ids = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
            for i, seq in enumerate(sequences):
                ids[i, :len(seq)] = torch.tensor([ord(aa) % 10 + 1 for aa in seq])
            return {"input_ids": ids, "attention_mask": (ids > 0).long()}

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *_a, **_k: FakeESM())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeTokenizer())
    reference = ["DEDEDEDE", "AKAKGAGA", "RLRLGAGA"]
    out = tmp_path / "model-cache"
    out.mkdir()
    artifacts = compare.classifier_inputs()
    scorers = original_scorers(reference, artifacts, out, "cpu")
    values = scorers.score_library(["KALAGRLKAA", "KGLAGRLKAA"])
    compare.validate_scores(values, 2)
    np.testing.assert_array_equal(values["activity"], [0.5, 0.5])
    np.testing.assert_array_equal(scorers.risk(["KALAGRLKAA"]), [0.5])
    cache_hash = sha256(out / "reference_embeddings.npy")
    original_scorers(reference, artifacts, out, "cpu")
    assert sha256(out / "reference_embeddings.npy") == cache_hash
    revision["value"] = "changed-test-revision"
    with pytest.raises(ValueError, match="revision changed"):
        original_scorers(reference, artifacts, out, "cpu")
