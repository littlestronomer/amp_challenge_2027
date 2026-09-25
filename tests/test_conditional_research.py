import json
from pathlib import Path

import numpy as np
import pytest

from amp_challenge_2027.conditional_research import (
    raw_metrics,
    seal,
    summarize_pairs,
    verify,
    write_json,
)
from amp_challenge_2027.config import PANEL_GENERA


def test_raw_denominator_includes_repeats_reference_and_invalid():
    seqs = ["ACDEFGHIK", "ACDEFGHIK", "KLMNPQRST", "bad"]
    a = np.array([.9, .9, .9, np.nan])
    h = np.array([.1, .1, .1, np.nan])
    p = np.full((4, len(PANEL_GENERA)), .9)
    p[-1] = np.nan
    report = raw_metrics(seqs, a, h, p, {"KLMNPQRST"})
    assert report["raw_draws"] == 4
    assert report["joint_unique_count"] == 1
    assert report["joint_unique_yield_per_1000"] == 250
    assert report["raw_joint_pass_fraction"] == .5
    assert report["valid_unique_novel_fraction"] == .25
    assert report["raw_joint_pass_wilson95"][0] < .5 < report["raw_joint_pass_wilson95"][1]
    assert sum(row["n"] for row in report["risk_strata"]) == 3


def test_nonfinite_valid_prediction_is_not_silently_ignored():
    with pytest.raises(ValueError, match="probabilities"):
        raw_metrics(["ACDEFGHIK"], np.array([np.nan]), np.array([.1]),
                    np.ones((1, len(PANEL_GENERA))), set())


def _row(label, seed, yield_value=10):
    return {"label": label, "seed": seed, "raw_draws": 10000,
            "training_seed": seed, "training_arm": "conditional",
            "joint_unique_yield_per_1000": yield_value, "raw_activity_mean": .85,
            "panel_means": {g: .8 for g in PANEL_GENERA}, "raw_pairwise_distance_256": .7,
            "valid_unique_novel_fraction": .9}


def test_advancement_requires_all_three_paired_seeds_and_preserved_panel():
    rows = [_row(label, seed, 10 if label == "base" else 20)
            for label in ("base", "candidate") for seed in (42, 43, 44)]
    assert summarize_pairs(rows, "base")[0]["advance_to_50000"]
    assert not summarize_pairs(rows[:-1], "base")[0]["advance_to_50000"]
    rows[-1]["panel_means"][PANEL_GENERA[0]] = .7
    assert not summarize_pairs(rows, "base")[0]["advance_to_50000"]


def test_sampling_replicates_cannot_claim_independent_training_seeds():
    rows = [_row(label, seed, 10 if label == "base" else 20)
            for label in ("base", "candidate") for seed in (42, 43, 44)]
    for row in rows:
        row["training_seed"] = 42
    report = summarize_pairs(rows, "base")[0]
    assert not report["advance_to_50000"]
    assert report["paired_seed_bootstrap95"] is None


def test_artifact_tampering_fails_closed(tmp_path):
    write_json(tmp_path / "run.json", {"seed": 42})
    seal(tmp_path, ["run.json"])
    verify(tmp_path, ["run.json"])
    write_json(tmp_path / "run.json", {"seed": 43})
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)


def test_sampling_save_load_and_condition_request(tmp_path):
    torch = pytest.importorskip("torch")
    from amp_challenge_2027.assay_conditioning import fit_condition_schema
    from amp_challenge_2027.conditional_research import sample_experiment
    from amp_challenge_2027.model import DecoderConfig, build_model, save_model

    torch.set_num_threads(1)
    conditions = [{"endpoint": "MIC", "target": "E. coli", "value_upper": 16.}]
    schema = fit_condition_schema([{"split": "train", "conditions": conditions}])
    model, _ = build_model(DecoderConfig(hidden_size=16, num_heads=2, num_layers=2, dropout=0., assay_schema=schema))
    source = tmp_path / "checkpoint"
    save_model(model, source)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"conditions": conditions}))
    a, b = tmp_path / "first", tmp_path / "second"
    for out in (a, b):
        sample_experiment(source, out, label="conditional", draws=5, batch_size=2, device="cpu", conditions=request)
    assert (a / "raw.fasta").read_bytes() == (b / "raw.fasta").read_bytes()
    assert json.loads((a / "summary.json").read_text())["raw_draws"] == 5
    verify(a, ["draws.csv", "raw.fasta"])


def test_compare_offline_with_frozen_stub_oracles(tmp_path):
    from amp_challenge_2027.conditional_research import compare_experiments, write_csv

    class Oracles:
        identity = {"kind": "test_stub_not_biological"}

        def score(self, sequences):
            return np.full(len(sequences), .9), np.full(len(sequences), .1), np.full((len(sequences), len(PANEL_GENERA)), .8)

    reference = tmp_path / "reference.fasta"
    reference.write_text(">ref\nRRRRRRRRRR\n")
    run = tmp_path / "draws"
    run.mkdir()
    write_json(run / "run.json", {"kind": "conditional_raw_draws_v1", "label": "base", "seed": 42,
                                  "conditions": [], "sampling": {"draws": 2}})
    (run / "raw.fasta").write_text(">draw_1\nACDEFGHIK\n>draw_2\nACDEFGHIK\n")
    write_csv(run / "draws.csv", [{"draw_id": i, "sequence": "ACDEFGHIK"} for i in (1, 2)])
    seal(run, ["run.json", "draws.csv", "raw.fasta"])
    out = tmp_path / "comparison"
    result = compare_experiments([run], out, baseline="base", reference=reference, scorer=Oracles())
    assert result["rows"][0]["joint_unique_yield_per_1000"] == 500
    assert result["decisions"] == []
    verify(out, ["REPORT.md", "comparison.csv"])


def test_cli_help_has_all_stages():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/conditional_generator.py"
    output = subprocess.check_output([sys.executable, str(script), "--help"], text=True)
    assert "fetch,prepare,train,sample,compare" in output


def test_research_selection_preserves_production_scores_and_seed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from amp_challenge_2027 import pipeline, score, select
    from amp_challenge_2027.conditional_research import select_unchanged
    from amp_challenge_2027.config import MDR_PANEL_GENERA

    sequences = ["ACDEFGHIK", "KLMNPQRST", "RSTVWYACDE", "DEFGHIKLMN"]
    reference = ["RRRRRRRRRR", "KKKKKKKKKK"]
    activity = np.asarray([.71, .85, .64, .94], dtype=np.float32)
    panel = np.random.default_rng(42).uniform(0, 1, (4, len(PANEL_GENERA))).astype(np.float32)
    risk = np.asarray([.1, .8, .6, .2])
    mdr = [i for i, g in enumerate(PANEL_GENERA) if g in MDR_PANEL_GENERA]

    class Precision:
        def __init__(self, _refs, esm_model=None, **_kwargs):
            self._esm_model = esm_model
            self._model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fixture-revision"))

        def _embed(self, seqs):
            return np.ones((len(seqs), 2))

        def score(self, _seqs):
            return np.asarray([.82, .78, .87, .91], dtype=np.float32)

    monkeypatch.setattr(pipeline, "_torch_ready", lambda: True)
    for module in (pipeline, score):
        monkeypatch.setattr(module, "PrecisionProxyScorer", Precision)
    monkeypatch.setattr(score.ActivityScorer, "load", lambda **_: SimpleNamespace(score=lambda _: activity))
    monkeypatch.setattr(score.PanelScorer, "load", lambda **_: SimpleNamespace(
        breadth=lambda _: (panel > .5).mean(1).astype(np.float32),
        mdr_breadth=lambda _: (panel[:, mdr] > .5).mean(1).astype(np.float32)))
    production = pipeline.build_composite_scorer(reference, seed=44)
    expected, _ = production.score(sequences)

    def capture(clean, **kwargs):
        assert clean == sequences
        assert kwargs["seed"] == 44
        assert kwargs["max_novelty_candidates"] == 2000
        np.testing.assert_array_equal(kwargs["scores"], expected)
        return SimpleNamespace(library=clean, top=clean)

    monkeypatch.setattr(select, "select_library_and_top", capture)
    result = select_unchanged(sequences, activity, risk, panel, reference, tmp_path, "cpu", seed=44)
    assert not result["library_complete"]
    assert json.loads((tmp_path / "selection.json").read_text())["seed"] == 44
