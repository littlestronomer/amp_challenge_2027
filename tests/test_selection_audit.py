"""Cache-only iteration tests. No pretrained-model inference or downloads."""

from __future__ import annotations

import csv
import json

import analyze_selection as analyze
import compare_top100 as compare
import numpy as np
import pytest
import sweep_top100_selection as sweep
from experiment_utils import mark_files, sha256, write_json
from selection_cache import load_source
from test_top100_comparison import fixture_run

from amp_challenge_2027.select import select_library_and_top
from amp_challenge_2027.selection_audit import (
    distribution,
    fit_normalization,
    normalized_score,
    paired_deltas,
    rank_correlation,
    selection_stages,
)


def cache_fixture(tmp_path, monkeypatch, score_fn=None):
    argv, _, calls = fixture_run(tmp_path, monkeypatch)
    if score_fn is not None:
        monkeypatch.setattr(compare.AuditScorers, "score_library", lambda _self, sequences: score_fn(sequences))
    compare.main(argv)
    source = tmp_path / "out"
    write_json(source / "backbones.json", {name: "test-only-revision" for name in ("activity", "panel", "hemolysis", "precision")})
    np.save(source / "reference_embeddings.npy", np.zeros((1, 4)))
    mark_files(source, "reference_cache.json", ["backbones.json", "reference_embeddings.npy"])
    return ["--source", str(source), "--reference", str(tmp_path / "reference.fasta"),
            "--out", str(tmp_path / "audit")], calls


def test_six_p0_and_six_p1_cells_replay_preserve_and_resume(tmp_path, monkeypatch):
    argv, calls = cache_fixture(tmp_path, monkeypatch)
    original = {p: sha256(p) for p in (tmp_path / "out").rglob("*") if p.is_file()}
    sweep.main(argv + ["--list"])
    assert not (tmp_path / "audit").exists()
    sweep.main(argv)
    assert calls == {"init": 1, "score": 6, "risk": 6}  # fixture only; zero extra models
    assert {p: sha256(p) for p in original} == original
    for cell in (tmp_path / "audit/P0").glob("*/seed*"):
        source = tmp_path / "out" / cell.parent.name / cell.name
        assert (cell / "top.fasta").read_bytes() == (source / "top.fasta").read_bytes()
        assert not (cell / "library.fasta").exists()  # reference old immutable bytes, do not rewrite
    anchor = json.loads((tmp_path / "audit/normalization_anchor.json").read_text())["components"]
    for cell in (tmp_path / "audit/P1").glob("*/seed*"):
        assert json.loads((cell / "normalization.json").read_text()) == anchor
    assert (tmp_path / "audit/P1/hybrid/seed42/top.fasta").read_bytes() == (tmp_path / "out/hybrid/seed42/top.fasta").read_bytes()
    with (tmp_path / "audit/results.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 12
    with (tmp_path / "audit/policy_paired_deltas.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 6
    before = (tmp_path / "audit/results.csv").read_bytes()
    monkeypatch.setattr(analyze, "selection_stages", lambda *_a, **_k: pytest.fail("complete cell must resume"))
    sweep.main(argv)
    assert (tmp_path / "audit/results.csv").read_bytes() == before


def test_normalization_change_alone_can_reverse_ranking():
    parts = {"a": np.array([0., 1.]), "b": np.array([1., 0.])}
    weights = {"a": 2., "b": 1.}
    local, _ = normalized_score(parts, weights, fit_normalization(parts))
    fixed = fit_normalization({"a": np.array([0., 10.]), "b": np.array([0., 1.])})
    anchored, _ = normalized_score(parts, weights, fixed)
    assert local.argmax() == 1 and anchored.argmax() == 0
    reordered, _ = normalized_score(dict(reversed(list(parts.items()))), weights, fixed)
    np.testing.assert_array_equal(reordered, anchored)


def test_end_to_end_p1_changes_candidate_selection_without_new_risk_inference(tmp_path, monkeypatch):
    def scores(sequences):
        size = len(sequences)
        activity = np.linspace(.2, .8, size)
        if "KRKAGLAGKRA" in sequences:  # The test fixture's candidate-library sentinel.
            activity = .45 + .1 * activity
        return {"activity": activity, "conformity": np.linspace(.8, .2, size),
                "precision": np.zeros(size), "panel": np.full((size, len(compare.PANEL_GENERA)), .6)}

    argv, calls = cache_fixture(tmp_path, monkeypatch, score_fn=scores)
    sweep.main(argv)
    for seed in (42, 43, 44):
        control = tmp_path / f"audit/P0/p3_s1/seed{seed}/top.fasta"
        candidate = tmp_path / f"audit/P1/p3_s1/seed{seed}/top.fasta"
        assert control.read_bytes() != candidate.read_bytes()
        summary = json.loads(candidate.with_name("summary.json").read_text())
        assert summary["hemo_risk_status"] == "incomplete_needs_scoring"
        assert summary["hemo_risk_mean"] is None
    assert calls["risk"] == 6  # Only the original comparison fixture made risk predictions.


def test_zero_variance_and_correlations_are_explicit():
    parts = {"constant": np.ones(4)}
    fitted = fit_normalization(parts)
    assert fitted["constant"]["denominator"] == 1
    combined, _ = normalized_score(parts, {"constant": 1}, fitted)
    np.testing.assert_array_equal(combined, np.zeros(4))
    assert distribution(parts["constant"])["tied_observation_fraction"] == 1
    assert distribution(parts["constant"])["fraction_at_one"] == 1
    assert distribution(np.array([]))["mean"] is None
    assert rank_correlation(np.ones(4), np.arange(4)) is None
    assert rank_correlation(np.array([1, 1, 2]), np.array([1, 1, 2])) == pytest.approx(1)
    assert rank_correlation(np.arange(4), -np.arange(4)) == pytest.approx(-1)


@pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan]), np.ones((2, 2))])
def test_invalid_fit_rejected(bad):
    with pytest.raises(ValueError):
        fit_normalization({"bad": bad})


def test_funnel_matches_production_and_rejects_underfilled(tmp_path, monkeypatch):
    cache_fixture(tmp_path, monkeypatch)
    _, cells, reference = load_source(tmp_path / "out", tmp_path / "reference.fasta")
    for cell in cells:
        stages = selection_stages(cell["sequences"], cell["combined"], set(reference), top_k=4, seed=42, shortlist=2000)
        production = select_library_and_top(cell["sequences"], reference_set=set(reference), scores=cell["combined"],
                                             top_k=4, library_size=12, seed=42, max_novelty_candidates=2000)
        assert [cell["sequences"][i] for i in stages["top"]] == production.top
        assert [len(stages[k]) for k in ("library", "plausible", "shortlist", "novel_shortlist", "top")] == sorted(
            (len(v) for v in stages.values()), reverse=True)
    with pytest.raises(ValueError, match="Underfilled"):
        selection_stages(cells[0]["sequences"], cells[0]["combined"], set(reference), top_k=4, seed=42, shortlist=2)


def test_changed_selection_has_no_fabricated_hemolysis_summary(tmp_path, monkeypatch):
    cache_fixture(tmp_path, monkeypatch)
    _, cells, reference = load_source(tmp_path / "out", tmp_path / "reference.fasta")
    cell = cells[0]
    unused = next(i for i, seq in enumerate(cell["sequences"]) if seq not in cell["top"])
    old = [cell["sequences"].index(seq) for seq in cell["top"][:3]]
    summary, details = analyze.top_report(cell, {"top": np.array([*old, unused])}, cell["combined"], reference)
    assert summary["hemo_risk_known_count"] == 3
    assert summary["hemo_risk_status"] == "incomplete_needs_scoring"
    assert summary["hemo_risk_mean"] is summary["hemo_risk_p75"] is summary["hemo_risk_max"] is None
    assert details[-1]["hemo_risk"] is None


@pytest.mark.parametrize("damage", ["bytes", "rows", "seeds", "nonfinite", "backbone", "reference"])
def test_source_corruption_fails_before_output(tmp_path, monkeypatch, damage):
    argv, _ = cache_fixture(tmp_path, monkeypatch)
    source, cell = tmp_path / "out", tmp_path / "out/hybrid/seed42"
    if damage == "bytes":
        with (cell / "library.fasta").open("a") as handle:
            handle.write("\n")
    elif damage == "rows":
        with np.load(cell / "scores.npz") as data:
            scores = {key: data[key][::-1] for key in data.files}
        np.savez_compressed(cell / "scores.npz", **scores)
        mark_files(cell, "complete.json", list(json.loads((cell / "complete.json").read_text())["files"]))
    elif damage == "seeds":
        manifest = json.loads((source / "run.json").read_text())
        manifest["cells"][0]["seed"] = 99
        write_json(source / "run.json", manifest)
    elif damage == "nonfinite":
        with np.load(cell / "scores.npz") as data:
            scores = {key: data[key] for key in data.files}
        scores["panel"][0, 0] = np.nan
        np.savez_compressed(cell / "scores.npz", **scores)
        mark_files(cell, "complete.json", list(json.loads((cell / "complete.json").read_text())["files"]))
    elif damage == "backbone":
        write_json(source / "backbones.json", {"activity": "changed"})
    else:
        (tmp_path / "reference.fasta").write_text(">changed\nDEDEDEDE\n")
    with pytest.raises(ValueError):
        sweep.main(argv)
    assert not (tmp_path / "audit").exists()


def test_parity_failure_stops_before_p1(tmp_path, monkeypatch):
    argv, _ = cache_fixture(tmp_path, monkeypatch)
    original = analyze.selection_stages

    def reordered(*args, **kwargs):
        stages = original(*args, **kwargs)
        stages["top"] = stages["top"][::-1]
        return stages

    monkeypatch.setattr(analyze, "selection_stages", reordered)
    with pytest.raises(ValueError, match="byte-identical"):
        sweep.main(argv)
    assert not (tmp_path / "audit/P1").exists()


def test_interruption_resumes_only_completed_cells(tmp_path, monkeypatch):
    argv, _ = cache_fixture(tmp_path, monkeypatch)
    original, counter = analyze.selection_stages, {"n": 0}

    def interrupted(*args, **kwargs):
        counter["n"] += 1
        if counter["n"] == 3:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(analyze, "selection_stages", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        sweep.main(argv)
    sweep.main(argv)
    assert counter["n"] == 13  # 12 cells plus the failed call, no rerun of completed work


def test_replay_only_and_changed_recipe_rejected(tmp_path, monkeypatch):
    argv, _ = cache_fixture(tmp_path, monkeypatch)
    analyze.main(argv)
    assert not (tmp_path / "audit/P1").exists()
    with pytest.raises(ValueError, match="inputs changed"):
        sweep.main(argv)
    with pytest.raises(ValueError, match="separate"):
        sweep.main(argv + ["--out", str(tmp_path / "out/nested")])
    cached = tmp_path / "audit/P0/hybrid/seed42/top.fasta"
    cached.write_text(">bad\nKALAGRLKAA\n")
    with pytest.raises(ValueError, match="changed"):
        analyze.main(argv)


def test_paired_comparisons_reject_duplicate_seed_cells():
    rows = [{"case": case, "seed": 42, "policy": "P0", "top_sequences": [case], "activity_mean": i}
            for i, case in enumerate(("hybrid", "p3_s1"))]
    assert paired_deltas(rows, "policy")[0]["activity_mean"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        paired_deltas(rows + [rows[0]], "policy")


def test_incomplete_seed_risk_does_not_become_a_three_seed_mean():
    rows = [{"case": case, "seed": seed, "policy": "P1", "hemo_risk_mean": .4 if seed == 42 else None}
            for case in ("hybrid", "p3_s1") for seed in (42, 43, 44)]
    for row in analyze.aggregate(rows):
        assert row["n_seeds"] == 1 and row["n_seeds_expected"] == 3
        assert row["mean"] is None and row["std"] is None


def test_completed_summary_tampering_is_not_overwritten(tmp_path, monkeypatch):
    argv, _ = cache_fixture(tmp_path, monkeypatch)
    sweep.main(argv)
    (tmp_path / "audit/results.csv").write_text("corrupt summary\n")
    with pytest.raises(ValueError, match="changed"):
        sweep.main(argv)
