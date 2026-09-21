"""Small, local checks for evidence inventory and top-100 audit contracts."""

from __future__ import annotations

import csv
import hashlib
import json

import pytest
import report_top100_readiness as top100

from amp_challenge_2027.evidence_inventory import collect_evidence


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_config(root, *, required=True):
    return {
        "kind": "competition_evidence_inventory_v1",
        "sources": [{
            "id": "fixture", "kind": "fixture", "root": "evidence",
            "required": required, "inspection_status": "unknown",
            "files": ["result.json"],
            "marker": {"name": "complete.json", "files": ["result.json"]},
        }],
    }


def test_evidence_inventory_verifies_producer_marker_and_hashes(tmp_path):
    repo = tmp_path / "repo"
    evidence = repo / "evidence"
    result = evidence / "result.json"
    _write_json(result, {"value": 7})
    _write_json(evidence / "complete.json", {"files": {"result.json": _sha256(result)}})
    config = repo / "config.json"
    _write_json(config, _inventory_config(repo))

    out = repo / "inventory"
    assert collect_evidence(repo, config, out) == 0
    inventory = json.loads((out / "inventory.json").read_text())
    source = inventory["sources"][0]
    assert source["marker"]["status"] == "verified"
    assert source["files"][0]["status"] == "verified"
    marker = json.loads((out / "complete.json").read_text())
    assert marker["files"]["inventory.json"] == _sha256(out / "inventory.json")


def test_evidence_inventory_fails_closed_on_bad_marker(tmp_path):
    repo = tmp_path / "repo"
    evidence = repo / "evidence"
    result = evidence / "result.json"
    _write_json(result, {"value": 7})
    _write_json(evidence / "complete.json", {"files": {"result.json": "0" * 64}})
    config = repo / "config.json"
    _write_json(config, _inventory_config(repo))

    out = repo / "inventory"
    assert collect_evidence(repo, config, out) == 2
    inventory = json.loads((out / "inventory.json").read_text())
    assert inventory["invalid_inputs"] is True
    assert inventory["sources"][0]["marker"]["status"] == "invalid"


def test_evidence_inventory_rejects_root_traversal(tmp_path):
    repo = tmp_path / "repo"
    config = repo / "config.json"
    _write_json(config, _inventory_config(repo))
    bad = json.loads(config.read_text())
    bad["sources"][0]["root"] = "../outside"
    _write_json(config, bad)

    with pytest.raises(ValueError, match="non-traversing"):
        collect_evidence(repo, config, repo / "inventory")


@pytest.mark.parametrize("required, expected_code", [(True, 2), (False, 0)])
def test_evidence_inventory_reports_missing_required_and_optional_roots(
    tmp_path, required, expected_code
):
    repo = tmp_path / "repo"
    config = repo / "config.json"
    _write_json(config, _inventory_config(repo, required=required))
    out = repo / "inventory"

    assert collect_evidence(repo, config, out) == expected_code
    inventory = json.loads((out / "inventory.json").read_text())
    assert inventory["required_sources_missing"] is required
    assert inventory["sources"][0]["root_status"] == "missing"


def test_evidence_inventory_rejects_duplicate_ids(tmp_path):
    repo = tmp_path / "repo"
    config = repo / "config.json"
    invalid = _inventory_config(repo)
    invalid["sources"].append(dict(invalid["sources"][0]))
    _write_json(config, invalid)

    with pytest.raises(ValueError, match="duplicate source id"):
        collect_evidence(repo, config, repo / "inventory")


def test_evidence_inventory_leaves_unknown_csv_uninterpreted(tmp_path):
    repo = tmp_path / "repo"
    evidence = repo / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "result.json").write_text("metric,value\nAUROC,0.99\n")
    config = repo / "config.json"
    _write_json(config, _inventory_config(repo, required=False))

    out = repo / "inventory"
    assert collect_evidence(repo, config, out) == 0
    inventory = json.loads((out / "inventory.json").read_text())
    assert inventory["sources"][0]["files"][0]["status"] == "unverified"
    assert (out / "metrics.csv").read_text().count("\n") == 1


def test_evidence_inventory_is_deterministic_and_refuses_overwrite(tmp_path):
    repo = tmp_path / "repo"
    evidence = repo / "evidence"
    result = evidence / "result.json"
    _write_json(result, {"value": 7})
    _write_json(evidence / "complete.json", {"files": {"result.json": _sha256(result)}})
    config = repo / "config.json"
    _write_json(config, _inventory_config(repo))
    before = _sha256(result)

    first, second = repo / "first", repo / "second"
    assert collect_evidence(repo, config, first) == 0
    assert collect_evidence(repo, config, second) == 0
    assert _sha256(result) == before
    for name in ("inventory.json", "metrics.csv", "STATUS.md", "complete.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(FileExistsError):
        collect_evidence(repo, config, first)


def test_family_metric_parser_rejects_nonfinite_values(tmp_path):
    repo = tmp_path / "repo"
    evidence = repo / "evidence"
    evidence.mkdir(parents=True)
    result = evidence / "results.csv"
    fields = ["task", "architecture", "sequences", "families", "macro_auroc",
              "macro_average_precision", "macro_brier", "macro_log_loss",
              "macro_ece_10_equal_width", "auroc_ci_lower", "auroc_ci_upper"]
    result.write_text(",".join(fields) + "\nactivity,model,20,10,NaN,0.7,0.2,0.5,0.1,0.4,0.8\n")
    (evidence / "complete.json").write_text(json.dumps({"files": {"results.csv": _sha256(result)}}))
    config = repo / "config.json"
    source = _inventory_config(repo)["sources"][0]
    source.update(kind="family_benchmark_test", files=["results.csv"])
    source["marker"] = {"name": "complete.json", "files": ["results.csv"]}
    _write_json(config, {"kind": "competition_evidence_inventory_v1", "sources": [source]})

    out = repo / "inventory"
    assert collect_evidence(repo, config, out) == 2
    inventory = json.loads((out / "inventory.json").read_text())
    assert inventory["invalid_inputs"] is True
    assert inventory["sources"][0]["files"][0]["status"] == "invalid"


def _write_fasta(path, sequences):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">seq{i}\n{sequence}\n" for i, sequence in enumerate(sequences)))


def test_top100_audit_reports_score_coverage_and_bounded_subset_draws(tmp_path, monkeypatch):
    monkeypatch.setattr(top100, "LIBRARY_SIZE", 3)
    monkeypatch.setattr(top100, "TOP_K", 2)
    sequences = ["ACDEFGHIKL", "MNPQRSTVWY", "GGGGGAAAAA"]
    library = tmp_path / "library.fasta"
    top = tmp_path / "top.fasta"
    reference = tmp_path / "reference.fasta"
    _write_fasta(library, sequences)
    _write_fasta(top, sequences[:2])
    _write_fasta(reference, ["YYYYYYYYYY"])
    scores = tmp_path / "scores.csv"
    with scores.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "composite_score", "activity"])
        writer.writeheader()
        writer.writerows([
            {"sequence": sequences[0], "composite_score": "0.7", "activity": "0.8"},
            {"sequence": sequences[1], "composite_score": "0.6", "activity": "0.5"},
        ])

    rows, summary = top100.audit(library, top, reference, scores, None, draws=8, seed=12)
    assert len(rows) == 2
    assert summary["score_coverage_complete"] is True
    assert summary["score_summary"]["score_activity"]["coverage"] == 2
    assert summary["random_25_subset_diagnostic"]["subset_size"] == 2
    assert summary["random_25_subset_diagnostic"]["draws"] == 8
