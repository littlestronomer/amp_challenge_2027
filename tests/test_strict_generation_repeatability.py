import json

import pytest
from compare_strict_generation_runs import run
from experiment_utils import sha256

from amp_challenge_2027.evidence_inventory import collect_evidence


def _run(root, *, output_suffix="", score=b"rank,score\n1,1\n", weight=1):
    root.mkdir()
    files = {"library": b">a\nACDEFGHI\n", "top": b">a\nACDEFGHI\n", "top_scores": score}
    outputs = {}
    for name, contents in files.items():
        filename = {"library": "library.fasta", "top": "top.fasta", "top_scores": "top_scores.csv"}[name]
        path = root / filename
        path.write_bytes(contents)
        outputs[name] = {"path": filename, "sha256": sha256(path)}
    manifest = {"kind": "amp_generation_manifest_v2", "status": "complete",
                "recipe": {"weight": weight, "out_dir": f"output{output_suffix}"},
                "runtime": {"python": "3.x"}, "outputs": outputs}
    (root / "generation_manifest.json").write_text(json.dumps(manifest))


def test_strict_run_comparison_bundles_identical_outputs(tmp_path):
    _run(tmp_path / "one", output_suffix="1")
    _run(tmp_path / "two", output_suffix="2")
    out = run(tmp_path / "one", tmp_path / "two", tmp_path / "bundle")
    report = json.loads((out / "comparison.json").read_text())
    assert report["outputs_equal"] is True
    assert report["runtime_equal"] is True
    assert (out / "complete.json").is_file()


def test_strict_run_comparison_preserves_difference_report(tmp_path):
    _run(tmp_path / "one")
    _run(tmp_path / "two", score=b"rank,score\n1,0\n")
    with pytest.raises(ValueError, match="differ"):
        run(tmp_path / "one", tmp_path / "two", tmp_path / "bundle")
    report = json.loads((tmp_path / "bundle/comparison.json").read_text())
    assert report["outputs_equal"] is False


def test_strict_run_comparison_refuses_recipe_change(tmp_path):
    _run(tmp_path / "one")
    _run(tmp_path / "two", weight=2)
    with pytest.raises(ValueError, match="recipes differ"):
        run(tmp_path / "one", tmp_path / "two", tmp_path / "bundle")


def test_evidence_inventory_rechecks_both_manifests_and_output_hashes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo / "source1", output_suffix="1")
    _run(repo / "source2", output_suffix="2")
    bundle = run(repo / "source1", repo / "source2", repo / "strict-repeatability-v1")
    # The strict repeatability bundle itself is the configured source root.
    source_files = ["comparison.json", "run1/generation_manifest.json", "run2/generation_manifest.json",
                    "run1/library.fasta", "run2/library.fasta", "run1/top.fasta", "run2/top.fasta",
                    "run1/top_scores.csv", "run2/top_scores.csv", "complete.json"]
    config = repo / "config.json"
    config.write_text(json.dumps({"kind": "competition_evidence_inventory_v1", "sources": [{
        "id": "strict_repeatability", "kind": "strict_repeatability", "root": bundle.name,
        "required": True, "inspection_status": "already_inspected", "files": source_files,
        "marker": {"name": "complete.json", "type": "strict_repeatability",
                   "files": source_files[:-1], "runs": ["run1", "run2"]}}]}))
    assert collect_evidence(repo, config, repo / "evidence") == 0
    inventory = json.loads((repo / "evidence/inventory.json").read_text())
    assert inventory["sources"][0]["marker"]["status"] == "verified"
