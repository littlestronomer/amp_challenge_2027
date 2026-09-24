import hashlib
import subprocess

import pytest
from collect_training_lineage import collect, csv_summary, local_path


def test_collection_distinguishes_snapshot_identity_from_training_link(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "--allow-empty", "-qm", "fixture"], cwd=root, check=True)
    data = root / "data/processed/generative.csv"
    data.parent.mkdir(parents=True)
    data.write_text("sequence,source_dbs\nACDEFGHI,DRAMP|DBAASP\nACDEFGHI,DRAMP\n")
    (root / "data/antibacterial.fasta").write_text(">reference\nACDE\nFGHI\n")
    head = root / "checkpoint/generator/model.pt"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"not a pickle; hash only")
    copy = root / "checkpoint/original/model.pt"
    copy.parent.mkdir()
    copy.write_bytes(head.read_bytes())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = {"models": [{"id": "generator", "artifact": "checkpoint/generator/model.pt",
                          "artifact_sha256": digest(head),
                          "candidate_data": "data/processed/generative.csv",
                          "candidate_data_sha256": digest(data),
                          "training_link_status": "historical_declaration_only"}]}
    baseline = {"validated_commit": "reported", "runtime_files": {
        "checkpoint/generator/model.pt": digest(head)}, "outputs": {"top.fasta": "0" * 64}}
    result = collect(root, tmp_path / "out", ledger, baseline)
    assert result["runtime_matches_validated_baseline"]
    assert result["models"][0]["candidate_data"]["status"] == "match"
    assert result["models"][0]["candidate_data"]["summary"]["same_sequence_set_as_reference"]
    assert not result["historical_training_provenance_verified"]
    assert not result["models"][0]["historical_training_provenance_verified"]
    assert result["byte_identical_model_copies"][0]["path"] == "checkpoint/original/model.pt"
    assert not result["byte_identical_model_copies"][0]["training_data_link_verified"]
    assert result["fresh_clone_outputs"][0]["status"] == "missing"
    assert csv_summary(data)["declared_source_counts"] == {"source_dbs": {"DBAASP": 1, "DRAMP": 2}}
    assert csv_summary(data)["unique_sequences"] == 1
    assert "ACDEFGHI" not in (tmp_path / "out/lineage.json").read_text()
    data.write_text("sequence,source_dbs\nYYYYYYYY,DRAMP\n")
    head.write_bytes(b"changed checkpoint")
    changed = collect(root, tmp_path / "changed", ledger, baseline)
    assert not changed["runtime_matches_validated_baseline"]
    assert changed["models"][0]["candidate_data"]["status"] == "mismatch"
    assert not changed["models"][0]["candidate_data"]["summary"]["same_sequence_set_as_reference"]
    with pytest.raises(ValueError, match="inputs changed"):
        collect(root, tmp_path / "out", ledger, baseline)


def test_collector_rejects_output_over_inputs_and_escape_paths(tmp_path):
    with pytest.raises(ValueError, match="input/evidence"):
        collect(tmp_path, tmp_path / "checkpoint/out", {}, {})
    outside = tmp_path.parent / "outside-file"
    (tmp_path / "link").symlink_to(outside)
    for name in ("../outside-file", "link", str(outside)):
        with pytest.raises(ValueError, match="outside repository"):
            local_path(tmp_path, name)
