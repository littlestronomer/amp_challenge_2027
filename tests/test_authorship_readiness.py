import subprocess
import hashlib
import json

from audit_authorship_readiness import inventory, run, source_inventory


def test_inventory_distinguishes_committed_missing_and_local_only(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.json").write_text('{"temperature": 1}')
    subprocess.run(["git", "add", "tracked.json"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "local.json").write_text("{}")
    rows = inventory(tmp_path, ["tracked.json", "local.json", "missing.json"])
    assert rows[0]["matches_commit"]
    assert rows[1]["present"] and not rows[1]["committed"]
    assert not rows[2]["present"]
    (tmp_path / "tracked.json").write_text('{"temperature": 2}')
    assert not inventory(tmp_path, ["tracked.json"])[0]["matches_commit"]


def test_audit_does_not_certify_missing_release_or_data(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    result = run(root, tmp_path / "audit")
    assert result["eligibility"] == "not_certified"
    assert result["github"]["status"] == "unknown"
    assert all(not item["present"] for item in result["data_inventory"])
    assert (tmp_path / "audit/REPORT.md").is_file()


def test_source_inventory_checks_bytes_and_rejects_paths_outside_raw(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    content = b">record\nACDEFGHI\n"
    (raw / "sample.fasta").write_bytes(content)
    record = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    (tmp_path / "outside.fasta").write_bytes(content)
    (raw / "link.fasta").symlink_to(tmp_path / "outside.fasta")
    manifest = {"sample.fasta": record, "missing.fasta": record,
                "../../outside.fasta": record, "link.fasta": record,
                "malformed.fasta": {"sha256": "x"}}
    (raw / "sources.json").write_text(json.dumps(manifest))
    result = source_inventory(tmp_path)
    states = {r["path"]: r["status"] for r in result["entries"]}
    assert states == {"sample.fasta": "verified", "missing.fasta": "missing",
                      "../../outside.fasta": "invalid", "link.fasta": "invalid",
                      "malformed.fasta": "invalid"}
    assert result["training_lineage"] == "not_verified"
    (raw / "sample.fasta").write_bytes(b">record\nYYYYYYYY\n")
    changed = {r["path"]: r for r in source_inventory(tmp_path)["entries"]}
    assert changed["sample.fasta"]["status"] == "mismatch"
    (raw / "sample.fasta").write_bytes(content)
    manifest["sample.fasta"] = {**record, "size_bytes": len(content) + 1}
    (raw / "sources.json").write_text(json.dumps(manifest))
    changed = {r["path"]: r for r in source_inventory(tmp_path)["entries"]}
    assert changed["sample.fasta"]["status"] == "mismatch"


def test_source_inventory_handles_missing_and_invalid_manifest(tmp_path):
    assert source_inventory(tmp_path)["status"] == "missing"
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    for content in ("not json", "[]"):
        (raw / "sources.json").write_text(content)
        assert source_inventory(tmp_path)["status"] == "invalid"
