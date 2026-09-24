import subprocess

from audit_authorship_readiness import inventory, run


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
