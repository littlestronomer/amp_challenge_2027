"""Inventory release gaps without claiming eligibility or publishing anything.

Run on the training machine to expose files present only in its working tree.
No model loading, inference, submission, or Git mutation is performed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from experiment_utils import REPO_ROOT, mark_files, prepare_run, sha256, write_json
from fetch_data import DRAMP_DIRECT_SOURCES

RELEASE_FILES = [
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "docs/ABSTRACT.md",
    "docs/DATA_DISCLOSURE.md",
    "docs/SUBMISSION_IDENTITY.json",
    "data/antibacterial.fasta",
    "checkpoint/generator/config.json",
    "checkpoint/generator/model.pt",
    "checkpoint/generator_blend/config.json",
    "checkpoint/generator_blend/model.pt",
    "checkpoint/reward/classifier.pt",
    "checkpoint/reward/classifier_config.json",
    "checkpoint/reward/classifier_panel.pt",
    "checkpoint/reward/classifier_panel_config.json",
    "checkpoint/reward_hemo/classifier.pt",
    "checkpoint/reward_hemo/classifier_config.json",
]
DATA_FILES = [
    "data/raw/sources.json",
    "data/processed/generative.csv",
    "data/processed/activity_labels_full.csv",
    "data/processed/hemolysis_labels.csv",
]


def source_inventory(root: Path) -> dict:
    """Check declared downloads locally; matching bytes do not prove training use."""
    raw = (root / "data/raw").resolve()
    manifest = raw / "sources.json"
    if not manifest.is_file():
        return {"status": "missing", "entries": []}
    try:
        records = json.loads(manifest.read_text())
        if not isinstance(records, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, OSError) as exc:
        return {"status": "invalid", "reason": str(exc), "entries": []}
    entries = []
    # fetch_dramp stores files in raw/dramp, but its registry keys are logical
    # source IDs (e.g. dramp-general/general_amps.fasta), not relative paths.
    download_paths = {
        f"{source}/{info['dest']}": Path("dramp") / info["dest"]
        for source, info in DRAMP_DIRECT_SOURCES.items()
    }
    for name, declared in sorted(records.items()):
        row = {"path": name, "status": "invalid", "declared": declared}
        relative = download_paths.get(name, Path(name))
        row["local_path"] = str(Path("data/raw") / relative)
        path = (raw / relative).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(raw):
            row["reason"] = "Source path escapes data/raw"
        elif not isinstance(declared, dict) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", str(declared.get("sha256", ""))
        ) or type(declared.get("size_bytes")) is not int or declared["size_bytes"] < 0:
            row["reason"] = "Missing or malformed SHA-256/size declaration"
        elif not path.is_file():
            row["status"] = "missing"
        else:
            row["actual_sha256"] = sha256(path)
            row["actual_bytes"] = path.stat().st_size
            row["status"] = (
                "verified" if row["actual_sha256"] == declared["sha256"].lower()
                and row["actual_bytes"] == declared["size_bytes"] else "mismatch"
            )
        entries.append(row)
    return {"status": "parsed", "entries": entries, "training_lineage": "not_verified"}


def inventory(root: Path, names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        path = root / name
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", name], cwd=root, capture_output=True
            ).returncode
            == 0
        )
        committed = (
            subprocess.run(
                ["git", "cat-file", "-e", f"HEAD:{name}"], cwd=root, capture_output=True
            ).returncode
            == 0
        )
        present = path.is_file()
        matches = False
        if present and committed:
            matches = (
                subprocess.run(["git", "diff", "--quiet", "HEAD", "--", name], cwd=root).returncode
                == 0
            )
        rows.append(
            {
                "path": name,
                "present": present,
                "tracked": tracked,
                "committed": committed,
                "matches_commit": matches,
                "sha256": sha256(path) if present else None,
                "bytes": path.stat().st_size if present else None,
            }
        )
    return rows


def run(root: Path, out: Path, github_repo: str | None = None) -> dict:
    root, out = root.resolve(), out.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    artifacts = inventory(root, RELEASE_FILES)
    data = inventory(root, DATA_FILES)
    sources = source_inventory(root)
    visibility = {"status": "unknown", "reason": "Use --github-repo to inspect GitHub visibility"}
    if github_repo:
        try:
            response = subprocess.run(
                ["gh", "repo", "view", github_repo, "--json", "isPrivate,defaultBranchRef,url"],
                capture_output=True,
                text=True,
                check=True,
            )
            visibility = {"status": "checked", **json.loads(response.stdout)}
        except (OSError, subprocess.CalledProcessError) as exc:
            visibility = {"status": "unknown", "reason": str(exc)}
    gaps = [
        f"Release file missing or differs from committed version: {r['path']}"
        for r in artifacts
        if not (r["present"] and r["committed"] and r["matches_commit"])
    ]
    if sources["status"] != "parsed":
        gaps.append(f"Download source manifest: {sources['status']}")
    gaps.extend(
        f"Download source {row['status']}: {row['path']}"
        for row in sources["entries"] if row["status"] != "verified"
    )
    if visibility.get("isPrivate"):
        gaps.append("Repository is private: full co-authorship requirements require public access")
    if visibility.get("defaultBranchRef", {}).get("name") not in (None, branch):
        gaps.append(
            "Working branch differs from GitHub default; designate and validate the exact release branch"
        )
    # Missing source records cannot prove that proprietary data were or were not used.
    gaps.extend(
        [
            "Training-data provenance needs snapshot-to-checkpoint reconciliation and source-terms review",
            "No automated certification of absence of undisclosed/private training data",
            "Fresh-clone default generation must pass twice and match the exact submitted FASTA bytes",
            "Official submission receipt, wet-lab admission and authorship eligibility require organizer confirmation",
        ]
    )
    result = {
        "kind": "authorship_readiness_inventory_v2",
        "commit": commit,
        "branch": branch,
        "github": visibility,
        "release_files": artifacts,
        "data_inventory": data,
        "download_sources": sources,
        "gaps": gaps,
        "eligibility": "not_certified",
    }
    prepare_run(out, result)
    write_json(out / "audit.json", result)
    lines = [
        "# Authorship release audit",
        "",
        f"Commit: `{commit}`; branch: `{branch}`.",
        "",
        "This is a file/provenance inventory, not an eligibility decision.",
        "",
        "| Release file | Present | Committed | Matches HEAD |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{r['path']}` | {r['present']} | {r['committed']} | {r['matches_commit']} |"
        for r in artifacts
    ]
    lines += ["", "## Remaining work", ""] + [f"- {gap}" for gap in gaps]
    lines += ["", "## Training-source records available on this machine", ""]
    lines += [
        f"- `{r['path']}`: {'present' if r['present'] else 'not found at declared path'}"
        + (f"; SHA-256 `{r['sha256']}`; {r['bytes']} bytes" if r['present'] else "")
        for r in data
    ]
    lines += ["", "## Download manifest verification", "",
              "This registry may be incomplete. Matching hashes establish file integrity, "
              "not training use, license approval, or checkpoint lineage.", "",
              f"Manifest status: {sources['status']}.", ""]
    lines += [
        f"- `{r['path']}` → `{r['local_path']}`: {r['status']}"
        + (f"; actual SHA-256 `{r['actual_sha256']}`" if 'actual_sha256' in r else "")
        + (f"; {r['reason']}" if 'reason' in r else "")
        for r in sources["entries"]
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    mark_files(out, "complete.json", ["run.json", "audit.json", "REPORT.md"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--github-repo", help="Optional read-only visibility check using authenticated gh"
    )
    args = parser.parse_args()
    result = run(args.root, args.out, args.github_repo)
    print(
        f"Authorship inventory: {args.out / 'REPORT.md'}; {len(result['gaps'])} unresolved checks"
    )


if __name__ == "__main__":
    main()
