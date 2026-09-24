"""Inventory release gaps without claiming eligibility or publishing anything.

Run on the training machine to expose files present only in its working tree.
No model loading, inference, submission, or Git mutation is performed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiment_utils import REPO_ROOT, mark_files, prepare_run, sha256, write_json

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
        "kind": "authorship_readiness_inventory_v1",
        "commit": commit,
        "branch": branch,
        "github": visibility,
        "release_files": artifacts,
        "data_inventory": data,
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
        for r in data
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
