"""Collect explicitly configured local competition evidence into a new report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amp_challenge_2027.config import PROJECT_ROOT
from amp_challenge_2027.evidence_inventory import collect_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments/competition_evidence_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        code = collect_evidence(args.repo_root, args.config, args.out)
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Evidence report: {args.out.resolve()}")
    if code == 2:
        print("Required artifacts are missing or invalid; see STATUS.md", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
