"""Research-only measured-condition generator experiments (never promotes a release)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Works both from an installed checkout and with `python scripts/...` on SSH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Snapshot MarLys, full DBAASP JSON and DRAMP XLSX")
    fetch.add_argument("--out", required=True, type=Path)
    fetch.add_argument("--marlys", required=True, type=Path)
    fetch.add_argument("--dbaasp-dir", type=Path, help="Existing full JSON cards; legacy CSVs are insufficient")
    fetch.add_argument("--dramp", type=Path, help="Reuse an existing General Data XLSX")
    fetch.add_argument("--workers", type=int, default=4)
    fetch.add_argument("--max-records", type=int, help="Incomplete snapshot for smoke tests only")
    fetch.add_argument("--include-hemolytik", action="store_true", help="Optional research import; unresolved source terms recorded")
    fetch.add_argument("--hemolytik", type=Path, help="Existing Hemolytik complete CSV (requires explicit opt-in)")
    prep = commands.add_parser("prepare", help="Preserve observations, establish chemistry eligibility and family splits")
    prep.add_argument("--snapshot", required=True, type=Path)
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--seed", type=int, default=2027)
    prep.add_argument("--threshold", type=float, default=.8)
    prep.add_argument("--allow-incomplete", action="store_true", help="Smoke-test preparation only; tracked in provenance")
    train = commands.add_parser("train", help="Fine-tune an incumbent or train a matched architecture preset")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--arm", choices=["frozen", "selective", "unconditional", "conditional"], default="conditional")
    train.add_argument("--preset", choices=list("ABCD"), help="Fresh training: A dense, B AttnRes, C MoE, D both")
    train.add_argument("--checkpoint", type=Path, help="Warm-start generator directory; omit with fresh --preset")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--warmup-epochs", type=int, default=1)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--examples-per-epoch", type=int, help="Override matched training exposure; recorded")
    train.add_argument("--min-selective", type=int, default=32)
    train.add_argument("--model-config", type=Path, help="Explicit architecture overrides for smoke tests; recorded")
    sample = commands.add_parser("sample", help="Generate and retain all raw draws with a frozen request")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--label", required=True, help="Same label for an arm's three seeds; distinct labels for requests")
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--draws", type=int, default=10000)
    sample.add_argument("--batch-size", type=int, default=64)
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--conditions", type=Path)
    sample.add_argument("--temperature", type=float, default=1.)
    sample.add_argument("--reference", type=Path, default=Path("data/antibacterial.fasta"))
    compare = commands.add_parser("compare", help="Score fresh outputs with frozen legacy heads and compare paired seeds")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--baseline", required=True, help="A label supplied to sample")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--reference", type=Path, default=Path("data/antibacterial.fasta"))
    compare.add_argument("--device", default="cuda")
    compare.add_argument("--activity-floor", type=float, default=.8)
    compare.add_argument("--risk-ceiling", type=float, default=.5)
    compare.add_argument("--revision", help="Pin the 35M ESM backbone revision")
    compare.add_argument("--select-top", action="store_true", help="Run the unchanged production selector (adds conformity/8M ESM scoring)")
    return cli


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    if command == "fetch":
        from amp_challenge_2027.conditional_data import fetch_sources
        result = fetch_sources(**args)
    elif command == "prepare":
        from amp_challenge_2027.conditional_data import prepare_dataset
        result = prepare_dataset(**args)
    elif command == "train":
        from amp_challenge_2027.conditional_training import train_experiment
        model_config = args.pop("model_config")
        args["model_overrides"] = json.loads(model_config.read_text()) if model_config else None
        result = train_experiment(**args)
    elif command == "sample":
        from amp_challenge_2027.conditional_research import sample_experiment
        result = sample_experiment(**args)
    else:
        from amp_challenge_2027.conditional_research import compare_experiments
        result = compare_experiments(**args)
    status = result.get("status", "complete")
    print(f"[conditional-{command}] status={status}; output={args['out']}", flush=True)
    # Unsupported selective evidence is a reported result, not a trained checkpoint.
    return 2 if status in {"insufficient_joint_labels", "insufficient_condition_labels", "insufficient_supervision"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
