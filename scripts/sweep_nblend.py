"""N-way fixed-weights library blend probe over frozen component pools.

Reads N pre-generated, verified component pools (FASTA), builds exact-quota
blends at fixed integer weight vectors, and evaluates them under the official
metric suite. An optional incumbent ``--control`` library is evaluated in the
same run for paired deltas. Triage runs on the fast 8M embedder; the ``--gate``
stage re-evaluates the strongest triage cells with the official-fidelity
embedder (the only decision-grade instrument — PHASE1_RESULTS.md lesson:
8M FBD deltas below ~0.05 are not decision-grade).

No models are trained, no checkpoints change, and no promotion happens here.
Every cell is hash-pinned; interrupted runs resume verified cells only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import (  # noqa: E402
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
    write_summary,
)

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA  # noqa: E402
from amp_challenge_2027.data import iter_fasta, read_reference_set, write_fasta  # noqa: E402
from amp_challenge_2027.library_blending import (  # noqa: E402
    apportioned_quotas,
    fixed_weights_blend,
    parse_weights,
)
from amp_challenge_2027.pipeline import clean_candidates  # noqa: E402

# Metrics with a decision floor relative to the control at the gate; conformity's
# key has drifted across seqme versions, so lookups accept either spelling.
TOLERANCE_FLOOR = {"ConformityScore": 0.01, "Precision": 0.01, "Recall": 0.01, "Diversity": 0.01}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def metric_value(metrics: dict, *candidates: str) -> float:
    for key in candidates:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"metrics missing all of {candidates}; available: {sorted(metrics)}")


def conformity(metrics: dict) -> float:
    return metric_value(metrics, "ConformityScore", "Conformity score", "Conformity score (mean)")


def parse_named_spec(spec: str, flag: str, parser: argparse.ArgumentParser) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not NAME_RE.match(name):
        parser.error(f"{flag} must be NAME=PATH with a lowercase name (got {spec!r})")
    return name, Path(path)


def load_clean_pool(path: Path, reference: set[str], label: str) -> list[str]:
    sequences = [seq for _, seq in iter_fasta(path)]
    if not sequences or clean_candidates(sequences, reference) != sequences:
        raise ValueError(f"{label} pool must already be valid, unique and reference-free: {path}")
    return sequences


def build_cell(
    directory: Path,
    pools: list[list[str]],
    weights: tuple[int, ...],
    names: list[str],
    size: int,
    pool_hashes: dict[str, str],
) -> dict:
    if not verify_files(directory, "generation.json"):
        blend = fixed_weights_blend(pools, weights, size=size, labels=names)
        directory.mkdir(parents=True, exist_ok=True)
        write_fasta(blend.sequences, directory / "library.fasta")
        with (directory / "membership.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sequence", "source"])
            writer.writerows(zip(blend.sequences, blend.sources))
        stats = {
            "weights": list(weights),
            "labels": list(names),
            "quotas": list(blend.quotas),
            "pool_sizes": list(blend.pool_sizes),
            "pool_shared": list(blend.pool_shared),
            "selected_shared": list(blend.selected_shared),
            "library_size": size,
            **pool_hashes,
        }
        write_json(directory / "blend_stats.json", stats)
        mark_files(directory, "generation.json", ["library.fasta", "membership.csv", "blend_stats.json"])
    stats = json.loads((directory / "blend_stats.json").read_text())
    if any(stats.get(key) != value for key, value in pool_hashes.items()):
        raise ValueError(f"Component pool changed since this blend was built: {directory}")
    return stats


def evaluate_into(directory: Path, library: Path, *, reference: Path, esm_model: str, device: str) -> dict:
    """Evaluate ``library`` inside a fresh per-stage directory (marker isolation)."""
    if not verify_files(directory, "evaluation.json"):
        directory.mkdir(parents=True, exist_ok=True)
        write_fasta([seq for _, seq in iter_fasta(library)], directory / "library.fasta")
    return evaluate_library(directory, reference=reference, esm_model=esm_model, device=device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component", action="append", required=True, metavar="NAME=POOL.fasta",
        help="Verified clean component pool (repeatable, >= 2); weight order = flag order",
    )
    parser.add_argument(
        "--control", metavar="NAME=LIBRARY.fasta", default=None,
        help="Optional incumbent library evaluated alongside cells for paired deltas",
    )
    parser.add_argument("--ratios", nargs="+", required=True, help="N-way integer weights, e.g. 9:3:1 5:2:1")
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=50_000)
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t6_8M_UR50D", help="Triage embedder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gate", action="store_true", help="Re-evaluate the strongest cells with --gate-esm-model")
    parser.add_argument("--gate-esm-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--gate-top", type=int, default=4)
    parser.add_argument(
        "--triage-margin", type=float, default=0.05,
        help="Cells with triage FBD above control FBD + margin are not gated (8M non-decision band)",
    )
    parser.add_argument("--list", action="store_true", help="Validate inputs and show the plan; no writes")
    parser.add_argument("--generate-only", action="store_true", help="Build blends without evaluation")
    args = parser.parse_args(argv)

    if args.out.resolve() == args.reference.resolve():
        parser.error("--out must not be the reference file")
    try:
        ratios = list(dict.fromkeys(parse_weights(value) for value in args.ratios))
    except ValueError as error:
        parser.error(str(error))
    names: list[str] = []
    paths: list[Path] = []
    for spec in args.component:
        name, path = parse_named_spec(spec, "--component", parser)
        if name in names:
            parser.error(f"duplicate component name {name!r}")
        if not path.exists():
            parser.error(f"component pool not found: {path}")
        names.append(name)
        paths.append(path)
    control = None
    if args.control is not None:
        control_name, control_path = parse_named_spec(args.control, "--control", parser)
        if control_name in names:
            parser.error("--control name must differ from component names")
        if not control_path.exists():
            parser.error(f"control library not found: {control_path}")
        control = (control_name, control_path)
    for ratio in ratios:
        if len(ratio) != len(names):
            parser.error(f"ratio {':'.join(map(str, ratio))} does not match {len(names)} components")

    reference = read_reference_set(args.reference)
    if not reference:
        raise ValueError("Reference must not be empty")
    pools = [load_clean_pool(path, reference, name) for path, name in zip(paths, names)]
    pool_hashes = {f"{name}_sha256": sha256(path) for name, path in zip(names, paths)}
    control_pool = load_clean_pool(control[1], reference, control[0]) if control else None

    for ratio in ratios:
        quotas = apportioned_quotas(args.size, ratio)
        for name, pool, quota in zip(names, pools, quotas):
            if len(pool) < quota:
                parser.error(
                    f"ratio {':'.join(map(str, ratio))}: component {name} has {len(pool)} "
                    f"candidates < quota {quota}"
                )
        print(
            f"[nblend] w{':'.join(map(str, ratio))}: "
            + " + ".join(f"{name}={quota}" for name, quota in zip(names, quotas)),
            flush=True,
        )
    if control:
        print(f"[nblend] control {control[0]}: {len(control_pool)} sequences ({sha256(control[1])[:12]})", flush=True)
    print(f"[nblend] triage embedder {args.esm_model}; gate " + (f"on: {args.gate_esm_model}" if args.gate else "off"), flush=True)
    if args.list:
        return

    code = code_identity()
    recipe = {
        "kind": "nway_blend_probe_v1",
        "code": code,
        "components": [
            {"name": name, "path": str(path.resolve()), "sha256": sha256(path), "size": len(pool)}
            for name, path, pool in zip(names, paths, pools)
        ],
        "control": (
            {"name": control[0], "path": str(control[1].resolve()), "sha256": sha256(control[1]), "size": len(control_pool)}
            if control
            else None
        ),
        "ratios": [list(ratio) for ratio in ratios],
        "library_size": args.size,
        "reference_sha256": sha256(args.reference),
        "esm_model": args.esm_model,
        "device": args.device,
        "gate": {
            "enabled": args.gate,
            "esm_model": args.gate_esm_model if args.gate else None,
            "top": args.gate_top,
            "triage_margin": args.triage_margin,
        },
        "tolerance_floor": TOLERANCE_FLOOR,
    }
    prepare_run(args.out, recipe)

    rows: list[dict] = []
    for ratio in ratios:
        case = "w" + "_".join(map(str, ratio))
        directory = args.out / case
        stats = build_cell(directory, pools, ratio, names, args.size, pool_hashes)
        row = {
            "case": case,
            "ratio": ":".join(map(str, ratio)),
            "library_sha256": sha256(directory / "library.fasta"),
            **stats,
        }
        if not args.generate_only:
            row.update(evaluate_library(directory, reference=args.reference, esm_model=args.esm_model, device=args.device))
        rows.append(row)
        write_summary(args.out / "results.csv", rows)
        print(f"[nblend] complete {case}", flush=True)

    control_metrics: dict | None = None
    if control is not None and not args.generate_only:
        control_dir = args.out / f"control_{control[0]}"
        control_metrics = evaluate_into(
            control_dir, control[1], reference=args.reference, esm_model=args.esm_model, device=args.device
        )
        rows.append({"case": f"control_{control[0]}", "ratio": None, **control_metrics})
        write_summary(args.out / "results.csv", rows)
        print(f"[nblend] control triage complete", flush=True)

    if args.gate:
        if args.generate_only:
            print("[nblend] --gate ignored with --generate-only", flush=True)
            return
        if control_metrics is None:
            parser.error("--gate requires --control (paired gate deltas need an incumbent)")
        control_fbd = metric_value(control_metrics, "FBD")
        eligible = sorted(
            (row for row in rows if row.get("FBD") is not None and row["FBD"] <= control_fbd + args.triage_margin),
            key=lambda row: row["FBD"],
        )
        selected = eligible[: args.gate_top]
        print(
            f"[nblend] gate: {len(eligible)} eligible cells within +{args.triage_margin} triage FBD; "
            f"gating {len(selected)}: {[row['case'] for row in selected]}",
            flush=True,
        )
        gate_rows = []
        for row in selected:
            gate_dir = args.out / "gate" / row["case"]
            metrics = evaluate_into(
                gate_dir, args.out / row["case"] / "library.fasta",
                reference=args.reference, esm_model=args.gate_esm_model, device=args.device,
            )
            if sha256(gate_dir / "library.fasta") != row["library_sha256"]:
                raise ValueError(f"Gate library bytes differ from the triage cell: {row['case']}")
            gate_rows.append({"case": row["case"], "ratio": row["ratio"], **metrics})
        gate_control_dir = args.out / "gate" / f"control_{control[0]}"
        gate_control = evaluate_into(
            gate_control_dir, control[1], reference=args.reference,
            esm_model=args.gate_esm_model, device=args.device,
        )
        gate_rows.append({"case": f"control_{control[0]}", "ratio": None, **gate_control})
        write_summary(args.out / "gate_results.csv", gate_rows)

        verdicts = []
        for row in gate_rows:
            if row["case"].startswith("control_"):
                continue
            verdict = {"case": row["case"], "ratio": row["ratio"]}
            for metric in ("FBD", "MMD"):
                value = metric_value(row, metric)
                reference_value = metric_value(gate_control, metric)
                verdict[f"{metric}_delta"] = value - reference_value
                verdict[f"{metric}_improved"] = value < reference_value
            floors_ok = True
            for metric, floor in TOLERANCE_FLOOR.items():
                value = metric_value(row, metric) if metric != "ConformityScore" else conformity(row)
                reference_value = conformity(gate_control) if metric == "ConformityScore" else metric_value(gate_control, metric)
                verdict[f"{metric}_delta"] = value - reference_value
                verdict[f"{metric}_within_floor"] = value >= reference_value - floor
                floors_ok = floors_ok and verdict[f"{metric}_within_floor"]
            verdict["within_tolerance"] = (
                verdict["FBD_improved"] and verdict["MMD_improved"] and floors_ok
            )
            verdicts.append(verdict)
        write_summary(args.out / "gate_verdicts.csv", verdicts)
        write_json(
            args.out / "report.json",
            {
                "selected_cases": [row["case"] for row in selected],
                "gate_esm_model": args.gate_esm_model,
                "control": f"control_{control[0]}",
                "tolerance_floor": TOLERANCE_FLOOR,
                "verdicts": verdicts,
                "note": "Paired gate deltas vs the in-run control. Not official rankings; "
                "no automatic promotion — a winning cell still requires entry-point wiring, "
                "byte-determinism verification and verify_submission before adoption.",
            },
        )
        print("[nblend] gate complete; verdicts in gate_verdicts.csv / report.json", flush=True)


if __name__ == "__main__":
    main()
