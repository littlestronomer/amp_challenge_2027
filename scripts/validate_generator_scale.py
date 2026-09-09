"""Frozen-endpoint scale validation: sample, audit, evaluate, then report."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_utils import (
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from scale_validation import checked, frontier, growth, library_ids, sources, validate_pool
from selection_cache import separate_output

from amp_challenge_2027.config import REWARD_DIR, REWARD_HEMO_DIR
from amp_challenge_2027.data import iter_fasta, write_fasta


class Scorers:
    def __init__(self, recipe, device):
        import torch

        from amp_challenge_2027.reward_benchmark import FrozenEncoder, build_head

        self.device = device
        common = recipe["sources"]["common"]
        self.encoder = FrozenEncoder(common["configs"]["activity"]["esm_model"], common["revision"], device=device)
        self.heads, self.temperatures = [], []
        for name, root in (("activity", REWARD_DIR), ("hemolysis", REWARD_HEMO_DIR)):
            path = root / "classifier.pt"
            if common["inputs"].get(str(path.resolve())) != sha256(path):
                raise ValueError("Reward head not pinned by source")
            head = build_head(480, 1, "mlp").to(device).eval().requires_grad_(False)
            head.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            self.heads.append(head)
            self.temperatures.append(common["configs"][name]["temperature"])
        self.evaluators = []
        root = Path(recipe["sources"]["evaluator"]) / "hemolysis/original"
        for seed in (42, 43, 44):
            path = root / f"seed{seed}/head.pt"
            if common["inputs"].get(str(path.resolve())) != sha256(path):
                raise ValueError("Evaluator not pinned by source")
            head = build_head(480, 1, "mlp").to(device).eval().requires_grad_(False)
            head.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            self.evaluators.append(head)
        self.evaluation_temperature = json.loads((root / "calibration.json").read_text())["temperature"]

    def score(self, seqs):
        import torch

        result = []
        for start in range(0, len(seqs), 512):
            features = torch.as_tensor(self.encoder.encode(seqs[start:start+512], 64), device=self.device)
            with torch.no_grad():
                cols = [torch.sigmoid(head(features).squeeze(-1)/t) for head, t in zip(self.heads, self.temperatures, strict=True)]
                logits = torch.stack([h(features) for h in self.evaluators]).mean(0).squeeze(-1)
                cols.append(torch.sigmoid(logits/self.evaluation_temperature))
                result.append(torch.stack(cols, dim=1).cpu().numpy())
        return np.concatenate(result)


def generate_cell(dest, recipe, cell, scorers):
    import torch
    from pilot_grpo import assert_strict_runtime, strict_runtime

    from amp_challenge_2027.grpo import rollout
    from amp_challenge_2027.model import load_model

    if dest.exists():
        if not (dest / "complete.json").exists():
            raise ValueError(f"Incomplete sampling cell {dest}; preserve it and use a new --out. Mid-cell resume is not supported")
        checked(dest, required=("pool.csv", "status.json", "run.json"))
        saved = json.loads((dest / "run.json").read_text())
        if saved["recipe"] != recipe or saved["cell"] != cell:
            raise ValueError("Completed cell recipe differs")
        print(f"[scale] verified completed cell {cell}", flush=True)
        return
    info = recipe["sources"]["cells"][cell]
    model, _ = load_model(Path(info["checkpoint"]), map_location=recipe["device"])
    model = model.to(recipe["device"]).float().eval().requires_grad_(False)
    runtime = strict_runtime(info["sampling_seed"])
    runtime["device"] = recipe["device"]
    if str(recipe["device"]).startswith("cuda"):
        runtime["gpu"] = torch.cuda.get_device_name(torch.device(recipe["device"]))
    random = torch.Generator(device=recipe["device"]).manual_seed(info["sampling_seed"])
    dest.mkdir(parents=True)
    write_json(dest / "run.json", {"recipe": recipe, "cell": cell, "runtime": runtime})
    seqs = []
    while len(seqs) < recipe["draws"]:
        assert_strict_runtime()
        _, batch = rollout(model, min(32, recipe["draws"]-len(seqs)), random, recipe["device"])
        seqs.extend(batch)
        if len(seqs) % 1024 == 0 or len(seqs) == recipe["draws"]:
            print(f"[scale] {cell}: sampled {len(seqs)}/{recipe['draws']}", flush=True)
    del model
    gc.collect()
    if str(recipe["device"]).startswith("cuda"):
        torch.cuda.empty_cache()
    scores = scorers.score(seqs)
    frame = pd.DataFrame({"sequence": seqs, "activity": scores[:, 0], "risk": scores[:, 1], "evaluation_risk": scores[:, 2]})
    validate_pool(frame)
    reference = {s for _, s in iter_fasta(Path(recipe["sources"]["reference"]))}
    ids = library_ids(seqs, reference)
    frame.to_csv(dest / "pool.csv", index=False)
    names = ["pool.csv", "run.json", "status.json"]
    if len(ids) == 50000:
        write_fasta([seqs[i] for i in ids], dest / "library.fasta")
        names.append("library.fasta")
    write_json(dest / "status.json", {"raw_draws": len(seqs), "library_size": len(ids),
               "library_complete": len(ids) == 50000, "library_shortfall": 50000-len(ids),
               "draws_to_50000": ids[-1]+1 if len(ids) == 50000 else None,
               "exact_reference_draws": sum(s in reference for s in seqs),
               "duplicate_draws": len(seqs)-len(set(seqs)), "deployment_approved": False})
    # Refuse completed output if any externally pinned input changed mid-run.
    for path, digest in recipe["sources"]["common"]["inputs"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Pinned input changed during sampling: {path}")
    endpoint = info.get("root")
    if endpoint:
        checked(Path(endpoint), required=("policy/model.pt", "policy/config.json"))
        if sha256(Path(endpoint) / "complete.json") != info["marker_sha256"]:
            raise ValueError("Endpoint changed during sampling")
    mark_files(dest, "complete.json", names)


def audit_cell(dest, recipe):
    checked(dest, required=("pool.csv", "status.json", "run.json"))
    saved = json.loads((dest / "run.json").read_text())
    if saved["recipe"] != recipe or saved["cell"] != f"{dest.parent.name}/{dest.name}":
        raise ValueError("Sample cell identity differs")
    audit = dest / "audit"
    audit_recipe = {"sample_marker": sha256(dest / "complete.json"), "code": code_identity(),
                    "scales": recipe["scales"], "targets": [.5, .55, .6, .65], "shortlist": 2000}
    prepare_run(audit, audit_recipe)
    if (audit / "complete.json").exists():
        checked(audit)
        return
    frame = pd.read_csv(dest / "pool.csv")
    reference = {s for _, s in iter_fasta(Path(recipe["sources"]["reference"]))}
    rows, lengths = growth(frame, reference, recipe["scales"])
    write_summary(audit / "growth.csv", rows)
    write_summary(audit / "lengths.csv", lengths)
    names = ["run.json", "growth.csv", "lengths.csv", "status.json"]
    ids = library_ids(frame.sequence.tolist(), reference)
    complete = len(ids) == 50000
    if complete:
        actual = [s for _, s in iter_fasta(dest / "library.fasta")]
        if actual != frame.iloc[ids].sequence.tolist():
            raise ValueError("Library differs from first unique non-reference sequences")
        summaries, selections = frontier(frame.iloc[ids], reference)
        write_summary(audit / "frontier.csv", summaries)
        names.append("frontier.csv")
        if selections:
            write_summary(audit / "selected.csv", selections)
            names.append("selected.csv")
    write_json(audit / "status.json", {"library_complete": complete, "frontier_skipped": not complete,
                                       "library_exact_reference_overlap": 0,
                                       "top_novelty_rule": "all references: normalized-indel similarity <=.8",
                                       "full_submission_validation": False})
    mark_files(audit, "complete.json", names)


def report(root, recipe):
    tables = {"growth": [], "lengths": [], "frontier": [], "official": []}
    statuses = []
    runtimes = []
    audit_recipes = []
    proofs = {}
    for cell in recipe["sources"]["cells"]:
        dest = root / cell
        checked(dest, required=("pool.csv", "status.json"))
        checked(dest / "audit")
        audit_recipe = json.loads((dest / "audit/run.json").read_text())
        if audit_recipe["sample_marker"] != sha256(dest / "complete.json"):
            raise ValueError("Audit is not linked to this sampled pool")
        audit_recipes.append({k: v for k, v in audit_recipe.items() if k != "sample_marker"})
        saved = json.loads((dest / "run.json").read_text())
        if saved["recipe"] != recipe or saved["cell"] != cell:
            raise ValueError("Sample cell identity differs")
        runtimes.append(saved["runtime"])
        proofs[cell] = {"sample": sha256(dest / "complete.json"), "audit": sha256(dest / "audit/complete.json")}
        method, seed = cell.split("/seed")
        labels = {"method": method, "seed": int(seed)}
        statuses.append({**labels, **json.loads((dest / "status.json").read_text())})
        for name in ("growth", "lengths", "frontier"):
            path = dest / "audit" / f"{name}.csv"
            if path.exists():
                tables[name].append(pd.read_csv(path).assign(**labels))
        if (dest / "evaluation.json").exists():
            checked(dest, "evaluation.json", required=("metrics.json", "library.fasta"))
            checked(dest / "official", required=("run.json",))
            official_recipe = json.loads((dest / "official/run.json").read_text())
            if (official_recipe["sample_marker"] != sha256(dest / "complete.json") or
                    official_recipe["esm_model"] != "facebook/esm2_t33_650M_UR50D" or
                    official_recipe.get("seed") != 2027 or
                    official_recipe["reference_sha256"] != sha256(Path(recipe["sources"]["reference"]))):
                raise ValueError("Official metrics are not linked to this library/reference/protocol")
            tables["official"].append({**labels, **json.loads((dest / "metrics.json").read_text())})
            proofs[cell]["evaluation"] = sha256(dest / "evaluation.json")
    if any(r != runtimes[0] for r in runtimes[1:]):
        raise ValueError("Sampling runtimes differ across cells")
    if any(r != audit_recipes[0] for r in audit_recipes[1:]):
        raise ValueError("Audit recipes/code differ across cells")
    output = root / "report"
    output.mkdir(exist_ok=True)
    write_summary(output / "status.csv", statuses)
    for name, parts in tables.items():
        if parts:
            table = pd.DataFrame(parts) if name == "official" else pd.concat(parts, ignore_index=True)
            table.to_csv(output / f"{name}.csv", index=False)
    matched = []
    if tables["frontier"]:
        frame = pd.concat(tables["frontier"], ignore_index=True)
        for (seed, target), group in frame.groupby(["seed", "target_distance"]):
            full = set(group.method) == {"baseline", "raft", "grpo"} and group.complete.all()
            spread = group.achieved_distance.max()-group.achieved_distance.min() if full else None
            matched.append({"seed": int(seed), "target_distance": float(target),
                            "all_top100_complete": bool(full), "achieved_distance_spread": float(spread) if full else None,
                            "matched_within_001": bool(full and spread <= .01)})
    write_json(output / "report.json", {"matched_diversity": matched, "inputs": proofs, "deployment_approved": False,
        "complete_libraries": sum(s["library_complete"] for s in statuses), "expected_libraries": len(statuses),
        "official_evaluations": len(tables["official"]),
        "limitations": ["Proxy evaluation shares data/backbone lineage; no experimental safety claim",
                        "One fresh sampling seed per training endpoint; baseline reused across methods within each seed",
                        "Greedy separation is not a family graph or maximum independent set",
                        "Matched-diversity comparisons require complete sets and achieved spread <=.01",
                        "Fixed top2000 training-score shortlist; novelty/selection shortfalls never backfilled",
                        "Official component metrics do not reproduce the hidden competition ranking"]})
    print(pd.DataFrame(statuses).to_string(index=False))
    print(f"[scale] reports: {output}; no automatic promotion")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "audit", "evaluate", "report"))
    parser.add_argument("--pilot-root", type=Path, default=Path("sweep_results"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=("baseline", "raft", "grpo"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if not 2048 <= args.draws <= 200000 or args.draws % 32:
        raise ValueError("Draws must be a multiple of32 in [2048,200000]")
    if args.command == "sample":
        pinned = sources(args.pilot_root)
        separate_output(args.out, [Path(c["checkpoint"]) for c in pinned["cells"].values()] +
                        [Path(c["root"]) for c in pinned["cells"].values() if "root" in c] +
                        [Path(p) for p in pinned["common"]["inputs"]])
        recipe = {"kind": "frozen_generator_scale_v1", "sources": pinned, "draws": args.draws,
                  "scales": sorted({n for n in (2048, 8192, 32768, args.draws) if n <= args.draws}),
                  "device": args.device, "code": code_identity(),
                  "sampling": "pilot full categorical mask 8..50; fixed batches32; sampling seed40000+training seed",
                  "library": "first50000 unique non-reference valid sequences; no score filtering or automatic refill"}
    else:
        recipe = json.loads((args.out / "run.json").read_text())
        if recipe["kind"] != "frozen_generator_scale_v1":
            raise ValueError("Wrong scale run kind")
        if sources(args.pilot_root) != recipe["sources"]:
            raise ValueError("Source artifacts changed")
    cells = [c for c in sorted(recipe["sources"]["cells"]) if
             (args.method is None or c.startswith(args.method+"/")) and
             (args.seed is None or c.endswith(f"seed{args.seed}"))]
    if args.list:
        print(json.dumps({"recipe": recipe, "cells": cells}, indent=2))
        return
    if args.command == "sample":
        prepare_run(args.out, recipe)
        from pilot_grpo import strict_runtime

        strict_runtime(42)
        scorers = Scorers(recipe, recipe["device"])
        for cell in cells:
            generate_cell(args.out / cell, recipe, cell, scorers)
    elif args.command == "audit":
        for cell in cells:
            print(f"[scale-audit] {cell}", flush=True)
            audit_cell(args.out / cell, recipe)
    elif args.command == "evaluate":
        for cell in cells:
            dest = args.out / cell
            checked(dest, required=("pool.csv", "status.json"))
            if not json.loads((dest / "status.json").read_text())["library_complete"]:
                print(f"[scale-eval] SKIP {cell}: incomplete 50k library", flush=True)
                continue
            if (dest / "evaluation.json").exists() and not (dest / "official/run.json").exists():
                raise ValueError("Existing evaluation has no pinned official recipe")
            prepare_run(dest / "official", {"sample_marker": sha256(dest / "complete.json"),
                        "reference_sha256": sha256(Path(recipe["sources"]["reference"])),
                        "esm_model": "facebook/esm2_t33_650M_UR50D", "device": args.device, "seed": 2027, "code": code_identity()})
            evaluate_library(dest, reference=Path(recipe["sources"]["reference"]),
                             esm_model="facebook/esm2_t33_650M_UR50D", device=args.device, seed=2027)
            mark_files(dest / "official", "complete.json", ["run.json"])
    else:
        report(args.out, recipe)


if __name__ == "__main__":
    main()
