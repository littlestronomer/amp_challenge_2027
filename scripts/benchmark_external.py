"""Pinned raw-generator comparison; not native starter-kit submissions or Kaggle scores."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from experiment_utils import (
    REPO_ROOT,
    code_identity,
    evaluate_library,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from scale_validation import checked, frontier, growth, library_ids
from selection_cache import separate_output

KITS = {
    "ampdiffusion": {"repository": "https://github.com/szczurek-lab/ampdiffusion-starter-kit.git",
                     "commit": "1a862af9078e6b55c87d1fa576f3da81851ba94b"},
    "hydramp_raw": {"repository": "https://github.com/szczurek-lab/hydramp-starter-kit.git",
                    "commit": "7804df862872ccc6d09fe01c41bafbca194cfa31"},
}
METHODS = ("baseline", "teacher", "specialist", "anchored", "coverage")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def inventory(root, method):
    """Reject dirty source and unresolved/altered LFS inputs without loading pickle weights."""
    root = root.resolve()
    if git(root, "rev-parse", "HEAD").decode().strip() != KITS[method]["commit"]:
        raise ValueError(f"Wrong pinned commit for {method}")
    if git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError(f"Modified tracked files in {root}")
    extra = git(root, "ls-files", "--others", "--exclude-standard", "src", "checkpoint").strip()
    if extra:
        raise ValueError(f"Untracked source/checkpoint files in {root}")
    files = {}
    paths = git(root, "ls-files", "-z", "src", "checkpoint", "pyproject.toml", "uv.lock", ".python-version").decode().split("\0")
    for name in filter(None, paths):
        p = root / name
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"Missing or symlinked source: {p}")
        digest = sha256(p)
        if name.startswith("checkpoint/"):
            original = git(root, "show", f"HEAD:{name}")
            if original.startswith(b"version https://git-lfs.github.com/spec/v1"):
                expected = original.decode().split("oid sha256:")[1].splitlines()[0]
                if digest != expected:
                    raise ValueError(f"Missing/changed LFS weights: {p}; run git lfs pull in the starter kit")
        files[name] = digest
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise ValueError(f"Missing isolated environment: run uv sync --frozen in {root}")
    return {**KITS[method], "root": str(root), "files": files, "python": str(python)}


def verify_pins(recipe):
    for method, pinned in recipe["kits"].items():
        if inventory(Path(pinned["root"]), method) != pinned:
            raise ValueError("Starter kit changed after preflight")
    for path, expected in recipe["sources"]["common"]["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Pinned scorer/reference changed: {path}")


def controls(root, draws):
    run = json.loads((root / "run.json").read_text())
    if run["kind"] != "opd_evaluation_v1" or run["draws"] < draws:
        raise ValueError("Need a complete OPD sample source with at least the requested draws")
    # Cached probabilities must have been computed by the identical scorer implementation.
    for name in ("scripts/validate_generator_scale.py", "src/amp_challenge_2027/reward_benchmark.py"):
        if subprocess.check_output(["git", "show", f"{run['code']['commit']}:{name}"], cwd=REPO_ROOT) != (REPO_ROOT / name).read_bytes():
            raise ValueError("Cached-control scorer implementation changed")
    cells = {}
    for method in METHODS:
        for seed in (42, 43, 44):
            cell = f"{method}/seed{seed}"
            dest = root / cell
            checked(dest, required=("pool.csv", "run.json", "status.json"))
            saved = json.loads((dest / "run.json").read_text())
            if saved["recipe"] != run or saved["cell"] != cell:
                raise ValueError("Control recipe mismatch")
            cells[cell] = {"root": str(dest.resolve()), "marker": sha256(dest / "complete.json")}
    return run["sources"], cells


def valid(s):
    return isinstance(s, str) and 8 <= len(s) <= 50 and not set(s)-set("ACDEFGHIKLMNPQRSTVWY")


def raw_sample(dest, recipe, method, seed):
    pinned = recipe["kits"][method]
    cell_recipe = {"recipe": recipe, "method": method, "seed": seed}
    if (dest / "complete.json").exists():
        checked(dest, required=("run.json", "raw.json"))
        if json.loads((dest / "run.json").read_text()) != cell_recipe:
            raise ValueError("Raw sample recipe changed")
        return
    if dest.exists():
        raise ValueError(f"Partial raw cell {dest}: preserve it and use a new output root")
    prepare_run(dest, cell_recipe)
    command = [pinned["python"], str(REPO_ROOT / "scripts/external_raw_worker.py"),
               "--method", method, "--root", pinned["root"], "--out", str((dest / "raw.json").resolve()),
               "--draws", str(recipe["draws"]), "--seed", str(60000+seed),
               "--batch-size", "32", "--device", recipe["device"]]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONHASHSEED"] = str(60000+seed)
    with (dest / "generation.log").open("w") as log:
        result = subprocess.run(command, cwd=pinned["root"], env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join((dest / "generation.log").read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"External generation failed; see {dest / 'generation.log'}\n{tail}")
    raw = json.loads((dest / "raw.json").read_text())
    if len(raw["sequences"]) != recipe["draws"] or not all(isinstance(s, str) for s in raw["sequences"]):
        raise ValueError("Invalid raw output contract")
    verify_pins(recipe)
    mark_files(dest, "complete.json", ["run.json", "raw.json", "generation.log"])


def score(dest, recipe, scorers):
    import numpy as np
    import pandas as pd

    checked(dest, required=("run.json", "raw.json"))
    if json.loads((dest / "run.json").read_text())["recipe"] != recipe:
        raise ValueError("Raw source recipe mismatch")
    directory = dest / "scored"
    prepare_run(directory, {"raw_marker": sha256(dest / "complete.json"), "code": recipe["code"],
                            "inputs": recipe["sources"]["common"], "device": recipe["device"]})
    if (directory / "complete.json").exists():
        checked(directory, required=("run.json", "pool.csv"))
        return
    sequences = json.loads((dest / "raw.json").read_text())["sequences"]
    keep = [i for i, s in enumerate(sequences) if valid(s)]
    values = np.full((len(sequences), 3), np.nan)
    if keep:
        values[keep] = scorers.score([sequences[i] for i in keep])
    frame = pd.DataFrame({"sequence": sequences, "valid": [valid(s) for s in sequences],
                          "activity": values[:, 0], "risk": values[:, 1], "evaluation_risk": values[:, 2]})
    frame.to_csv(directory / "pool.csv", index=False)
    verify_pins(recipe)
    mark_files(directory, "complete.json", ["run.json", "pool.csv"])


def diagnose(frame, reference, scales):
    """Invalid draws count in yield denominators, but never receive invented scores."""
    rows, lengths = [], []
    for n in scales:
        sub = frame.iloc[:n]
        accepted = sub[sub.sequence.map(valid)]
        if len(accepted):
            values, bins = growth(accepted, reference, [len(accepted)])
            row = values[0]
            for k in list(row):
                if k.endswith("yield_per_1000") or k == "raw_unique_novel_fraction":
                    row[k] *= len(accepted)/n
            # Means apply only to valid peptides. Expose that conditioning explicitly.
            for k in ("raw_evaluation_risk_mean", "raw_evaluation_risk_p75", "raw_prefix256_pairwise_distance", "raw_mean_length"):
                row[k.replace("raw_", "valid_")] = row.pop(k)
            for b in bins:
                b.update(draws=n, raw_share=b["bin_draws"]/n)
            lengths.extend(bins)
        else:
            row = {"raw_joint_yield_per_1000": 0., "raw_evaluation_yield_per_1000": 0.,
                   "raw_reward_yield_per_1000": 0., "separated_yield_per_1000": 0.,
                   "raw_unique_novel_fraction": 0., "joint_pass_unique": 0, "joint_pass_separated_08": 0}
        rows.append({**row, "draws": n, "valid_draws": len(accepted), "invalid_draws": n-len(accepted)})
    return rows, lengths


def audit_cell(dest, frame, recipe, origin):
    import pandas as pd

    from amp_challenge_2027.data import iter_fasta, write_fasta

    prepare_run(dest, {"recipe": recipe, "origin": origin})
    if (dest / "complete.json").exists():
        checked(dest, required=("run.json", "growth.csv", "frontier.csv", "status.json"))
        return
    if len(frame) != recipe["draws"]:
        raise ValueError("Wrong scored draw count")
    frame = frame.copy()
    for column in ("activity", "risk", "evaluation_risk"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    reference = {s for _, s in iter_fasta(Path(recipe["sources"]["reference"]))}
    rows, lengths = diagnose(frame, reference, recipe["scales"])
    good = frame[frame.sequence.map(valid)]
    ids = library_ids(good.sequence.tolist(), reference)
    library = good.iloc[ids]
    summary, selected = frontier(library, reference)
    names = ["run.json", "growth.csv", "frontier.csv", "status.json"]
    write_summary(dest / "growth.csv", rows)
    write_summary(dest / "frontier.csv", summary)
    write_json(dest / "status.json", {"raw_draws": len(frame), "valid_draws": len(good),
               "invalid_draws": len(frame)-len(good), "library_size": len(ids),
               "library_complete": len(ids) == 50000, "library_shortfall": 50000-len(ids),
               "deployment_approved": False})
    if lengths:
        write_summary(dest / "lengths.csv", lengths)
        names.append("lengths.csv")
    if selected:
        write_summary(dest / "selected.csv", selected)
        names.append("selected.csv")
    if len(ids) == 50000:
        write_fasta(library.sequence.tolist(), dest / "library.fasta")
        names.append("library.fasta")
    mark_files(dest, "complete.json", names)


def checked_origin(recipe, method, seed, root):
    cell = f"{method}/seed{seed}"
    if method in METHODS:
        info = recipe["controls"][cell]
        source = Path(info["root"])
        checked(source, required=("pool.csv", "run.json"))
        if sha256(source / "complete.json") != info["marker"]:
            raise ValueError("Control changed")
        return info
    source = root / "raw" / cell
    checked(source, required=("raw.json", "run.json"))
    if json.loads((source / "run.json").read_text()) != {"recipe": recipe, "method": method, "seed": seed}:
        raise ValueError("Raw identity changed")
    checked(source / "scored", required=("pool.csv", "run.json"))
    expected = {"raw_marker": sha256(source / "complete.json"), "code": recipe["code"],
                "inputs": recipe["sources"]["common"], "device": recipe["device"]}
    if json.loads((source / "scored/run.json").read_text()) != expected:
        raise ValueError("Scoring provenance changed")
    return {"scored_marker": sha256(source / "scored/complete.json")}


def cross_seed_overlap(samples):
    """Exact sequence overlap is diagnostic, not proof of RNG dependence."""
    from itertools import combinations

    rows = []
    for a, b in combinations(sorted(samples), 2):
        x, y = samples[a], samples[b]
        sx, sy = set(x), set(y)
        shared = len(sx & sy)
        rows.append({"seed_a": a, "seed_b": b, "shared_unique": shared,
                     "unique_a": len(sx), "unique_b": len(sy),
                     "jaccard": shared / len(sx | sy) if sx | sy else 0.,
                     "same_position_count": sum(u == v for u, v in zip(x, y)),
                     "shift_one_batch_equal": x[32:] == y[:-32] if len(x) > 32 and len(x) == len(y) else False})
    return rows


def report(root, recipe):
    import pandas as pd

    tables = {k: [] for k in ("growth", "frontier", "status", "official")}
    runtimes, proofs, costs, samples = {}, {}, [], {}
    for method in (*METHODS, *KITS):
        for seed in (42, 43, 44):
            dest = root / "audit" / method / f"seed{seed}"
            checked(dest, required=("run.json", "growth.csv", "frontier.csv", "status.json"))
            origin = checked_origin(recipe, method, seed, root)
            if json.loads((dest / "run.json").read_text()) != {"recipe": recipe, "origin": origin}:
                raise ValueError("Audit recipe changed")
            proofs[f"{method}/seed{seed}"] = sha256(dest / "complete.json")
            if method in KITS:
                raw = json.loads((root / "raw" / method / f"seed{seed}/raw.json").read_text())
                samples.setdefault(method, {})[seed] = raw["sequences"]
                runtime = {k: raw[k] for k in ("runtime", "packages", "python")}
                if runtime != runtimes.setdefault(method, runtime):
                    raise ValueError("External runtime/decoder weights differ across seeds")
                costs.append({"method": method, "seed": seed, "raw_draws": recipe["draws"],
                              "generation_seconds": raw.get("elapsed_seconds"),
                              "device": raw["runtime"].get("device")})
            labels = {"method": method, "seed": seed}
            for name in ("growth", "frontier"):
                tables[name].append(pd.read_csv(dest / f"{name}.csv").assign(**labels))
            tables["status"].append(pd.DataFrame([{**labels, **json.loads((dest / "status.json").read_text())}]))
            if (dest / "evaluation.json").exists():
                checked(dest, "evaluation.json", required=("evaluation_recipe.json", "metrics.json", "library.fasta"))
                er = json.loads((dest / "evaluation_recipe.json").read_text())
                if er["seed"] != 2027 or er["esm_model"] != "facebook/esm2_t33_650M_UR50D" or er["reference_sha256"] != sha256(Path(recipe["sources"]["reference"])):
                    raise ValueError("Evaluation settings differ")
                tables["official"].append(pd.DataFrame([{**labels, **json.loads((dest / "metrics.json").read_text())}]))
    output = root / "report"
    output.mkdir(exist_ok=True)
    for name, parts in tables.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    df = pd.concat(tables["growth"], ignore_index=True)
    cols = ["raw_joint_yield_per_1000", "separated_yield_per_1000", "raw_unique_novel_fraction", "invalid_draws"]
    summary = df.groupby(["method", "draws"])[cols].agg(["mean", "std", "count"])
    summary.to_csv(output / "growth_summary.csv")
    write_summary(output / "generation_costs.csv", costs)
    write_summary(output / "cross_seed_overlap.csv", [
        {"method": method, **row} for method, cells in samples.items()
        for row in cross_seed_overlap(cells)])
    paired = []
    for (seed, draws), group in df.groupby(["seed", "draws"]):
        index = group.set_index("method")
        for external in KITS:
            for control in METHODS:
                paired.append({"seed": int(seed), "draws": int(draws), "comparison": f"{external}_minus_{control}",
                               **{k: float(index.loc[external, k]-index.loc[control, k]) for k in cols}})
    write_summary(output / "paired_growth.csv", paired)
    matched = []
    for (seed, target), group in pd.concat(tables["frontier"]).groupby(["seed", "target_distance"]):
        index = group.set_index("method")
        for external in KITS:
            for control in METHODS:
                a, b = index.loc[external], index.loc[control]
                comparable = bool(a.complete and b.complete and abs(a.achieved_distance-b.achieved_distance) <= .01)
                matched.append({"seed": int(seed), "target_distance": float(target),
                    "comparison": f"{external}_minus_{control}", "matched": comparable,
                    "activity_delta": float(a.activity-b.activity) if comparable else None,
                    "evaluation_risk_delta": float(a.evaluation_risk-b.evaluation_risk) if comparable else None})
    write_summary(output / "matched_frontier.csv", matched)
    write_json(output / "report.json", {"audit_proofs": proofs, "external_runtimes": runtimes,
        "official_evaluations": len(tables["official"]), "deployment_approved": False,
        "limitations": ["Matched decoded draws, not matched compute or training data",
                        "HydrAMP raw is NON-DEFAULT: classifier filtering disabled; no claim about native submission performance",
                        "Both external native top100 selection pipelines are replaced by our common selector",
                        "HydrAMP maximum length25, diffusion10..40, internal8..50: inspect length strata",
                        "Predictors share our training-data lineage; not independent biological or competition ranking",
                        "No automatic rank or model promotion; empty/short selections are failures, not relaxed constraints"]})
    print(summary.to_string())
    print(f"[external] reports: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "sample", "score", "audit", "evaluate", "report"))
    parser.add_argument("--kit-root", type=Path, default=Path("external_baselines"))
    parser.add_argument("--controls", type=Path, default=Path("sweep_results/opd-evaluation-v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=tuple(KITS))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args(argv)
    if not 32 <= args.draws <= 200000 or args.draws % 32:
        raise ValueError("Draw budget must be a multiple of32 in32..200000")
    if args.command in ("preflight", "sample"):
        sources, control_cells = controls(args.controls, args.draws)
        kits = {m: inventory(args.kit_root / m, m) for m in KITS}
        separate_output(args.out, [args.controls, args.kit_root, *map(Path, sources["common"]["inputs"])])
        recipe = {"kind": "external_raw_v1", "sources": sources, "controls": control_cells, "kits": kits,
                  "draws": args.draws, "device": args.device, "code": code_identity(),
                  "scales": sorted({x for x in (2048, 8192, 32768, args.draws) if x <= args.draws})}
        verify_pins(recipe)
        if args.command == "preflight":
            print(json.dumps(recipe, indent=2))
            return
        prepare_run(args.out, recipe)
    else:
        recipe = json.loads((args.out / "run.json").read_text())
        if recipe["kind"] != "external_raw_v1" or recipe["code"] != code_identity():
            raise ValueError("Benchmark code/runtime changed: use original environment or a new run")
        verify_pins(recipe)
    cells = [(m, s) for m in KITS for s in (42, 43, 44)
             if (args.method is None or m == args.method) and (args.seed is None or s == args.seed)]
    if args.command == "sample":
        for method, seed in cells:
            print(f"[external] sampling {method}/seed{seed}; see generation.log", flush=True)
            raw_sample(args.out / "raw" / method / f"seed{seed}", recipe, method, seed)
    elif args.command == "score":
        from pilot_grpo import strict_runtime
        from validate_generator_scale import Scorers

        strict_runtime(42)
        scorers = Scorers(recipe, recipe["device"])
        strict_runtime(42)
        for method, seed in cells:
            score(args.out / "raw" / method / f"seed{seed}", recipe, scorers)
    elif args.command == "audit":
        import pandas as pd

        for cell, info in recipe["controls"].items():
            source = Path(info["root"])
            checked(source, required=("pool.csv",))
            if sha256(source / "complete.json") != info["marker"]:
                raise ValueError("Control changed")
            frame = pd.read_csv(source / "pool.csv", keep_default_na=False).iloc[:recipe["draws"]]
            audit_cell(args.out / "audit" / cell, frame, recipe, info)
        for method, seed in cells:
            source = args.out / "raw" / method / f"seed{seed}" / "scored"
            origin = checked_origin(recipe, method, seed, args.out)
            frame = pd.read_csv(source / "pool.csv", keep_default_na=False)
            audit_cell(args.out / "audit" / method / f"seed{seed}", frame, recipe,
                       origin)
    elif args.command == "evaluate":
        for method in (*METHODS, *KITS):
            for seed in (42, 43, 44):
                dest = args.out / "audit" / method / f"seed{seed}"
                checked(dest, required=("status.json",))
                if json.loads((dest / "status.json").read_text())["library_complete"]:
                    evaluate_library(dest, reference=Path(recipe["sources"]["reference"]),
                                     esm_model="facebook/esm2_t33_650M_UR50D", device=recipe["device"], seed=2027)
    elif args.command == "report":
        report(args.out, recipe)


if __name__ == "__main__":
    main()
