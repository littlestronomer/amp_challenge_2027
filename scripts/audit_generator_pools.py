"""Read-only accounting of completed external-benchmark pools; no training/scoring."""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from benchmark_external import KITS, METHODS, checked_origin, valid
from experiment_utils import code_identity, mark_files, prepare_run, sha256, write_summary
from scale_validation import checked


def accounting(frame, reference):
    """Reference occurrences take precedence over repetition: categories sum to draws."""
    counts = Counter()
    seen = set()
    rows = []
    for sequence in frame.sequence:
        if not valid(sequence):
            category = "invalid"
        elif sequence in reference:
            category = "exact_reference"
        elif sequence in seen:
            category = "repeated_nonreference"
        else:
            category = "unique_nonreference"
        seen.add(sequence)
        counts[category] += 1
        rows.append(category)
    result = {k: counts[k] for k in ("invalid", "exact_reference", "repeated_nonreference", "unique_nonreference")}
    result["draws"] = len(frame)
    result["reference_unique"] = len(set(frame.sequence) & reference)
    return result, rows


def diagnose(frame, reference, scales):
    from Levenshtein import ratio

    totals, strata, families = [], [], []
    for count in scales:
        if count <= 0 or count > len(frame):
            raise ValueError("Invalid prefix budget")
        sub = frame.iloc[:count].copy()
        totals_row, categories = accounting(sub, reference)
        sub["category"] = categories
        valid_rows = sub[sub.sequence.map(valid)]
        for column in ("activity", "risk", "evaluation_risk"):
            values = pd.to_numeric(valid_rows[column], errors="raise")
            if not values.between(0, 1).all():
                raise ValueError(f"Invalid cached probabilities: {column}")
        passing = sub[(sub.category == "unique_nonreference") & (sub.activity >= .6)
                      & (sub.risk <= .5) & (sub.evaluation_risk <= .5)]
        # Deterministic greedy representative groups, NOT connected components.
        representatives, sizes = [], []
        for sequence in sorted(passing.sequence):
            for i, representative in enumerate(representatives):
                if ratio(sequence, representative, score_cutoff=.8) >= .8:
                    sizes[i] += 1
                    break
            else:
                representatives.append(sequence)
                sizes.append(1)
        totals_row.update(qualifying_unique=len(passing), qualifying_greedy_families=len(sizes),
                          qualifying_yield_per_1000=1000*len(passing)/count,
                          family_yield_per_1000=1000*len(sizes)/count)
        totals.append(totals_row)
        for representative, size in zip(representatives, sizes):
            families.append({"draws": count, "representative": representative, "size": size})
        for low, high in ((8, 14), (15, 24), (25, 34), (35, 50)):
            part = sub[sub.sequence.str.len().between(low, high)]
            good = passing[passing.sequence.str.len().between(low, high)]
            strata.append({"draws": count, "length_bin": f"{low}-{high}", "bin_draws": len(part),
                           "qualifying_unique": len(good),
                           **{k: int((part.category == k).sum()) for k in
                              ("invalid", "exact_reference", "repeated_nonreference", "unique_nonreference")}})
    return totals, strata, families


def main():
    from amp_challenge_2027.data import iter_fasta

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("sweep_results/external-pilot-v2"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    original = json.loads((source / "run.json").read_text())
    if original.get("kind") != "external_raw_v1":
        raise ValueError("Expected completed external benchmark source")
    reference_path = Path(original["sources"]["reference"])
    expected_reference = original["sources"]["common"]["inputs"].get(str(reference_path.resolve()))
    if expected_reference != sha256(reference_path):
        raise ValueError("Reference differs from frozen scoring inputs")
    reference = {s for _, s in iter_fasta(reference_path)}
    pools, proofs = [], {}
    for method in (*METHODS, *KITS):
        for seed in (42, 43, 44):
            origin = checked_origin(original, method, seed, source)
            pool = (Path(original["controls"][f"{method}/seed{seed}"]["root"]) / "pool.csv"
                    if method in METHODS else source / "raw" / method / f"seed{seed}/scored/pool.csv")
            pools.append((method, seed, pool))
            proofs[f"{method}/{seed}"] = {"origin": origin, "path": str(pool), "sha256": sha256(pool)}
    protected = [source, reference_path.resolve(), *(p.resolve() for _, _, p in pools)]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output must be separate from inputs")
    manifest = {"kind": "pool_accounting_v1", "source_manifest": sha256(source / "run.json"),
                "reference_sha256": sha256(reference_path), "pools": proofs, "code": code_identity(),
                "thresholds": {"activity_min": .6, "risk_max": .5, "evaluation_risk_max": .5},
                "limitations": ["Greedy lexicographic representative families, not biological families",
                                "Evaluation risk used diagnostically; not independent validation",
                                "No near-reference or reward-training-set similarity audit in this stage"]}
    prepare_run(output, manifest)
    if (output / "complete.json").exists():
        checked(output, required=("run.json", "pool_accounting.csv", "length_strata.csv"))
        print("Verified completed audit")
        return
    tables = [[], [], []]
    for method, seed, pool in pools:
        frame = pd.read_csv(pool, keep_default_na=False).iloc[:original["draws"]]
        if len(frame) != original["draws"]:
            raise ValueError("Pool shorter than declared draw budget")
        for col in ("activity", "risk", "evaluation_risk"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        print(f"[pool-audit] {method}/seed{seed}", flush=True)
        for target, rows in zip(tables, diagnose(frame, reference, original["scales"])):
            target.extend({"method": method, "seed": seed, **row} for row in rows)
        if sha256(pool) != proofs[f"{method}/{seed}"]["sha256"]:
            raise ValueError("Pool changed during audit")
    names = []
    if sha256(reference_path) != manifest["reference_sha256"]:
        raise ValueError("Reference changed during audit")
    for name, rows in zip(("pool_accounting", "length_strata", "qualifying_families"), tables):
        if rows:
            write_summary(output / f"{name}.csv", rows)
            names.append(f"{name}.csv")
    mark_files(output, "complete.json", ["run.json", *names])
    print(f"[pool-audit] complete: {output}")


if __name__ == "__main__":
    main()
