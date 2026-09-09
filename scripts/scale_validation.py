"""CPU diagnostics for frozen-generator scale validation; no policy updates."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from experiment_utils import sha256, verify_files
from generator_improvement import raw_metrics, validate_scores
from Levenshtein import ratio
from summarize_grpo_pilot import read_run


def checked(root, marker="complete.json", required=()):
    proof = json.loads((root / marker).read_text())
    if not set(required) <= set(proof["files"]) or any(
        Path(p).is_absolute() or ".." in Path(p).parts for p in proof["files"]
    ):
        raise ValueError("Invalid or incomplete artifact inventory")
    if not verify_files(root, marker):
        raise ValueError("Missing completion marker")
    return proof


def sources(pilot_root):
    """Pin all six endpoints and the original common generator/scorers."""
    inventory, signatures, common = {}, {}, None
    for method in ("raft", "grpo"):
        for seed in (42, 43, 44):
            root = pilot_root / f"{method}-generator-seed{seed}-v1"
            _, signature = read_run(root, improvement=True)
            run = json.loads((root / "run.json").read_text())
            checked(root, required=("policy/model.pt", "policy/config.json"))
            if run["args"]["method"] != method or run["args"]["seed"] != seed:
                raise ValueError("Source endpoint identity mismatch")
            if signature != signatures.setdefault(method, signature):
                raise ValueError("Within-method pilot recipes differ")
            identity = {k: run[k] for k in ("inputs", "configs", "revision", "sampling", "evaluator_marker", "code")}
            if common is not None and identity != common:
                raise ValueError("Pilot inputs/scorers/code differ across methods")
            common = identity
            for p, digest in run["inputs"].items():
                if sha256(Path(p)) != digest:
                    raise ValueError(f"Pinned input changed: {p}")
            inventory[f"{method}/seed{seed}"] = {
                "root": str(root.resolve()), "marker_sha256": sha256(root / "complete.json"),
                "checkpoint": str((root / "policy").resolve()),
                "sampling_seed": 40000 + seed,
            }
            inventory.setdefault(f"baseline/seed{seed}", {
                "checkpoint": str(Path(run["args"]["checkpoint"]).resolve()),
                "sampling_seed": 40000 + seed,
            })
    return {"common": common, "cells": inventory, "reference": str(Path(run["args"]["reference"]).resolve()),
            "evaluator": str(Path(run["args"]["evaluator"]).resolve())}


def library_ids(sequences, reference, size=50000):
    seen, ids = set(), []
    for i, s in enumerate(sequences):
        if not 8 <= len(s) <= 50 or set(s) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("Invalid generated sequence; no silent filtering")
        if s in seen or s in reference:
            continue
        seen.add(s)
        ids.append(i)
        if len(ids) == size:
            break
    return ids


def separated_ids(frame, limit=None, target_distance=0.):
    """Score-ordered greedy set; enforce similarity <.8 and cumulative mean floor."""
    order = frame.assign(quality=frame.activity-frame.risk).sort_values(
        ["quality", "sequence"], ascending=[False, True])
    ids, selected, total = [], [], 0.
    for i, row in order.iterrows():
        if target_distance == 0.:
            if any(ratio(row.sequence, s, score_cutoff=.8) >= .8 for s in selected):
                continue
            distances = []
        else:
            distances = []
            for s in selected:
                distance = 1-ratio(row.sequence, s)
                if distance <= .2 + 1e-12:
                    break
                distances.append(distance)
            if len(distances) != len(selected):
                continue
        if target_distance and any(d <= .2 + 1e-12 for d in distances):
            continue
        n = len(selected)
        proposed = total + sum(distances)
        if target_distance and n and proposed / ((n+1)*n/2) < target_distance:
            continue
        ids.append(i)
        selected.append(row.sequence)
        total = proposed
        if limit is not None and len(ids) == limit:
            break
    return ids


def reference_violation(sequence, references):
    """Exact validator semantics: reject >.8, allow equality; safe length pruning."""
    for ref in references:
        if 2*min(len(sequence), len(ref))/(len(sequence)+len(ref)) <= .8:
            continue
        if ratio(sequence, ref, score_cutoff=.8) > .8:
            return True
    return False


def validate_pool(frame):
    validate_scores(frame.sequence.tolist(), frame.activity.to_numpy(), frame.risk.to_numpy(),
                    frame.evaluation_risk.to_numpy())
    library_ids(frame.sequence.tolist(), set(), size=len(frame))


def growth(frame, reference, scales):
    validate_pool(frame)
    rows, lengths = [], []
    for count in scales:
        if count > len(frame):
            raise ValueError("Scale exceeds saved pool")
        sub = frame.iloc[:count]
        metrics = raw_metrics(sub.sequence.tolist(), sub.activity.to_numpy(), sub.risk.to_numpy(),
                              sub.evaluation_risk.to_numpy(), reference)
        novel = sub[~sub.sequence.isin(reference)].drop_duplicates("sequence")
        passing = novel[(novel.activity >= .6) & (novel.risk <= .5) & (novel.evaluation_risk <= .5)]
        ids = separated_ids(passing)
        rows.append({"draws": count, **metrics, "joint_pass_unique": len(passing),
                     "joint_pass_separated_08": len(ids), "separated_yield_per_1000": 1000*len(ids)/count})
        for low, high in ((8, 14), (15, 24), (25, 34), (35, 50)):
            part = sub[sub.sequence.str.len().between(low, high)]
            good = passing[passing.sequence.str.len().between(low, high)]
            lengths.append({"draws": count, "length_bin": f"{low}-{high}", "bin_draws": len(part),
                            "raw_share": len(part)/count,
                            "activity_mean": float(part.activity.mean()) if len(part) else None,
                            "evaluation_risk_mean": float(part.evaluation_risk.mean()) if len(part) else None,
                            "joint_yield_per_1000_bin_draws": 1000*len(good)/len(part) if len(part) else None})
        print(f"[scale-audit] draws={count} passing={len(passing)} separated={len(ids)}", flush=True)
    return rows, lengths


def frontier(frame, reference, shortlist=2000, top=100, targets=(.5, .55, .6, .65)):
    """Choose by training scores only; evaluation risk is used for reporting only."""
    eligible = frame[(frame.activity >= .6) & ~frame.sequence.isin(reference)].drop_duplicates("sequence")
    eligible = eligible.assign(quality=eligible.activity-eligible.risk).sort_values(
        ["quality", "sequence"], ascending=[False, True]).head(shortlist)
    references = sorted(reference)
    accepted = []
    for j, (i, row) in enumerate(eligible.iterrows()):
        if not reference_violation(row.sequence, references):
            accepted.append(i)
        if (j+1) % 250 == 0:
            print(f"[scale-audit] reference novelty checked {j+1}/{len(eligible)}", flush=True)
    pool = eligible.loc[accepted]
    summaries, selections = [], []
    for target in targets:
        ids = separated_ids(pool, limit=top, target_distance=target)
        selected = pool.loc[ids]
        distances = [1-ratio(a, b) for a, b in combinations(selected.sequence, 2)]
        summaries.append({"target_distance": target, "selected": len(ids), "complete": len(ids) == top,
                          "shortlist_checked": len(eligible), "shortlist_reference_pass": len(pool),
                          "achieved_distance": float(np.mean(distances)) if distances else None,
                          "activity": float(selected.activity.mean()) if len(ids) else None,
                          "reward_risk": float(selected.risk.mean()) if len(ids) else None,
                          "evaluation_risk": float(selected.evaluation_risk.mean()) if len(ids) else None,
                          "mean_length": float(selected.sequence.str.len().mean()) if len(ids) else None})
        selections.extend({"target_distance": target, "rank": rank, **row.to_dict()}
                          for rank, (_, row) in enumerate(selected.iterrows(), 1))
    return summaries, selections
