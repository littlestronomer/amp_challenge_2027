"""Sequence-family held-out predictor benchmark, deliberately separate from deployment.

Levenshtein ratio is normalized indel similarity, not aligned biological identity.
Connected components (not leader clusters) enforce the specified pairwise boundary.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import AMINO_ACIDS, MAX_LENGTH, MIN_LENGTH, PANEL_GENERA
from amp_challenge_2027.data import SPECIES_TO_PANEL
from amp_challenge_2027.reward_audit import discrimination

TASKS = ("activity", "panel", "hemolysis")
SPLITS = ("train", "validation", "calibration", "test")
FRACTIONS = (.60, .15, .10, .15)


def curate_labels(path: Path, task: str) -> tuple[list[dict], list[dict], list[str], dict]:
    """One record per sequence. Mask conflicting output labels; never take last/max.

    Binary activity is explicitly restricted to unanimous measured contexts. It is
    NOT an organism-independent potency label. Conflicting contexts are retained
    in the audit, not silently called inactive or resolved by a majority vote.
    """
    if task not in TASKS:
        raise ValueError("Unknown task")
    outputs = list(PANEL_GENERA) if task == "panel" else ["risky" if task == "hemolysis" else "active"]
    evidence = defaultdict(list)
    counts = Counter()
    bridges = set()
    events = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sequence", "label"} <= set(reader.fieldnames or []):
            raise ValueError(f"Missing sequence/label columns: {path}")
        for line, row in enumerate(reader, 2):
            counts["raw_rows"] += 1
            seq = row["sequence"].strip().upper()
            raw = row["label"].strip().lower()
            label = 1 if raw in {"active", "1", "1.0", "true"} else 0 if raw in {"inactive", "0", "0.0", "false"} else None
            organism = (row.get("organism") or "").strip()
            reason = None
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH or set(seq) - set(AMINO_ACIDS):
                reason = "invalid_sequence"
            else:
                # Also retain valid but excluded sequences as family-graph bridges.
                bridges.add(seq)
            if reason is None and label is None:
                reason = "unknown_label"
            output = outputs[0]
            if reason is None and task == "panel":
                matches = {organism} if organism in outputs else {
                    genus for species, genus in SPECIES_TO_PANEL.items()
                    if species.lower() in organism.lower()
                }
                if len(matches) != 1 or not matches <= set(outputs):
                    reason = "unmapped_or_ambiguous_organism"
                else:
                    output = next(iter(matches))
            if reason:
                counts[reason] += 1
                events.append({"task": task, "csv_line": line, "sequence": seq,
                               "organism": organism, "label": raw, "output": "", "reason": reason})
            else:
                evidence[(seq, output)].append((label, line, organism))
    per_seq = {}
    for (seq, output), rows in sorted(evidence.items()):
        values = {r[0] for r in rows}
        counts["duplicate_evidence_rows"] += len(rows) - 1
        if len(values) > 1:
            counts["conflicting_sequence_output_groups"] += 1
            for label, line, organism in rows:
                events.append({"task": task, "csv_line": line, "sequence": seq, "organism": organism,
                               "label": label, "output": output, "reason": "conflicting_output_masked"})
            continue
        rec = per_seq.setdefault(seq, {"sequence": seq, "targets": [0] * len(outputs), "mask": [0] * len(outputs)})
        j = outputs.index(output)
        rec["targets"][j], rec["mask"][j] = rows[0][0], 1
    records = [per_seq[s] for s in sorted(per_seq)]
    counts["retained_unique_sequences"] = len(records)
    counts["graph_sequences_including_excluded_bridges"] = len(bridges)
    return records, events, sorted(bridges), {"counts": dict(counts), "outputs": outputs}


def family_components(sequences: list[str], threshold: float, *, progress=None) -> dict[str, int]:
    """Exact all-pair threshold graph, bounded memory union-find including bridges."""
    from Levenshtein import ratio

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    seqs = sorted(set(sequences), key=lambda s: (len(s), s))
    if not seqs or any(not s for s in seqs):
        raise ValueError("Require nonempty sequences")
    parent = list(range(len(seqs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, seq in enumerate(seqs):
        for j in range(i + 1, len(seqs)):
            # Length bound is exact for normalized indel similarity. Lengths sorted.
            if 2 * len(seq) / (len(seq) + len(seqs[j])) < threshold:
                break
            if ratio(seq, seqs[j]) >= threshold:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
        if progress and (i % 500 == 0 or i == len(seqs) - 1):
            progress(f"family graph {i + 1}/{len(seqs)}")
    return {s: root(i) for i, s in enumerate(seqs)}


def assign_families(groups: dict[str, int], seed: int) -> dict[str, str]:
    """Label-blind, fixed-seed largest-first balancing by unique sequence count.

    No seed search, class stratification, or family breaking to improve scores.
    A giant connected component can make target fractions unattainable.
    """
    sizes = Counter(groups.values())
    if len(sizes) < len(SPLITS):
        raise ValueError("Fewer than four families: cannot build a four-way holdout")
    ids = sorted(sizes)
    np.random.default_rng(seed).shuffle(ids)
    ids.sort(key=lambda group: -sizes[group])
    target = np.array(FRACTIONS) * len(groups)
    used = np.zeros(len(SPLITS))
    assignment = {}
    for group in ids:
        dest = int(np.argmax(target - used))
        assignment[group] = SPLITS[dest]
        used[dest] += sizes[group]
    if (used == 0).any():
        raise ValueError("Empty partition; do not lower the boundary or search seeds based on test performance")
    return {seq: assignment[group] for seq, group in groups.items()}


def audit_boundary(assignment: dict[str, str], threshold: float, *, progress=None) -> dict:
    """Independent exhaustive cross-partition check, including non-training pairs."""
    from Levenshtein import ratio

    seqs = sorted(assignment, key=lambda s: (len(s), s))
    checks = violations = 0
    for i, seq in enumerate(seqs):
        for other in seqs[i + 1:]:
            if 2 * len(seq) / (len(seq) + len(other)) < threshold:
                break
            if assignment[seq] != assignment[other]:
                checks += 1
                violations += ratio(seq, other) >= threshold
        if progress and i % 500 == 0:
            progress(f"independent boundary check {i + 1}/{len(seqs)}")
    if violations:
        raise ValueError(f"Cross-partition similarity violations: {violations}")
    return {"threshold": threshold, "edge_rule": "Levenshtein.ratio >= threshold",
            "guarantee": "all cross-partition ratios strictly below threshold",
            "exact_cross_partition_comparisons": checks, "violations": int(violations),
            "length_prefilter": "exact normalized-indel upper bound; no approximate search"}


def arrays(records: list[dict], split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [r for r in records if r["split"] == split]
    width = len(records[0]["targets"])
    return (np.array([r["targets"] for r in selected], dtype=np.float32).reshape(-1, width),
            np.array([r["mask"] for r in selected], dtype=np.float32).reshape(-1, width),
            [r["sequence"] for r in selected])


def support_report(records: list[dict], outputs: list[str], task: str) -> list[dict]:
    rows = []
    for split in SPLITS:
        y, mask, seqs = arrays(records, split)
        families = {r["family"] for r in records if r["split"] == split}
        for j, output in enumerate(outputs):
            labels = y[mask[:, j] == 1, j]
            rows.append({"task": task, "split": split, "output": output, "sequences": len(seqs),
                         "families": len(families), "observed": len(labels), "positives": int(labels.sum()),
                         "negatives": int(len(labels) - labels.sum()),
                         "both_classes": len(set(labels)) == 2})
    return rows


def macro_rank(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    values = [discrimination(y[mask[:, j] == 1, j], logits[mask[:, j] == 1, j])["auroc"]
              for j in range(y.shape[1])]
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def calibrate_temperature(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """Fit only a positive scalar on calibration labels; equal output weighting.

    Positive T preserves ranking. No threshold optimization or test labels here.
    """
    if logits.shape != y.shape or mask.shape != y.shape or not np.isfinite(logits).all():
        raise ValueError("Invalid calibration arrays")
    supported = [j for j in range(y.shape[1]) if len(set(y[mask[:, j] == 1, j])) == 2]
    if not supported:
        return {"temperature": 1., "status": "not_fitted_no_two_class_outputs", "outputs": []}
    grid = np.unique(np.r_[1., np.geomspace(.1, 10., 401)])
    losses = []
    for t in grid:
        losses.append(np.mean([np.mean(np.logaddexp(0, logits[mask[:, j] == 1, j] / t)
                                      - y[mask[:, j] == 1, j] * logits[mask[:, j] == 1, j] / t)
                               for j in supported]))
    best = int(np.argmin(losses))
    return {"temperature": float(grid[best]), "status": "fitted_calibration_only",
            "outputs": supported, "macro_log_loss": float(losses[best]),
            "at_grid_boundary": best in {0, len(grid) - 1}, "grid": "0.1..10 log-spaced 401 plus 1"}


def family_bootstrap(logits: np.ndarray, y: np.ndarray, mask: np.ndarray, families: list[int],
                     *, replicates: int = 1000, seed: int = 2027, comparator=None) -> dict:
    """Family-resampled AUROC intervals, with fixed evaluable output set.

    Undefined resamples are counted and omitted, never replaced by chance. When
    supplied, comparator uses the identical resampled families for paired deltas.
    """
    if (replicates < 1 or logits.shape != y.shape or y.shape != mask.shape or len(families) != len(y)
            or (comparator is not None and comparator.shape != logits.shape)):
        raise ValueError("Invalid family bootstrap inputs")
    groups = np.unique(families)
    outputs = [j for j in range(y.shape[1]) if len(set(y[mask[:, j] == 1, j])) == 2]
    if not outputs or len(groups) < 2:
        return {"status": "insufficient_support", "families": len(groups), "valid_replicates": 0}
    members = [np.flatnonzero(np.array(families) == g) for g in groups]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        ids = np.concatenate([members[g] for g in rng.integers(len(groups), size=len(groups))])
        if any(len(set(y[ids, j][mask[ids, j] == 1])) != 2 for j in outputs):
            continue
        val = macro_rank(logits[np.ix_(ids, outputs)], y[np.ix_(ids, outputs)], mask[np.ix_(ids, outputs)])
        if comparator is not None:
            val -= macro_rank(comparator[np.ix_(ids, outputs)], y[np.ix_(ids, outputs)], mask[np.ix_(ids, outputs)])
        values.append(val)
    sufficient = len(values) >= max(100, .8 * replicates)
    estimate = macro_rank(logits[:, outputs], y[:, outputs], mask[:, outputs])
    if comparator is not None:
        estimate -= macro_rank(comparator[:, outputs], y[:, outputs], mask[:, outputs])
    return {"status": "descriptive_family_bootstrap" if sufficient else "insufficient_valid_resamples",
            "families": len(groups), "low_family_support_lt_20": len(groups) < 20,
            "estimate": estimate, "outputs": outputs, "requested_replicates": replicates,
            "valid_replicates": len(values), "undefined_replicates": replicates - len(values),
            "percentile_95_lower": float(np.quantile(values, .025)) if sufficient else None,
            "percentile_95_upper": float(np.quantile(values, .975)) if sufficient else None,
            "scope": "conditional on this split and pretrained representation; not external validation"}
