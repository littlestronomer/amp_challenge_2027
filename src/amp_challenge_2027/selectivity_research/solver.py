"""Deterministic constrained top-k selection over a frozen scored pool.

The solver deliberately operates on cached scores.  It does not train a model or
silently change the generator; every constraint is recorded in the returned
recipe and can therefore be audited independently of the MILP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amp_challenge_2027.config import MDR_PANEL_GENERA, TOP_SIMILARITY_THRESHOLD

from .constraints import baseline_indices
from .metrics import summarize


@dataclass(frozen=True)
class SolveResult:
    status: str
    message: str
    indices: np.ndarray
    summary: dict
    metadata: dict

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "indices": [int(x) for x in self.indices],
            "summary": self.summary,
            "metadata": self.metadata,
        }


def _pairs(sequences: list[str], indices: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    from Levenshtein import ratio

    conflicts = []
    for pos, left in enumerate(indices):
        for right in indices[:pos]:
            if ratio(sequences[int(left)], sequences[int(right)]) >= threshold:
                conflicts.append((int(left), int(right)))
    return conflicts


def _reference_conflicts(sequences: list[str], indices: np.ndarray, reference: set[str] | None,
                         threshold: float) -> list[int]:
    if not reference:
        return []
    from Levenshtein import ratio

    bad = []
    for index in indices:
        sequence = sequences[int(index)]
        if max(ratio(sequence, target) for target in reference) > threshold:
            bad.append(int(index))
    return bad


def _milp(frame, candidates: np.ndarray, genera: list[str], protocol: dict, profile: str,
          forbidden: set[int], *, time_limit: float, mip_rel_gap: float):
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        return None, "scipy is required; install the project ml extra", "failed_dependency"

    arrays = {
        "activity": frame.iloc[candidates]["activity"].to_numpy(float),
        "risk": frame.iloc[candidates]["hemolysis_risk"].to_numpy(float),
        "breadth": (frame.iloc[candidates][[f"panel:{g}" for g in genera]].to_numpy(float) > 0.5).mean(axis=1),
    }
    panel = frame.iloc[candidates][[f"panel:{g}" for g in genera]].to_numpy(float)
    mdr = [j for j, genus in enumerate(genera) if genus in MDR_PANEL_GENERA]
    arrays["mdr"] = (panel[:, mdr] > 0.5).mean(axis=1) if mdr else np.zeros(len(candidates))

    # Linear lower bounds are based on the frozen incumbent, with tolerances
    # declared in the protocol rather than hidden in the implementation.
    inc = baseline_indices(frame)
    inc_panel = frame.iloc[inc][[f"panel:{g}" for g in genera]].to_numpy(float)
    inc_breadth = (inc_panel > 0.5).mean(axis=1).mean()
    inc_mdr = ((inc_panel[:, mdr] > 0.5).mean(axis=1).mean() if mdr else 0.0)
    inc_activity = frame.iloc[inc]["activity"].mean()
    inc_tail = int((frame.iloc[inc]["hemolysis_risk"].to_numpy(float) > float(protocol["risk_tail_threshold"])).sum())
    row_count = len(genera) + 6
    matrix = lil_matrix((row_count, len(candidates)), dtype=float)
    lower = np.full(row_count, -np.inf, dtype=float)
    upper = np.full(row_count, np.inf, dtype=float)
    row = 0
    matrix[row, :] = 1.0; lower[row] = upper[row] = 100.0; row += 1
    matrix[row, :] = arrays["activity"]; lower[row] = 100.0 * (inc_activity - float(protocol["activity_drop_max"])); row += 1
    for j in range(len(genera)):
        matrix[row, :] = panel[:, j]
        lower[row] = 100.0 * (inc_panel[:, j].mean() - float(protocol["panel_drop_max"]))
        row += 1
    matrix[row, :] = arrays["breadth"]; lower[row] = 100.0 * (inc_breadth - float(protocol["breadth_drop_max"])); row += 1
    matrix[row, :] = arrays["mdr"]; lower[row] = 100.0 * (inc_mdr - float(protocol["mdr_drop_max"])); row += 1
    matrix[row, :] = (arrays["risk"] > float(protocol["risk_tail_threshold"])).astype(float)
    upper[row] = inc_tail; row += 1
    # The final row is a harmless redundant bound that makes the generated
    # matrix shape explicit and stable for manifests.
    matrix[row, :] = arrays["risk"]; lower[row] = -np.inf; upper[row] = np.inf

    ceiling = float(profile)
    ub = np.ones(len(candidates), dtype=float)
    if ceiling < 1.0:
        ub[arrays["risk"] > ceiling] = 0.0
    for local, full in enumerate(candidates):
        if int(full) in forbidden:
            ub[local] = 0.0
    try:
        result = milp(c=arrays["risk"], integrality=np.ones(len(candidates)),
                      bounds=Bounds(np.zeros(len(candidates)), ub),
                      constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                      options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap),
                               "presolve": True})
    except Exception as exc:  # retain a stable report for solver/version failures
        return None, str(exc), "failed_solver"
    if result.x is None:
        return result, str(result.message), "infeasible" if result.status == 2 else "failed_solver"
    selected = candidates[np.flatnonzero(result.x > 0.5)]
    status = "optimal" if result.status == 0 else "feasible_unproven_optimal"
    return result, status, str(result.message), selected


def solve_pool(frame, eligible: np.ndarray, genera: list[str], protocol: dict, profile: str = "1.0",
               *, reference: set[str] | None = None, time_limit: float = 300.0,
               mip_rel_gap: float = 1e-3, max_cut_rounds: int = 20) -> SolveResult:
    """Solve one risk-ceiling profile and add exact novelty/diversity cuts."""
    candidates = np.flatnonzero(np.asarray(eligible, dtype=bool)).astype(np.int64)
    forbidden: set[int] = set()
    metadata = {"profile": profile, "candidate_count": int(len(candidates)),
                "cut_rounds": 0, "pair_cuts": 0, "reference_cuts": 0,
                "time_limit_seconds": float(time_limit), "mip_rel_gap": float(mip_rel_gap)}
    if len(candidates) < 100:
        return SolveResult("infeasible", "fewer than 100 eligible candidates", np.array([], dtype=int), {}, metadata)
    try:
        base = baseline_indices(frame)
    except ValueError as exc:
        return SolveResult("failed_validation", str(exc), np.array([], dtype=int), {}, metadata)
    if not np.all(np.asarray(eligible, dtype=bool)[base]):
        return SolveResult("failed_validation", "incumbent contains an ineligible row", np.array([], dtype=int), {}, metadata)
    sequences = frame["sequence"].astype(str).tolist()
    last_status = "failed_solver"
    last_message = ""
    for round_index in range(max_cut_rounds + 1):
        outcome = _milp(frame, candidates, genera, protocol, profile, forbidden,
                        time_limit=time_limit, mip_rel_gap=mip_rel_gap)
        if len(outcome) == 3:
            _result, last_message, last_status = outcome
            break
        result, status, message, selected = outcome
        last_status, last_message = status, message
        pair_conflicts = _pairs(sequences, selected, TOP_SIMILARITY_THRESHOLD)
        novelty_conflicts = _reference_conflicts(sequences, selected, reference, TOP_SIMILARITY_THRESHOLD)
        if novelty_conflicts:
            forbidden.update(novelty_conflicts)
            metadata["reference_cuts"] += len(novelty_conflicts)
        if pair_conflicts:
            # Re-solving with a candidate-level cut for every member is a
            # conservative valid cut and avoids retaining a clustered top set.
            for left, right in pair_conflicts:
                forbidden.add(max(left, right))
            metadata["pair_cuts"] += len(pair_conflicts)
        metadata["cut_rounds"] = round_index + 1
        if not pair_conflicts and not novelty_conflicts:
            summary = summarize(frame, selected, genera, incumbent=base)
            metadata.update({"objective_risk_mean": float(result.fun / 100.0),
                             "solver_status_code": int(result.status),
                             "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0)})
            accepted = (summary["pairwise_distance_mean"] >= summary["distance_floor"])
            if not accepted:
                last_status = "failed_validation"
                last_message = "post-solve pairwise distance is below incumbent floor"
            return SolveResult(last_status, last_message, selected.astype(int), summary, metadata)
        if len(forbidden) >= len(candidates) - 100:
            break
    return SolveResult(last_status, last_message, np.array([], dtype=int), {}, metadata)
