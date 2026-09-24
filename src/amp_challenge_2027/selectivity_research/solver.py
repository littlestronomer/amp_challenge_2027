"""Deterministic constrained top-k selection over a frozen scored pool.

The solver deliberately operates on cached scores.  It does not train a model or
silently change the generator; every constraint is recorded in the returned
recipe and can therefore be audited independently of the MILP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from amp_challenge_2027.config import MDR_PANEL_GENERA, TOP_SIMILARITY_THRESHOLD

from .constraints import baseline_indices, linear_bounds
from .contracts import validate_pool_frame, validate_solution
from .metrics import summarize

SOLVER_VERSION = "pair-cuts-v2"
FEASIBLE_STATUSES = {"optimal", "feasible_unproven_optimal"}


def constraint_margins(frame, selected, genera, protocol, profile):
    """Independent score checks; no reuse of the sparse constraint matrix."""
    validate_solution(selected, len(frame))
    a, _, floors = linear_bounds(frame, genera, protocol)
    summary = summarize(frame, selected, genera)
    margins = {
        "activity": float(a["activity"][selected].mean() - floors["activity"]),
        "breadth": float(a["breadth"][selected].mean() - floors["breadth"]),
        "mdr": float(a["mdr"][selected].mean() - floors["mdr"]),
        "risk_tail": float(
            floors["risk_tail"] - (a["risk"][selected] > protocol["risk_tail_threshold"]).sum()
        ),
        "risk_ceiling": float(float(profile) - a["risk"][selected].max()),
        "distance": float(summary["pairwise_distance_mean"] - floors["distance"]),
    }
    margins.update(
        {
            f"panel:{g}": float(a["panel"][selected, j].mean() - floors["panel"][j])
            for j, g in enumerate(genera)
        }
    )
    return margins


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


def _reference_conflicts(
    sequences: list[str], indices: np.ndarray, reference: set[str] | None, threshold: float
) -> list[int]:
    if not reference:
        return []
    from Levenshtein import ratio

    bad = []
    for index in indices:
        sequence = sequences[int(index)]
        if max(ratio(sequence, target) for target in reference) > threshold:
            bad.append(int(index))
    return bad


def _baseline_is_feasible(
    frame, base: np.ndarray, profile: str, reference: set[str] | None
) -> bool:
    """Check the incumbent independently before trusting a MILP infeasibility."""
    ceiling = float(profile)
    risk = frame.iloc[base]["hemolysis_risk"].to_numpy(float)
    if ceiling < 1.0 and (risk > ceiling).any():
        return False
    sequences = frame["sequence"].astype(str).tolist()
    return not _pairs(sequences, base, TOP_SIMILARITY_THRESHOLD) and not _reference_conflicts(
        sequences, base, reference, TOP_SIMILARITY_THRESHOLD
    )


def _incumbent_fallback(frame, base, genera, protocol, profile, reference, message, metadata):
    if _baseline_is_feasible(frame, base, profile, reference):
        margins = constraint_margins(frame, base, genera, protocol, profile)
        if any(value < -1e-7 for value in margins.values()):
            return None
        metadata = dict(metadata)
        metadata["fallback"] = "incumbent_feasible_after_milp_failure"
        metadata["solution_origin"] = "incumbent_fallback"
        metadata["constraint_margins"] = margins
        metadata["validated"] = True
        metadata["improved_vs_incumbent"] = False
        summary = summarize(frame, base, genera, incumbent=base)
        summary["activity_floor"] = summary["activity_mean"] - protocol["activity_drop_max"]
        summary["distance_floor"] = (
            summary["pairwise_distance_mean"] - protocol["distance_drop_max"]
        )
        return SolveResult(
            "feasible_unproven_optimal",
            message + "; incumbent independently verified",
            base.astype(int),
            summary,
            metadata,
        )
    return None


def _milp(
    frame,
    candidates: np.ndarray,
    genera: list[str],
    protocol: dict,
    profile: str,
    forbidden: set[int],
    pair_cuts: set[tuple[int, int]],
    *,
    time_limit: float,
    mip_rel_gap: float,
):
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:  # pragma: no cover - exercised on minimal installs
        return None, "scipy is required; install the project ml extra", "failed_dependency"

    arrays = {
        "activity": frame.iloc[candidates]["activity"].to_numpy(float),
        "risk": frame.iloc[candidates]["hemolysis_risk"].to_numpy(float),
        "breadth": (
            frame.iloc[candidates][[f"panel:{g}" for g in genera]].to_numpy(float) > 0.5
        ).mean(axis=1),
    }
    panel = frame.iloc[candidates][[f"panel:{g}" for g in genera]].to_numpy(float)
    mdr = [j for j, genus in enumerate(genera) if genus in MDR_PANEL_GENERA]
    arrays["mdr"] = (panel[:, mdr] > 0.5).mean(axis=1) if mdr else np.zeros(len(candidates))

    # Linear lower bounds are based on the frozen incumbent, with tolerances
    # declared in the protocol rather than hidden in the implementation.
    inc = baseline_indices(frame)
    inc_panel = frame.iloc[inc][[f"panel:{g}" for g in genera]].to_numpy(float)
    inc_breadth = (inc_panel > 0.5).mean(axis=1).mean()
    inc_mdr = (inc_panel[:, mdr] > 0.5).mean(axis=1).mean() if mdr else 0.0
    inc_activity = frame.iloc[inc]["activity"].mean()
    inc_tail = int(
        (
            frame.iloc[inc]["hemolysis_risk"].to_numpy(float)
            > float(protocol["risk_tail_threshold"])
        ).sum()
    )
    row_count = len(genera) + 5 + len(pair_cuts)
    matrix = lil_matrix((row_count, len(candidates)), dtype=float)
    lower = np.full(row_count, -np.inf, dtype=float)
    upper = np.full(row_count, np.inf, dtype=float)
    row = 0
    matrix[row, :] = 1.0
    lower[row] = upper[row] = 100.0
    row += 1
    matrix[row, :] = arrays["activity"]
    lower[row] = 100.0 * (inc_activity - float(protocol["activity_drop_max"]))
    row += 1
    for j in range(len(genera)):
        matrix[row, :] = panel[:, j]
        lower[row] = 100.0 * (inc_panel[:, j].mean() - float(protocol["panel_drop_max"]))
        row += 1
    matrix[row, :] = arrays["breadth"]
    lower[row] = 100.0 * (inc_breadth - float(protocol["breadth_drop_max"]))
    row += 1
    matrix[row, :] = arrays["mdr"]
    lower[row] = 100.0 * (inc_mdr - float(protocol["mdr_drop_max"]))
    row += 1
    matrix[row, :] = (arrays["risk"] > float(protocol["risk_tail_threshold"])).astype(float)
    upper[row] = inc_tail
    row += 1
    # Each pair gets a NEW, empty row. Reusing a dense risk row here creates
    # sum(risk*x) + x_i + x_j <= 1, which falsely eliminates feasible sets.
    local_index = {int(value): position for position, value in enumerate(candidates)}
    for left, right in sorted(pair_cuts):
        if left not in local_index or right not in local_index:
            raise ValueError("Pair cut refers to a candidate outside the eligible pool")
        matrix[row, local_index[left]] = 1.0
        matrix[row, local_index[right]] = 1.0
        upper[row] = 1.0
        row += 1

    ceiling = float(profile)
    ub = np.ones(len(candidates), dtype=float)
    if ceiling < 1.0:
        ub[arrays["risk"] > ceiling] = 0.0
    for local, full in enumerate(candidates):
        if int(full) in forbidden:
            ub[local] = 0.0
    try:
        result = milp(
            c=arrays["risk"],
            integrality=np.ones(len(candidates)),
            bounds=Bounds(np.zeros(len(candidates)), ub),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={
                "time_limit": float(time_limit),
                "mip_rel_gap": float(mip_rel_gap),
                "presolve": True,
            },
        )
    except Exception as exc:  # retain a stable report for solver/version failures
        return None, str(exc), "failed_solver"
    if result.x is None:
        status = {1: "unknown_timeout", 2: "infeasible"}.get(result.status, "failed_solver")
        return result, str(result.message), status
    if result.status not in (0, 1):
        return result, str(result.message), "failed_solver"
    if (
        not np.isfinite(result.x).all()
        or (result.x < -1e-7).any()
        or (result.x > ub + 1e-7).any()
        or not np.allclose(result.x, np.round(result.x), rtol=0, atol=1e-7)
    ):
        return result, "Solver returned a fractional or invalid incumbent", "failed_validation"
    selected = candidates[np.flatnonzero(result.x > 0.5)]
    if len(selected) != 100:
        return result, "Solver returned an underfilled/overfilled incumbent", "failed_validation"
    status = "optimal" if result.status == 0 else "feasible_unproven_optimal"
    return result, status, str(result.message), selected


def solve_pool(
    frame,
    eligible: np.ndarray,
    genera: list[str],
    protocol: dict,
    profile: str = "1.0",
    *,
    reference: set[str] | None = None,
    time_limit: float = 300.0,
    mip_rel_gap: float = 1e-3,
    max_cut_rounds: int = 20,
) -> SolveResult:
    """Solve one risk-ceiling profile and add exact novelty/diversity cuts."""
    started = monotonic()
    validate_pool_frame(frame, genera=genera)
    if np.asarray(eligible).shape != (len(frame),) or np.asarray(eligible).dtype != bool:
        raise ValueError("Eligibility must be an aligned boolean vector")
    if (
        not np.isfinite(time_limit)
        or time_limit <= 0
        or max_cut_rounds < 0
        or not np.isfinite(mip_rel_gap)
        or mip_rel_gap < 0
        or not np.isfinite(float(profile))
        or not 0 <= float(profile) <= 1
    ):
        raise ValueError("Invalid solver budget, gap, or risk ceiling")
    for key in (
        "activity_drop_max",
        "panel_drop_max",
        "breadth_drop_max",
        "mdr_drop_max",
        "distance_drop_max",
        "risk_tail_threshold",
    ):
        if not np.isfinite(protocol[key]) or not 0 <= protocol[key] <= 1:
            raise ValueError(f"Invalid protocol field: {key}")
    if reference is None:
        raise ValueError("Pass the explicit novelty reference set (empty only for synthetic tests)")
    candidates = np.flatnonzero(np.asarray(eligible, dtype=bool)).astype(np.int64)
    forbidden: set[int] = set()
    pair_cuts: set[tuple[int, int]] = set()
    metadata = {
        "solver_version": SOLVER_VERSION,
        "profile": profile,
        "candidate_count": int(len(candidates)),
        "cut_rounds": 0,
        "pair_cuts": 0,
        "reference_cuts": 0,
        "time_limit_seconds": float(time_limit),
        "mip_rel_gap": float(mip_rel_gap),
        "budget_scope": "shared_across_cut_rounds",
        "validated": False,
        "solution_origin": "none",
        "improved_vs_incumbent": False,
    }
    if len(candidates) < 100:
        return SolveResult(
            "infeasible",
            "fewer than 100 eligible candidates",
            np.array([], dtype=int),
            {},
            metadata,
        )
    try:
        base = baseline_indices(frame)
    except ValueError as exc:
        return SolveResult("failed_validation", str(exc), np.array([], dtype=int), {}, metadata)
    if not np.all(np.asarray(eligible, dtype=bool)[base]):
        return SolveResult(
            "failed_validation",
            "incumbent contains an ineligible row",
            np.array([], dtype=int),
            {},
            metadata,
        )
    sequences = frame["sequence"].astype(str).tolist()
    last_status = "failed_solver"
    last_message = ""
    for round_index in range(max_cut_rounds + 1):
        remaining = time_limit - (monotonic() - started)
        if remaining <= 0:
            last_status, last_message = "unknown_timeout", "Shared solve budget exhausted"
            break
        outcome = _milp(
            frame,
            candidates,
            genera,
            protocol,
            profile,
            forbidden,
            pair_cuts,
            time_limit=remaining,
            mip_rel_gap=mip_rel_gap,
        )
        if len(outcome) == 3:
            _result, last_message, last_status = outcome
            break
        result, status, message, selected = outcome
        last_status, last_message = status, message
        pair_conflicts = _pairs(sequences, selected, TOP_SIMILARITY_THRESHOLD)
        novelty_conflicts = _reference_conflicts(
            sequences, selected, reference, TOP_SIMILARITY_THRESHOLD
        )
        if novelty_conflicts:
            forbidden.update(novelty_conflicts)
            metadata["reference_cuts"] = len(forbidden)
        if pair_conflicts:
            for left, right in pair_conflicts:
                pair_cuts.add(tuple(sorted((left, right))))
            metadata["pair_cuts"] = len(pair_cuts)
        metadata["cut_rounds"] = round_index + 1
        if not pair_conflicts and not novelty_conflicts:
            summary = summarize(frame, selected, genera, incumbent=base)
            margins = constraint_margins(frame, selected, genera, protocol, profile)
            summary["activity_floor"] = float(
                frame.iloc[base]["activity"].mean() - protocol["activity_drop_max"]
            )
            summary["distance_floor"] = (
                summarize(frame, base, genera)["pairwise_distance_mean"]
                - protocol["distance_drop_max"]
            )
            metadata.update(
                {
                    "objective_risk_mean": float(result.fun / 100.0),
                    "solver_status_code": int(result.status),
                    "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
                    "constraint_margins": margins,
                }
            )
            gap = getattr(result, "mip_gap", None)
            metadata["achieved_mip_gap"] = (
                float(gap) if gap is not None and np.isfinite(gap) else None
            )
            accepted = all(value >= -1e-7 for value in margins.values())
            if not accepted:
                last_status = "failed_validation"
                last_message = "Independent constraint validation failed"
                break
            metadata.update(
                {
                    "validated": True,
                    "solution_origin": "milp",
                    "elapsed_seconds": monotonic() - started,
                    "improved_vs_incumbent": summary["risk_delta_vs_incumbent"] < -1e-7,
                }
            )
            return SolveResult(last_status, last_message, selected.astype(int), summary, metadata)
    if last_status in FEASIBLE_STATUSES:
        last_status = "unknown_cut_limit"
        last_message = "cut rounds exhausted before a valid top-100 was found"
    metadata["termination_status"] = last_status
    metadata["termination_message"] = last_message
    fallback = _incumbent_fallback(
        frame, base, genera, protocol, profile, reference, last_message, metadata
    )
    if fallback is not None:
        fallback.metadata["elapsed_seconds"] = monotonic() - started
        if last_status == "infeasible":
            fallback.metadata["solver_contradiction"] = True
        return fallback
    metadata["elapsed_seconds"] = monotonic() - started
    return SolveResult(last_status, last_message, np.array([], dtype=int), {}, metadata)
