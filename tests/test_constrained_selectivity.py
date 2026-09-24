from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amp_challenge_2027.config import PANEL_GENERA
from amp_challenge_2027.selectivity_research import solver


@pytest.fixture
def cut_pool():
    rng = np.random.default_rng(751)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    sequences = ["".join(rng.choice(alphabet, size=20)) for _ in range(101)]
    # Two cheapest peptides conflict with each other, but either is compatible
    # with the other 99. Exactly one must be excluded; the incumbent is valid.
    sequences[100] = sequences[0][:-1] + ("A" if sequences[0][-1] != "A" else "C")
    frame = pd.DataFrame(
        {
            "sequence": sequences,
            "library_index": range(101),
            "activity": 0.8,
            "hemolysis_risk": 0.3,
            "conformity": 0.5,
            "precision": 0.9,
            "incumbent_member": [True] * 100 + [False],
        }
    )
    frame.loc[0, "hemolysis_risk"] = 0.2
    frame.loc[100, "hemolysis_risk"] = 0.1
    for genus in PANEL_GENERA:
        frame[f"panel:{genus}"] = 0.8
    protocol = {
        "activity_drop_max": 0.03,
        "panel_drop_max": 0.03,
        "breadth_drop_max": 0.02,
        "mdr_drop_max": 0.02,
        "distance_drop_max": 0.02,
        "risk_tail_threshold": 0.8,
    }
    return frame, protocol


def test_actual_conflict_cut_preserves_feasibility_and_finds_improvement(cut_pool):
    frame, protocol = cut_pool
    result = solver.solve_pool(
        frame, np.ones(101, bool), PANEL_GENERA, protocol, "0.5", reference=set(), time_limit=10
    )
    assert result.status == "optimal"
    assert result.metadata["solution_origin"] == "milp"
    assert result.metadata["pair_cuts"] == 1
    assert result.metadata["cut_rounds"] >= 2
    assert set(result.indices) == set(range(1, 101))
    assert result.summary["risk_delta_vs_incumbent"] == pytest.approx(-0.001)
    assert min(result.metadata["constraint_margins"].values()) >= -1e-7


def test_direct_pair_cut_does_not_include_dense_risk_coefficients(cut_pool):
    frame, protocol = cut_pool
    # Risk sum is about 30. The defective row falsely required it to be <=1.
    result = solver._milp(
        frame,
        np.arange(101),
        PANEL_GENERA,
        protocol,
        "1.0",
        set(),
        {(0, 100)},
        time_limit=10,
        mip_rel_gap=0.001,
    )
    assert len(result) == 4
    assert result[1] == "optimal"
    assert set(result[3]) == set(range(1, 101))


def test_cut_limit_is_unknown_not_infeasible(cut_pool):
    frame, protocol = cut_pool
    frame.loc[:99, "hemolysis_risk"] = 0.9
    frame.loc[100, "hemolysis_risk"] = 0.1
    frame.loc[99, "hemolysis_risk"] = 0.95
    result = solver.solve_pool(
        frame,
        np.ones(101, bool),
        PANEL_GENERA,
        protocol,
        "0.92",
        reference=set(),
        time_limit=10,
        max_cut_rounds=0,
    )
    assert result.status == "unknown_cut_limit"
    assert not result.indices.size


def test_true_infeasibility_has_no_fallback(cut_pool):
    frame, protocol = cut_pool
    result = solver.solve_pool(
        frame, np.ones(101, bool), PANEL_GENERA, protocol, "0.15", reference=set(), time_limit=10
    )
    assert result.status == "infeasible"
    assert not result.indices.size


def test_reference_cut_and_boundary(cut_pool):
    frame, protocol = cut_pool
    result = solver.solve_pool(
        frame,
        np.ones(101, bool),
        PANEL_GENERA,
        protocol,
        "1.0",
        reference={frame.iloc[100].sequence},
        time_limit=10,
    )
    # Both similar candidates conflict with reference: only 99 remain.
    assert result.status == "infeasible"
    assert result.metadata["reference_cuts"] == 2
    assert solver._reference_conflicts(["AAAAAAAAAC"], np.array([0]), {"AAAAAAAACC"}, 0.8) == [0]
    assert solver._reference_conflicts(["AAAAAAAAAC"], np.array([0]), {"AAAAAAACCC"}, 0.8) == []


def test_timeout_is_reported_separately_from_incumbent_fallback(cut_pool, monkeypatch):
    frame, protocol = cut_pool
    monkeypatch.setattr(solver, "_milp", lambda *a, **k: (None, "Time limit", "unknown_timeout"))
    result = solver.solve_pool(
        frame, np.ones(101, bool), PANEL_GENERA, protocol, reference=set(), time_limit=10
    )
    assert result.status == "feasible_unproven_optimal"
    assert result.metadata["solution_origin"] == "incumbent_fallback"
    assert result.metadata["termination_status"] == "unknown_timeout"
    assert not result.metadata["improved_vs_incumbent"]
    np.testing.assert_array_equal(result.indices, np.arange(100))


def test_shared_budget_decreases_across_rounds(cut_pool, monkeypatch):
    frame, protocol = cut_pool
    actual = solver._milp
    budgets = []

    def wrapped(*args, **kwargs):
        budgets.append(kwargs["time_limit"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(solver, "_milp", wrapped)
    solver.solve_pool(
        frame, np.ones(101, bool), PANEL_GENERA, protocol, reference=set(), time_limit=10
    )
    assert len(budgets) >= 2
    assert 0 < budgets[-1] < budgets[0] <= 10


def test_distance_tolerance_comes_from_protocol(cut_pool):
    frame, protocol = cut_pool
    protocol["distance_drop_max"] = 0.123
    margins = solver.constraint_margins(frame, np.arange(100), PANEL_GENERA, protocol, "1.0")
    assert margins["distance"] == pytest.approx(0.123)


def test_fractional_solver_output_is_not_rounded_into_a_selection(cut_pool, monkeypatch):
    from types import SimpleNamespace

    import scipy.optimize

    frame, protocol = cut_pool
    monkeypatch.setattr(
        scipy.optimize,
        "milp",
        lambda **kw: SimpleNamespace(x=np.full(101, 0.5), status=1, message="timeout"),
    )
    result = solver._milp(
        frame,
        np.arange(101),
        PANEL_GENERA,
        protocol,
        "1.0",
        set(),
        set(),
        time_limit=10,
        mip_rel_gap=0.001,
    )
    assert result[2] == "failed_validation"


def test_cli_solve_report_and_nested_integrity(cut_pool, tmp_path):
    import json

    import selectivity_report
    import selectivity_solve
    from experiment_utils import mark_files, sha256, write_json
    from selectivity_artifacts import checked_stage

    frame, protocol = cut_pool
    protocol.update(
        kind="constrained_selectivity_v1", top_size=100, seeds=[42], risk_ceilings=[0.5, 1.0]
    )
    config = tmp_path / "protocol.json"
    write_json(config, protocol)
    reference = tmp_path / "ref.fasta"
    reference.write_text(">reference\nYYYYYYYYYYYYYYYYYYYY\n")
    pools = tmp_path / "pools"
    (pools / "seed42").mkdir(parents=True)
    frame.to_csv(pools / "seed42/pool.csv", index=False)
    write_json(
        pools / "run.json", {"kind": "selectivity_pool_v1", "reference_sha256": sha256(reference)}
    )
    write_json(pools / "cells.json", {"42": {"pool_sha256": sha256(pools / "seed42/pool.csv")}})
    mark_files(pools, "complete.json", ["run.json", "cells.json"])
    eligibility = tmp_path / "eligibility"
    (eligibility / "seed42").mkdir(parents=True)
    pd.DataFrame({"library_index": range(101), "eligible": True}).to_csv(
        eligibility / "seed42/eligibility.csv", index=False
    )
    write_json(
        eligibility / "run.json",
        {
            "kind": "selectivity_eligibility_v1",
            "reference_sha256": sha256(reference),
            "pools_run_sha256": sha256(pools / "run.json"),
        },
    )
    write_json(
        eligibility / "cells.json",
        {"42": {"sha256": sha256(eligibility / "seed42/eligibility.csv")}},
    )
    mark_files(eligibility, "complete.json", ["run.json", "cells.json"])
    solutions, report = tmp_path / "solutions", tmp_path / "report"
    assert (
        selectivity_solve.main(
            [
                "--pools",
                str(pools),
                "--eligibility",
                str(eligibility),
                "--protocol",
                str(config),
                "--reference",
                str(reference),
                "--out",
                str(solutions),
                "--seeds",
                "42",
                "--risk-ceilings",
                "0.5",
                "--time-limit",
                "10",
            ]
        )
        == 0
    )
    assert (
        selectivity_report.main(
            [
                "--pools",
                str(pools),
                "--solutions",
                str(solutions),
                "--protocol",
                str(config),
                "--out",
                str(report),
            ]
        )
        == 0
    )
    assert "milp" in (report / "REPORT.md").read_text()
    assert "-0.0010" in (report / "REPORT.md").read_text()
    cell = solutions / "seed42/ceiling-0.5"
    margins = json.loads((cell / "constraint_margins.json").read_text())
    assert "activity" in margins and "risk_mean" not in margins
    (cell / "top.fasta").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        checked_stage(solutions, "selectivity_solution_v1")


def test_protocol_is_frozen_and_valid():
    from amp_challenge_2027.selectivity_research.contracts import load_protocol

    protocol = load_protocol(
        __import__("pathlib").Path("experiments/constrained_selectivity_v1.json")
    )
    assert protocol["top_size"] == 100
    assert protocol["risk_ceilings"] == [0.25, 0.5, 0.75, 1.0]


def test_solver_returns_a_constrained_top100():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    import numpy as np

    from amp_challenge_2027.config import PANEL_GENERA
    from amp_challenge_2027.selectivity_research.solver import solve_pool

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rng = np.random.default_rng(123)
    rows = []
    for i in range(110):
        sequence = "".join(rng.choice(list(alphabet), size=12))
        row = {
            "sequence": sequence,
            "library_index": i,
            "activity": 0.8,
            "hemolysis_risk": 0.1 if i >= 100 else 0.3,
            "conformity": 0.5,
            "precision": 0.9,
            "incumbent_member": i < 100,
            "incumbent_rank": i + 1 if i < 100 else "",
        }
        row.update({f"panel:{genus}": 0.8 for genus in PANEL_GENERA})
        rows.append(row)
    frame = pd.DataFrame(rows)
    protocol = {
        "activity_drop_max": 0.03,
        "panel_drop_max": 0.03,
        "breadth_drop_max": 0.02,
        "mdr_drop_max": 0.02,
        "distance_drop_max": 0.02,
        "risk_tail_threshold": 0.8,
    }
    result = solve_pool(
        frame,
        np.ones(len(frame), dtype=bool),
        PANEL_GENERA,
        protocol,
        "1.0",
        reference=set(),
        time_limit=10,
    )
    assert result.status in {"optimal", "feasible_unproven_optimal"}
    assert len(result.indices) == 100
    assert result.summary["risk_mean"] <= 0.3
