from __future__ import annotations

import pytest


def test_protocol_is_frozen_and_valid():
    from amp_challenge_2027.selectivity_research.contracts import load_protocol

    protocol = load_protocol(__import__("pathlib").Path("experiments/constrained_selectivity_v1.json"))
    assert protocol["top_size"] == 100
    assert protocol["risk_ceilings"] == [0.25, 0.5, 0.75, 1.0]


def test_solver_returns_a_constrained_top100():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    import numpy as np

    from amp_challenge_2027.config import PANEL_GENERA
    from amp_challenge_2027.selectivity_research.solver import solve_pool

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for i in range(110):
        # Encode the index in base-20 so the deterministic fixtures are not a
        # family of near-duplicate cyclic shifts.
        value = i
        digits = []
        for _ in range(12):
            digits.append(alphabet[value % len(alphabet)])
            value = value // len(alphabet) + 17
        sequence = "".join(digits)
        row = {"sequence": sequence, "library_index": i, "activity": .8,
               "hemolysis_risk": .1 if i >= 100 else .3, "conformity": .5,
               "precision": .9, "incumbent_member": i < 100,
               "incumbent_rank": i + 1 if i < 100 else ""}
        row.update({f"panel:{genus}": .8 for genus in PANEL_GENERA})
        rows.append(row)
    frame = pd.DataFrame(rows)
    protocol = {"activity_drop_max": .03, "panel_drop_max": .03,
                "breadth_drop_max": .02, "mdr_drop_max": .02,
                "distance_drop_max": .02, "risk_tail_threshold": .8}
    result = solve_pool(frame, np.ones(len(frame), dtype=bool), PANEL_GENERA, protocol,
                        "1.0", reference=set(), time_limit=10)
    assert result.status in {"optimal", "feasible_unproven_optimal"}
    assert len(result.indices) == 100
    assert result.summary["risk_mean"] <= .3
