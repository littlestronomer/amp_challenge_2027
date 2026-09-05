"""Sweep metric collection: both seqme frame layouts flatten correctly.

The 2026-08-29 sweep run exposed the wide layout (index=['library'], metrics
as (metric, statistic) COLUMNS) — the tall-only collector recorded a single
metric named 'library'. These tests pin both layouts with duck-typed frames
(no pandas dependency).
"""

from __future__ import annotations

import pytest
import sweep_selection
from experiment_utils import metrics_for_json


class _Cell:
    """df.loc[key, col] → scalar; df.loc[key] → row view with .iloc."""

    def __init__(self, grid):
        self.grid = grid  # {(key, col): value}

    def __getitem__(self, key):
        return self.grid[key]


class _LocRows:
    def __init__(self, rows):
        self.rows = rows  # {key: {"cols": [...], "vals": [...]}}

    def __getitem__(self, key):
        row = self.rows[key]

        class _Row:
            def __init__(self, cols, vals):
                self.cols = cols
                self.iloc = _ILoc(vals)

        return _Row(row["cols"], row["vals"])


class _ILoc:
    def __init__(self, vals):
        self.vals = vals

    def __getitem__(self, i):
        return self.vals[i]


class _Frame:
    def __init__(self, index, columns, loc):
        self.index = index
        self.columns = columns
        self.loc = loc


def test_wide_layout_current_seqme_behavior():
    cols = [
        ("Uniqueness", "value"),
        ("Diversity", "value"),
        ("Diversity", "deviation"),
        ("FBD", "value"),
        ("ConformityScore", "value"),
        ("ConformityScore", "deviation"),
    ]
    grid = {
        ("library", ("Uniqueness", "value")): 1.0,
        ("library", ("Diversity", "value")): 0.85,
        ("library", ("Diversity", "deviation")): 0.002,
        ("library", ("FBD", "value")): 0.51,
        ("library", ("ConformityScore", "value")): 0.59,
        ("library", ("ConformityScore", "deviation")): 0.0011,
    }
    df = _Frame(["library"], cols, _Cell(grid))

    row: dict = {}
    names = sweep_selection._collect_metrics(row, df, dataset_name="library")

    assert names == ["Uniqueness", "Diversity", "FBD", "ConformityScore"]
    assert row["Uniqueness"] == 1.0
    assert row["Diversity"] == 0.85
    assert row["Diversity.deviation"] == 0.002
    assert row["FBD"] == 0.51
    assert row["ConformityScore"] == 0.59
    assert "library" not in row  # the old bug: dataset name became a "metric"


def test_tall_layout_legacy_behavior():
    cols = [("group", "value"), ("group", "deviation")]
    df = _Frame(
        ["Uniqueness", "FBD"],
        cols,
        _LocRows(
            {
                "Uniqueness": {"cols": cols, "vals": [1.0, float("nan")]},
                "FBD": {"cols": cols, "vals": [0.51, 0.02]},
            }
        ),
    )

    row: dict = {}
    names = sweep_selection._collect_metrics(row, df)

    assert names == ["Uniqueness", "FBD"]
    assert row["Uniqueness"] == 1.0
    assert row["FBD"] == 0.51
    assert row["FBD.deviation"] == 0.02


def test_metrics_json_allows_missing_deviation_but_not_missing_metrics():
    cols = [("FBD", "value"), ("FBD", "deviation")]
    df = _Frame(["library"], cols, _Cell({
        ("library", cols[0]): 0.25, ("library", cols[1]): float("nan"),
    }))
    assert metrics_for_json(df, expected_names=["FBD"]) == {"FBD": 0.25, "FBD.deviation": None}
    with pytest.raises(ValueError, match="Incomplete metric results"):
        metrics_for_json(df, expected_names=["FBD", "MMD"])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_metrics_json_rejects_nonfinite_primary_measurement(value):
    cols = [("FBD", "value")]
    df = _Frame(["library"], cols, _Cell({("library", cols[0]): value}))
    with pytest.raises(ValueError, match="non-finite primary"):
        metrics_for_json(df, expected_names=["FBD"])
