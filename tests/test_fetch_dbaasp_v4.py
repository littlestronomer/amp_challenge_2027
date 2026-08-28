"""DBAASP v4 fetcher: µM normalization + detail-card parsing (offline fixtures).

Live connectivity was verified 2026-08-28 (GET /peptides?limit/offset, GET
/peptides/{id}); these tests pin the parsing/normalization logic so API drift
shows up as a test failure, not as silently wrong labels.
"""

from __future__ import annotations

import pytest
from fetch_dbaasp_v4 import concentration_to_um, parse_detail


def test_concentration_to_um_units():
    assert concentration_to_um("14.5", "µM", mw=100.0) == (14.5, "14.5 µM")
    assert concentration_to_um("0.5", "µg/ml", mw=100.0)[0] == pytest.approx(5.0)
    assert concentration_to_um("500", "nM", mw=100.0) == (0.5, "0.5 µM")
    assert concentration_to_um("1.5", "mM", mw=100.0) == (1500.0, "1500 µM")
    assert concentration_to_um("12.5", "AU/ml", mw=100.0)[0] is None  # unusable unit
    assert concentration_to_um("n/a", "µM", mw=100.0)[0] is None


def test_censoring_normalized_to_ascii():
    val, text = concentration_to_um(">64", "µM", mw=100.0)
    assert (val, text) == (64.0, ">64 µM")
    val2, text2 = concentration_to_um("≤0.13", "µg/ml", mw=1300.0)
    assert val2 == pytest.approx(0.1)
    assert text2.startswith("<=0.1")  # unicode ≤ → ASCII <=


def test_parse_detail_real_card_shape():
    card = {
        "id": 1,
        "dbaaspId": "DBAASPR_1",
        "sequence": "",  # multimer parent: empty top-level sequence
        "complexity": {"name": "Multimer"},
        "monomers": [{"sequence": "ENREVPPGFTALIKTLRKCKII"}],
        "targetActivities": [
            {
                "targetSpecies": {"name": "Escherichia coli ATCC 25922"},
                "activityMeasureGroup": {"name": "MIC"},
                "concentration": "14.5",
                "unit": {"name": "µM"},
            },
            {
                "targetSpecies": {"name": "Staphylococcus aureus ATCC 25923"},
                "activityMeasureGroup": {"name": "MIC"},
                "concentration": "0.5",
                "unit": {"name": "µg/ml"},
            },
            {
                "targetSpecies": {"name": "E. faecium"},
                "activityMeasureGroup": {"name": "IC50"},  # non-MIC → skipped
                "concentration": "3",
                "unit": {"name": "µM"},
            },
            {
                "targetSpecies": {"name": "Vibrio splendidus"},  # non-convertible unit
                "activityMeasureGroup": {"name": "MIC"},
                "concentration": "2",
                "unit": {"name": "AU/ml"},
            },
        ],
        "hemoliticCytotoxicActivities": [
            {"activityType": "HC50", "concentration": "100", "unit": {"name": "µM"}}
        ],
    }
    activities, hemo, info = parse_detail(card)
    assert info["sequence"] == "ENREVPPGFTALIKTLRKCKII"  # single monomer promoted
    assert info["complexity"] == "Multimer"
    assert [a["target_organism"] for a in activities] == [
        "Escherichia coli ATCC 25922",
        "Staphylococcus aureus ATCC 25923",
    ]
    assert activities[0]["concentration"] == "14.5 µM"
    # 0.5 µg/ml → µM via this peptide's MW (~2.5 kDa): 0.5*1000/MW
    from convert_dramp_xlsx import peptide_mw

    expected = 0.5 * 1000.0 / peptide_mw("ENREVPPGFTALIKTLRKCKII")
    assert float(activities[1]["concentration"].split()[0]) == pytest.approx(expected, rel=1e-4)
    assert activities[1]["concentration"].endswith("µM")
    assert hemo and hemo[0]["kind"] == "HC50"
