"""DRAMP xlsx → mic_dramp.csv converter: clause extraction + unit conversion.

Fixtures use REAL Activity-text shapes observed in general_amps.xlsx (2026-08-28
probe), so the regexes are tested against the mess they must actually parse.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest
from convert_dramp_xlsx import (
    convert_to_um,
    extract_mic_entries,
    match_panel_genus,
    peptide_mw,
    read_xlsx_rows,
)

SEQ = "KLLKLLKKLLKL"  # 12 residues, valid alphabet


# ---------------------------------------------------------------------------
# Unit conversion table
# ---------------------------------------------------------------------------


def test_unit_conversion_table():
    mw = peptide_mw(SEQ)
    assert convert_to_um(100.0, "µM", mw)[0] == 100.0
    assert convert_to_um(1000.0, "nM", mw)[0] == pytest.approx(1.0)
    assert convert_to_um(1000.0, "pmol/ml", mw)[0] == pytest.approx(1.0)  # pmol/ml == nM
    assert convert_to_um(1.0, "nmol/ml", mw)[0] == pytest.approx(1000.0)  # nmol/ml == mM
    val, _ = convert_to_um(10.0, "µg/ml", mw)
    assert val == pytest.approx(10.0 * 1000.0 / mw)
    val2, _ = convert_to_um(1000.0, "ng/ml", mw)
    assert val2 == pytest.approx(1.0 * 1000.0 / mw)
    assert convert_to_um(5.0, "AU/µg", mw)[0] is None  # arbitrary units unusable


# ---------------------------------------------------------------------------
# Genus mapping (full names + abbreviations + capitalized epithets)
# ---------------------------------------------------------------------------


def test_genus_mapping_variants():
    assert match_panel_genus("Staphylococcus aureus ATCC6538P") == "Staphylococcus aureus"
    assert match_panel_genus("S. aureus") == "Staphylococcus aureus"
    assert match_panel_genus("E. coli K-12") == "Escherichia coli"
    assert match_panel_genus("Salmonella Enteritidis 4") == "Salmonella"
    assert match_panel_genus("E. faecium") == "Enterococcus faecium"
    assert match_panel_genus("Campylobacter jejuni") == ""  # not on the panel
    assert match_panel_genus("Staphylococcus simulans") == ""  # wrong species


# ---------------------------------------------------------------------------
# MIC clause extraction on real text shapes
# ---------------------------------------------------------------------------


def test_extract_real_activity_shapes():
    # Real general_amps.xlsx strings (2026-08-28 probe)
    text1 = (
        "Gram-positive bacteria: Staphylococcus aureus ATCC6538P (MIC≤0.13 μg/ml), "
        "Staphylococcus epidermidis (MIC=2 µg/ml)"
    )
    e1 = extract_mic_entries(text1, SEQ)
    # clause 2 (epidermidis) is not a panel species → only clause 1 survives,
    # and the delimiter-cut must keep clause 2 from inheriting aureus
    assert [e["organism"] for e in e1] == ["Staphylococcus aureus"]
    # first clause: ≤0.13 µg/ml → tiny µM value
    assert e1[0]["value_um"] == pytest.approx(0.13 * 1000.0 / peptide_mw(SEQ), rel=1e-3)
    assert "≤" in e1[0]["raw_clause"]

    text2 = "Human pathogens: L100 Staphylococcus aureus ATCC6538P (MIC≤0.13 μg/ml)"
    assert extract_mic_entries(text2, SEQ)[0]["organism"] == "Staphylococcus aureus"

    text3 = "Gram-negative bacteria: Salmonella Enteritidis 1 (MIC=0.19 ug/ml)"
    e3 = extract_mic_entries(text3, SEQ)
    assert e3[0]["organism"] == "Salmonella"

    text4 = "Gram-positive bacteria: Enterococcus faecalis JCM 5803 (MIC=50.8 nM)"
    e4 = extract_mic_entries(text4, SEQ)
    assert e4[0]["value_um"] == pytest.approx(0.0508)

    text5 = "Clavibacter michiganensis subspecies sepedonicus strain 2136 (MIC=30 pmol/ml)."
    assert extract_mic_entries(text5, SEQ) == []  # no panel genus in window


def test_range_takes_upper_bound_and_abbreviation():
    text = "Gram-negative bacteria: E. coli (MIC=0.025-6.4 µg/ml)"
    e = extract_mic_entries(text, SEQ)
    assert len(e) == 1
    assert e[0]["organism"] == "Escherichia coli"
    assert e[0]["value_um"] == pytest.approx(6.4 * 1000.0 / peptide_mw(SEQ), rel=1e-3)
    assert "range-upper" in e[0]["note"]


def test_multiple_clauses_multiple_genera():
    text = (
        "Gram-positive bacteria: Enterococcus faecalis JCM 5803 (MIC=813 nM), "
        "Enterococcus faecalis 2 (MIC=50 µM)"
    )
    e = extract_mic_entries(text, SEQ)
    assert [x["value_um"] for x in e] == [pytest.approx(0.813), pytest.approx(50.0)]


# ---------------------------------------------------------------------------
# End-to-end on a minimal fabricated xlsx (stdlib zip build)
# ---------------------------------------------------------------------------


HEADERS = [
    "DRAMP_ID",
    "Sequence",
    "Activity",
    "Linear/Cyclic/Branched",
    "N-terminal_Modification",
    "C-terminal_Modification",
]


def _build_mini_xlsx(path: Path, data_rows: list[list[str]]) -> None:
    shared = HEADERS + [cell for row in data_rows for cell in row]
    shared_xml = (
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + "".join(f"<si><t>{s}</t></si>" for s in shared)
        + "</sst>"
    )
    col_letter = "ABCDEFGHIJKLMNOP"

    def row_xml(cells: list[str], row_idx: int) -> str:
        parts = []
        for j, cell in enumerate(cells):
            idx = shared.index(cell)
            parts.append(f'<c r="{col_letter[j]}{row_idx}" t="s"><v>{idx}</v></c>')
        return f'<row r="{row_idx}">{"".join(parts)}</row>'

    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + row_xml(HEADERS, 1)
        + "".join(row_xml(r, i + 2) for i, r in enumerate(data_rows))
        + "</worksheet>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", shared_xml)
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def test_read_xlsx_rows_and_conversion_end_to_end(tmp_path):
    from convert_dramp_xlsx import main as convert_main

    xlsx = tmp_path / "general_amps.xlsx"
    _build_mini_xlsx(
        xlsx,
        [
            [
                "D1",
                SEQ,
                "Staphylococcus aureus ATCC6538P (MIC≤0.13 μg/ml)",
                "linear",
                "none",
                "none",
            ],
            [
                "D2",
                SEQ,
                "E. coli (MIC=0.025-6.4 µg/ml)",
                "linear",
                "Acetyl",
                "none",
            ],  # modified N-term → skipped
            ["D3", SEQ, "cyclic peptide, no MIC text", "cyclic", "none", "none"],  # skipped
        ],
    )
    rows = read_xlsx_rows(xlsx)
    assert rows[0]["Sequence"] == SEQ and rows[0]["Activity"].startswith("Staphylococcus")

    out_csv = tmp_path / "mic_dramp.csv"
    convert_main(["--xlsx", str(xlsx), "--out", str(out_csv)])

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    # Only the clean linear/unmodified record contributes a row.
    assert len(rows) == 1
    assert rows[0]["sequence"] == SEQ
    assert rows[0]["target_organism"] == "Staphylococcus aureus"
    assert rows[0]["unit"] == "uM"
    assert rows[0]["source_db"] == "DRAMP"
    assert rows[0]["assay"].startswith("DRAMP:MIC")
    assert float(rows[0]["mic_value_um"]) == pytest.approx(
        0.13 * 1000.0 / peptide_mw(SEQ), rel=1e-3
    )
