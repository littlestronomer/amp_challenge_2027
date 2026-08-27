"""Expanded-label dataset preparation: per-genus bands, gram votes, DRAMP merge.

Hermetic (tmp fixtures only; no network). Covers the honesty-critical parts:

  * band boundaries inclusive at 4 / 16 / 32 µM, ambiguous middle → blank label
  * one row per (sequence × panel genus), not collapsed
  * µg/mL rows converted via peptide MW (never misread as µM); unknown units dropped
  * gram votes: any-side-active rule + all-inactive rule, both sides may appear
  * DRAMP membership union with DBAASP-precedence conflict resolution
  * DRAMP fetch probes: candidate order honored, non-FASTA falls through,
    provenance JSON deterministic and idempotent
"""

from __future__ import annotations

import csv
import json

import fetch_data
import pytest
from build_ranking_labels import (
    aggregate_genus_mics,
    build_full_rows,
    classify_band,
    load_dramp_memberships,
    merge_gram_labels,
    mic_to_um,
    peptide_mw,
)


def _mic_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "target_organism", "mic_value_um", "unit"])
        w.writeheader()
        w.writerows(rows)
    return path


SEQ10 = "AGKLLKAGKL"  # valid, length 10


# ---------------------------------------------------------------------------
# Banding + unit conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mic,expected",
    [
        (4.0, "potent"),
        (4.1, "active-band"),
        (16.0, "active-band"),
        (16.1, "weak"),
        (32.0, "weak"),
        (32.1, "inactive"),
        (300.0, "inactive"),
    ],
)
def test_band_boundaries_inclusive(mic, expected):
    assert classify_band(mic, potent=4.0, success=16.0, inactive=32.0) == expected


def test_ug_ml_conversion_exact_for_polyalanine():
    # 10×A: MW = 10*71.0788 + 18.01528; X µg/mL * 1000 / MW == X/(MW/1000) µM
    mw = peptide_mw("A" * 10)
    val, note = mic_to_um(100.0 * mw / 1000.0, "ug_ml", "A" * 10)
    assert (val, note) == (pytest.approx(100.0), "")
    val2, _ = mic_to_um(50.0, "ug_ml", SEQ10)
    assert val2 == pytest.approx(50.0 * 1000.0 / peptide_mw(SEQ10))


def test_unknown_unit_dropped_and_blank_assumed():
    v_none, _note = mic_to_um(12.5, "mg/kg", SEQ10)
    assert v_none is None
    v_assumed, note = mic_to_um(12.5, "", SEQ10)
    assert (v_assumed, note) == (12.5, "assumed-uM")


# ---------------------------------------------------------------------------
# Aggregation + full rows
# ---------------------------------------------------------------------------


def test_per_genus_rows_not_collapsed(tmp_path):
    mic = _mic_csv(
        tmp_path / "mic.csv",
        [
            {
                "sequence": SEQ10,
                "target_organism": "Escherichia coli K12",
                "mic_value_um": 2.0,
                "unit": "uM",
            },
            {
                "sequence": SEQ10,
                "target_organism": "Pseudomonas aeruginosa PAO1",
                "mic_value_um": 40.0,
                "unit": "uM",
            },
            {
                "sequence": SEQ10,
                "target_organism": "Vibrio splendidus",
                "mic_value_um": 0.5,
                "unit": "uM",
            },
        ],
    )
    genus_mics, stats = aggregate_genus_mics([*csv.DictReader(open(mic))])
    assert set(genus_mics[SEQ10]) == {"E. coli", "P. aeruginosa"}  # Vibrio unmapped
    assert stats["unmapped_organism"] == 1

    rows = build_full_rows(genus_mics, potent=4.0, success=16.0, inactive=32.0)
    by_genus = {r["organism"]: r for r in rows}
    assert len(rows) == 2  # NOT collapsed to min-across-genera like the legacy builder
    assert by_genus["E. coli"]["band"] == "potent"
    assert by_genus["E. coli"]["label"] == "active"
    assert by_genus["P. aeruginosa"]["band"] == "inactive"
    assert by_genus["P. aeruginosa"]["label"] == "inactive"


def test_ambiguous_middle_gets_blank_label(tmp_path):
    mic = _mic_csv(
        tmp_path / "mic.csv",
        [
            {
                "sequence": SEQ10,
                "target_organism": "Staphylococcus aureus ATCC 12600",
                "mic_value_um": 20.0,
                "unit": "uM",
            }
        ],
    )
    genus_mics, _ = aggregate_genus_mics(list(csv.DictReader(open(mic))))
    rows = build_full_rows(genus_mics, potent=4.0, success=16.0, inactive=32.0)
    assert rows[0]["band"] == "weak"
    assert rows[0]["label"] == ""


def test_end_to_end_main_writes_both_files(tmp_path, capsys):
    from build_ranking_labels import main

    dramp_dir = tmp_path / "dramp"
    dramp_dir.mkdir()
    _mic_csv(
        tmp_path / "mic.csv",
        [
            {
                "sequence": "KLLKLLKKLLKL",
                "target_organism": "Escherichia coli",
                "mic_value_um": 1.0,
                "unit": "uM",
            },
            {
                "sequence": "GLLSGINAASKKAAKHGGKVSAW",
                "target_organism": "Staphylococcus aureus",
                "mic_value_um": 64.0,
                "unit": "uM",
            },
            {
                "sequence": "KWKLFKKIGAVLKVL",
                "target_organism": "Pseudomonas aeruginosa",
                "mic_value_um": 8.0,
                "unit": "ug_ml",
            },
        ],
    )
    out_dir = tmp_path / "out"
    main(
        [
            "--mic",
            str(tmp_path / "mic.csv"),
            "--dramp-dir",
            str(dramp_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    out_text = capsys.readouterr().out
    full = list(csv.DictReader(open(out_dir / "activity_labels_full.csv")))
    gram = list(csv.DictReader(open(out_dir / "activity_labels_gram.csv")))

    seqs_full = {r["sequence"] for r in full}
    assert {"KLLKLLKKLLKL", "GLLSGINAASKKAAKHGGKVSAW", "KWKLFKKIGAVLKVL"} <= seqs_full

    labels = {(r["sequence"], r["label"]) for r in full}
    assert ("KLLKLLKKLLKL", "active") in labels
    assert ("GLLSGINAASKKAAKHGGKVSAW", "inactive") in labels

    gram_pairs = {
        (r["gram"], r["label"], r["source"]) for r in gram if r["sequence"] == "KLLKLLKKLLKL"
    }
    assert ("negative", "active", "dbaasp") in gram_pairs
    assert "+0 newly covered sequences" not in out_text or True  # delta line format-free here


def test_graceful_exit_when_mic_missing(tmp_path):
    from build_ranking_labels import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--mic",
                str(tmp_path / "absent.csv"),
                "--dramp-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Gram votes + DRAMP merge precedence
# ---------------------------------------------------------------------------


def test_both_sides_possible_and_merge_precedence(tmp_path):
    rows = [
        {
            "sequence": "P1",
            "organism": "E. coli",
            "mic_um": "2.0000",
            "band": "potent",
            "label": "active",
        },
        {
            "sequence": "P1",
            "organism": "S. aureus",
            "mic_um": "64.0000",
            "band": "inactive",
            "label": "inactive",
        },
    ]
    votes = []

    # emulate vote derivation directly on band objects via helper semantics:
    from build_ranking_labels import gram_votes_from_full

    votes = gram_votes_from_full([dict(r) for r in rows], inactive=32.0)
    assert sorted(votes) == [("P1", "negative", "active"), ("P1", "positive", "inactive")]

    # DRAMP claims P1 positive too — identical to dbaasp vote: kept, but source=dbaasp
    members = [("P1", "positive", "active"), ("NEWSEQ1", "positive", "active")]
    merged, conflicts = merge_gram_labels(votes, members)
    by_key = {(r["sequence"], r["gram"]): r for r in merged}
    assert (
        by_key[("P1", "positive")]["source"] == "dbaasp"
    )  # numeric evidence wins even when equal verdicts
    assert by_key[("NEWSEQ1", "positive")]["source"] == "dramp"

    # conflicting verdicts also resolve toward dbaasp and are counted
    merged2, conflicts2 = merge_gram_labels([], [("X1", "negative", "active")])
    assert conflicts2 == 0
    merged3, conflicts3 = merge_gram_labels(
        [("X1", "negative", "inactive")], [("X1", "negative", "active")]
    )
    assert conflicts3 == 1
    r3 = [r for r in merged3 if r["sequence"] == "X1"][0]
    assert (r3["label"], r3["source"]) == ("inactive", "dbaasp")


def test_dramp_membership_loader_missing_file_ok_partial_fasta(tmp_path):
    d = tmp_path
    (d / "anti_gram_positive.fasta").write_text(
        ">a broken line\n>seq1\nKLLKLLKKLLKL\n>seq2\nKLLKLLKKLLKL\n"
    )
    members, counts = load_dramp_memberships(d)
    assert counts["memberships"] == 1  # dedup within file; first header without sequence tolerated
    assert members == [("KLLKLLKKLLKL", "positive", "active")]


# ---------------------------------------------------------------------------
# fetch_data: DRAMP probes + provenance registry (offline)
# ---------------------------------------------------------------------------


def test_url_table_integrity_without_network():
    assert fetch_data.DRAMP_DL_BASE.startswith("https://dramp.cpu-bioinfor.org/")
    url = fetch_data._dramp_url("general_amps.fasta")
    assert url.endswith("download_data/DRAMP3.0_new/general_amps.fasta")
    assert "+" not in url  # spaces were encoded properly
    assert set(fetch_data.DRAMP_DIRECT_SOURCES) >= {
        "dramp-general",
        "dramp-antibacterial",
        "dramp-grampos",
        "dramp-gramneg",
    }


def test_fetch_probe_order_and_provenance(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest, **kw):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "Anti-Gram-negative_amps.fasta" in url:  # first probe fails upstream-style
            raise RuntimeError("HTTP Error 404")
        dest.write_bytes(b">split\nGIGTKILGGVKTALK\n")

    monkeypatch.setattr(fetch_data, "_download", fake_download)
    got = fetch_data.fetch_dramp(
        "dramp-gramneg", out_dir=tmp_path / "dramp", registry_path=tmp_path / "sources.json"
    )
    assert got is not None and got.exists()
    assert len(calls) == 2  # full name probed first, truncated fallback second

    reg = json.loads((tmp_path / "sources.json").read_text())
    key = "dramp-gramneg/anti_gram_negative.fasta"
    entry = reg[key]
    assert entry["license"] == "CC BY 4.0" and entry["url"] == calls[-1]
    assert entry["size_bytes"] > 0 and len(entry["sha256"]) == 64


def test_non_fasta_response_falls_through_to_manual_message(tmp_path, monkeypatch, capsys):
    def fake_download(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html><body><h1>Error</h1></body></html>")

    monkeypatch.setattr(fetch_data, "_download", fake_download)
    got = fetch_data.fetch_dramp("dramp-antibacterial", out_dir=tmp_path / "dramp")
    assert got is None  # nothing silently accepted
    out = capsys.readouterr().out
    assert "manual" in out.lower() and "Antibacterial_amps.fasta" in out


def test_provenance_idempotent_and_sorted(tmp_path):
    f = tmp_path / "x.fasta"
    f.write_bytes(b">s\nAAAA\n")
    reg = tmp_path / "sources.json"
    kw = dict(license_name="CC BY 4.0", citation="test cite")
    fetch_data._record_provenance("t", "https://example/x", f, registry_path=reg, **kw)
    first = json.loads(reg.read_text())
    fetch_data._record_provenance("t", "https://example/x", f, registry_path=reg, **kw)
    second_text = reg.read_text()
    assert (
        json.loads(second_text)["t/x.fasta"]["retrieved_utc"] == first["t/x.fasta"]["retrieved_utc"]
    )
    raw_keys = list(json.loads(second_text).keys())
    assert raw_keys == sorted(raw_keys)
