"""Tests for the expanded-corpus builder (scripts/build_expanded_generative.py).

Numpy-only; runs in the minimal dev environment.
"""

from __future__ import annotations

from pathlib import Path

import build_expanded_generative as exp


def _write_fasta(path: Path, seqs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs, start=1):
            f.write(f">s{i}\n{s}\n")


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence"])
        for s, _src in rows:
            w.writerow([s])


SEQS = [
    "KLLAKLLAKLLA",
    "GVLKLAGVLKLA",
    "RRWFKLLAKLAA",
    "KLAGVKLAGVKK",
    "QMYVPKFAVGSTKR",  # synthetic — must NOT match anything in antibacterial.fasta
    "IGPSCTTKM",
]


def test_curate_and_merge_dedup_priority_and_reference(tmp_path: Path):
    pools = [
        ("marlys", [(SEQS[0], "marlys"), ("BAD!", "marlys"), (SEQS[1], "marlys")]),
        ("dramp", [(SEQS[1], "dramp"), (SEQS[2], "dramp")]),  # SEQS[1] dup across sources
        ("dbaasp", [(SEQS[3], "dbaasp")]),
    ]
    merged, stats = exp.curate_and_merge(pools, exclude_reference={SEQS[3]})

    assert [s for s, _ in merged] == [SEQS[0], SEQS[1], SEQS[2]]
    assert dict((s, src) for s, src in merged)[SEQS[1]] == "marlys", (
        "earlier-priority source must win cross-source duplicates"
    )
    assert stats["invalid"] == 1  # "BAD!"
    assert stats["duplicate"] == 1
    assert stats["reference_overlap"] == 1
    assert stats["kept"] == 3


def test_end_to_end_build(tmp_path: Path):
    generative = tmp_path / "processed" / "generative.csv"
    _write_csv(generative, [(s, "marlys") for s in SEQS[:2]])

    _write_fasta(tmp_path / "raw" / "dramp" / "general.fasta", [SEQS[1], SEQS[2]])
    _write_fasta(tmp_path / "raw" / "apd" / "apd.fasta", [SEQS[3]])

    # mic.csv feeding the DBAASP sequence extractor.
    import csv

    mic = tmp_path / "processed" / "mic.csv"
    with open(mic, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "target_organism", "mic_value_um", "unit", "assay", "source_db"])
        w.writerow([SEQS[4], "E. coli", "4.0", "uM", "", "DBAASP"])
        w.writerow(["INVALID!", "E. coli", "4.0", "uM", "", "DBAASP"])

    out = tmp_path / "processed" / "generative_expanded.csv"
    exp.main(
        [
            "--generative",
            str(generative),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--out",
            str(out),
        ]
    )

    import csv as _csv

    with open(out, newline="") as f:
        rows = list(_csv.DictReader(f))
    seqs = [r["sequence"] for r in rows]
    srcs = {r["sequence"]: r["source_dbs"] for r in rows}

    assert len(seqs) == len(set(seqs)), "global dedup required"
    assert set(seqs) == set(SEQS[:5])
    assert srcs[SEQS[1]] == "marlys"
    assert srcs[SEQS[2]] == "dramp"
    assert srcs[SEQS[3]] == "apd"
    assert srcs[SEQS[4]] == "dbaasp"


def test_property_summary_runs():
    text = exp._property_summary(SEQS)
    assert "n=6" in text and "charge" in text
