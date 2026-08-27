"""Rebuild-corpus wrapper: sandboxed outputs + byte-identical canonical freeze.

Hermetic (tmp-only world, real repo data never read). Verifies the contract
the seed-44 lineage depends on:

  1. Canonical ``generative.csv`` and every other pre-existing processed file
     stay byte-identical across a rebuild run.
  2. Rebuilt base corpus uses the same parse+curate code as the original
     build (MarLys headers, dedup, alphabet/length filters).
  3. Merged expanded corpus obeys source priority (marlys > dramp > dbaasp),
     drops reference-overlap sequences from EVERY source, and writes the same
     column format as ``build_expanded_generative.py``.
  4. Determinism: two runs over identical inputs produce identical bytes.
  5. The integrity guard actually fires when a watched file changes.
"""

from __future__ import annotations

import csv

import pytest
import rebuild_corpus

# ---------------------------------------------------------------------------
# Fixture: a complete raw/processed world inside tmp_path
# ---------------------------------------------------------------------------

MARLYS_FASTA = """>MLAMP0000001 len=12 charge=5.00 disulfide=0 dbs=DBAASP activity=antibacterial
KLLKLLKKLLKL
>MLAMP0000002 len=23 charge=3.10 disulfide=0 dbs=APD activity=antibacterial|antiGram+
GIGKFLHSAKKFGKAFVGEIMNS
>MLAMP0000003 len=8 charge=-1.20 disulfide=0 dbs=DRAMP activity=
AKXVKLLG
>MLAMP0000004 len=20 charge=4.40 disulfide=0 dbs=DBAASP activity=antibacterial
FKIGGAVKKVLKAAKILGGV
>MLAMP0000005 len=15 charge=2.90 disulfide=0 dbs=LAMP activity=antibacterial
GLASFLGKKALCHLA
>MLAMP0000006 len=8 charge=6.00 disulfide=0 dbs=APD activity=antibacterial
RRWWVIKW
>MLAMP0000007 len=51 charge=51.0 disulfide=0 dbs=DBAASP activity=antibacterial
KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
"""

DRAMP_FASTA = """>dramp_1 len=15
KWKLFKKIGAVLKVL
>dramp_2 len=12
KLLKLLKKLLKL
>dramp_3 len=20
FKIGGAVKKVLKAAKILGGV
>dramp_4 len=16
IVWCTWAFKCCGKKAL
"""

DBAASP_PEPTIDES = """peptide_id,sequence
P1,KWKLFKKIGAVLKVL
P2,GLLSGINAASKKAAKHGGKVSAW
P3,CWRWWXXPK
P4,GLFDIVKKVVGALGSL
"""

DBAASP_ACTIVITY = """peptide_id,target_organism,mic
P1,Escherichia coli ATCC 25922,12.5 µM
P1,Vibrio splendidus,32 µM
P2,Staphylococcus aureus ATCC 12600,<= 64
P4,Escherichia coli (human erythrocytes lysis control),15
P1,Human erythrocytes,25
"""


@pytest.fixture()
def world(tmp_path):
    """Lay out raw sources, a frozen processed dir, and a tiny reference set."""
    raw = tmp_path / "raw"
    (raw / "marlys").mkdir(parents=True)
    (raw / "dbaasp").mkdir()
    (raw / "dramp").mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()

    (raw / "marlys" / "marlys.fasta").write_text(MARLYS_FASTA)
    (raw / "dramp" / "general.fasta").write_text(DRAMP_FASTA)
    (raw / "dbaasp" / "peptides.csv").write_text(DBAASP_PEPTIDES)
    (raw / "dbaasp" / "activity.csv").write_text(DBAASP_ACTIVITY)

    # Deliberately divergent canonical corpus (subset of what MarLys contains):
    # exercises both the DIFFERS verdict and fallback behavior of the merge.
    with open(processed / "generative.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "source_id", "charge", "activity", "source_dbs"])
        w.writerow(["KLLKLLKKLLKL", "MLAMP0000001", 5.0, "antibacterial", "DBAASP"])
        w.writerow(["GLASFLGKKALCHLA", "MLAMP0000005", 2.9, "antibacterial", "LAMP"])

    # Sentinel file whose mutation the integrity guard must detect.
    sentinel = processed / "frozen_marker.txt"
    sentinel.write_text("seed-44 lineage data\n")

    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "antibacterial.fasta").write_text(
        ">ref1 len=20\nFKIGGAVKKVLKAAKILGGV\n>ref2 len=13\nAAAAAAAAAAAAA\n"
    )
    return {"raw": raw, "processed": processed, "ref": tmp_path / "ref"}


def _run(world, tmp_path, out_name):
    out_dir = tmp_path / out_name
    summary = rebuild_corpus.main(
        [
            "--raw-dir",
            str(world["raw"]),
            "--processed-dir",
            str(world["processed"]),
            "--out-dir",
            str(out_dir),
            "--canonical-generative",
            str(world["processed"] / "generative.csv"),
            "--antibacterial",
            str(world["ref"] / "antibacterial.fasta"),
        ]
    )
    return out_dir, summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_canonical_and_watchdir_stay_byte_identical(world, tmp_path):
    _run(world, tmp_path, "rebuild_a")
    assert (world["processed"] / "generative.csv").read_text().startswith("sequence,source_id")
    assert (world["processed"] / "frozen_marker.txt").read_text() == "seed-44 lineage data\n"


def test_base_rebuild_matches_original_builder_semantics(world, tmp_path):
    out_dir, _summary = _run(world, tmp_path, "rebuild_a")

    with open(out_dir / "generative.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    # Invalid dropped: 'AKXVKLLG' (nonstandard X) and 51x K (>max length).
    seqs = [r["sequence"] for r in rows]
    assert seqs == [
        "KLLKLLKKLLKL",
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "FKIGGAVKKVLKAAKILGGV",  # base build does NOT exclude reference overlap
        "GLASFLGKKALCHLA",
        "RRWWVIKW",
    ]
    assert rows[0]["source_dbs"] == "DBAASP"

    with open(out_dir / "mic.csv", newline="") as f:
        mic = list(csv.DictReader(f))
    assert [(r["sequence"], r["target_organism"]) for r in mic] == [
        ("KWKLFKKIGAVLKVL", "E. coli"),  # unknown species (Vibrio) skipped
        ("GLLSGINAASKKAAKHGGKVSAW", "S. aureus"),
        ("GLFDIVKKVVGALGSL", "E. coli"),  # panel-matched → unlocks hemolysis capture
    ]

    with open(out_dir / "hemolysis.csv", newline="") as f:
        hemo = list(csv.DictReader(f))
    # Hemolysis rows require BOTH the erythrocyte keyword AND a panel species
    # match (data.py gates on normalize_species before the keyword check), so
    # the bare 'Human erythrocytes' row is skipped but P4's compound row works.
    assert [r["sequence"] for r in hemo] == ["GLFDIVKKVVGALGSL"]


def test_expanded_merge_priority_reference_overlap_and_format(world, tmp_path):
    out_dir, summary = _run(world, tmp_path, "rebuild_a")

    assert (out_dir / "generative_expanded.csv").exists()
    assert summary["violations"] == []
    assert summary["expanded_size"] == 8

    with open(out_dir / "generative_expanded.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["sequence", "source_id", "charge", "activity", "source_dbs"]
        rows = list(reader)

    got = [(r["sequence"], r["source_id"].split("_")[0]) for r in rows]
    # Priority order buckets (stable within source): marlys canon → marlys fasta
    # recovered → dramp → dbaasp; M4/D3 excluded entirely (reference overlap);
    # D2 and P1 are duplicates resolved in favor of earlier priority sources.
    assert got == [
        ("KLLKLLKKLLKL", "marlys"),
        ("GLASFLGKKALCHLA", "marlys"),
        ("GIGKFLHSAKKFGKAFVGEIMNS", "marlys"),
        ("RRWWVIKW", "marlys"),
        ("KWKLFKKIGAVLKVL", "dramp"),
        ("IVWCTWAFKCCGKKAL", "dramp"),
        ("GLFDIVKKVVGALGSL", "dbaasp"),
        ("GLLSGINAASKKAAKHGGKVSAW", "dbaasp"),
    ]

    stats = summary["merge_stats"]
    assert stats["reference_overlap"] == 2
    assert stats["duplicate"] == 4  # marlys-fasta ×{canon M1,M5} + KLLK@dramp + KW@dbaasp
    assert stats["invalid"] == 1  # the 51x-K row passes the loader's alphabet-only gate


def test_deterministic_reruns_produce_identical_bytes(world, tmp_path):
    out_a, _ = _run(world, tmp_path, "rebuild_a")
    out_b, _ = _run(world, tmp_path, "rebuild_b")
    for name in ("generative.csv", "mic.csv", "hemolysis.csv", "generative_expanded.csv"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes(), name


def test_integrity_guard_fires_on_mutation(world, tmp_path, monkeypatch):
    _run(world, tmp_path, "rebuild_a")
    assert (world["processed"] / "frozen_marker.txt").read_text() == "seed-44 lineage data\n"

    # Mutate the watched file DURING a run — after the baseline snapshot but
    # before the closing one — the only moment the guard can legitimately fire.
    marker = world["processed"] / "frozen_marker.txt"
    real_snapshot = rebuild_corpus.dir_snapshot
    calls = {"n": 0}

    def mutating_snapshot(root):
        snap = real_snapshot(root)
        calls["n"] += 1
        if calls["n"] == 1:
            marker.write_text("tampered\n")
        return snap

    monkeypatch.setattr(rebuild_corpus, "dir_snapshot", mutating_snapshot)

    with pytest.raises(SystemExit) as exc:
        rebuild_corpus.main(
            [
                "--raw-dir",
                str(world["raw"]),
                "--processed-dir",
                str(world["processed"]),
                "--out-dir",
                str(tmp_path / "rebuild_c"),
                "--canonical-generative",
                str(world["processed"] / "generative.csv"),
                "--antibacterial",
                str(world["ref"] / "antibacterial.fasta"),
            ]
        )
    assert exc.value.code == rebuild_corpus.INTEGRITY_EXIT_CODE


def test_missing_sources_degrade_gracefully(world, tmp_path):
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    summary = rebuild_corpus.main(
        [
            "--raw-dir",
            str(empty_raw),
            "--processed-dir",
            str(tmp_path / "nothing"),  # no mic.csv anywhere
            "--out-dir",
            str(tmp_path / "rebuild_empty"),
            "--canonical-generative",
            str(tmp_path / "absent_generative.csv"),
            "--antibacterial",
            str(world["ref"] / "antibacterial.fasta"),
        ]
    )
    assert summary["base_counts"] == {}
    assert summary["expanded_size"] == 0
    assert (tmp_path / "rebuild_empty" / "generative_expanded.csv").exists()
