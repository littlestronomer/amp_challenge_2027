"""De-risking additions: clustered-split eval + hemolysis safety component.

A4.1: identity clustering must group single-mutation homologs, never group
unrelated peptides, and split without homolog leakage.
A4.3: the hemolysis builder's label rule (min-HC50 ≤ ceiling → risky), its
report mode, the HemoScorer's direction convention (safety = 1 − p_risky),
and byte-compat of --w-safety 0.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from train_reward_classifier import cluster_split_records, greedy_identity_clusters

from amp_challenge_2027 import score as score_mod
from amp_challenge_2027.pipeline import DEFAULT_WEIGHTS, build_composite_scorer

# ---------------------------------------------------------------------------
# A4.1 — identity clustering
# ---------------------------------------------------------------------------


def test_clustering_groups_homologs_not_unrelated():
    seqs = [
        "KLLKLLKKLLKL",  # leader family
        "KLLKLLKKLLKM",  # single mutation → same cluster
        "KLLKLLKKLLKML",  # one insertion → same cluster
        "GIGKFLHSAKKFGKAFVGEIMNS",  # unrelated (length also differs)
        "RRWWVIKW",  # unrelated
    ]
    assign = greedy_identity_clusters(seqs, threshold=0.7)
    assert assign["KLLKLLKKLLKL"] == assign["KLLKLLKKLLKM"] == assign["KLLKLLKKLLKML"]
    leaders = {assign[s] for s in seqs}
    assert len(leaders) == 3  # homolog family + two unrelated
    # deterministic across calls
    assert greedy_identity_clusters(seqs, threshold=0.7) == assign


def test_clustering_length_prefilter_correct():
    # A 8-mer and a 50-mer can never reach ratio 0.7 — prefilter must skip.
    assign = greedy_identity_clusters(["KKKKKKKK", "K" * 50], threshold=0.7)
    assert assign["KKKKKKKK"] != assign["K" * 50]


def test_cluster_split_no_homolog_leakage():
    records = [
        {"sequence": s, "label": 1}
        for s in [
            "KLLKLLKKLLKL",
            "KLLKLLKKLLKM",  # family A
            "GIGKFLHSAKKFGKAFVGEIMNS",
            "GIGKFLHSAKKFGKAFVGEIMNSK",  # family B
            "RRWWVIKW",  # family C
        ]
    ]
    train, val, n_clusters = cluster_split_records(records, threshold=0.7, seed=42)
    assert n_clusters == 3
    train_seqs, val_seqs = {r["sequence"] for r in train}, {r["sequence"] for r in val}
    assert not (train_seqs & val_seqs)
    # Whole families land on one side (no straddling)
    for family in (
        ("KLLKLLKKLLKL", "KLLKLLKKLLKM"),
        ("GIGKFLHSAKKFGKAFVGEIMNS", "GIGKFLHSAKKFGKAFVGEIMNSK"),
    ):
        side_train = family[0] in train_seqs
        assert (family[1] in train_seqs) == side_train


def test_cluster_split_deterministic():
    records = [
        {"sequence": s, "label": 0} for s in [f"KLLKLLKKLLK{i}" for i in range(10)] + ["RRWWVIKW"]
    ]
    t1, v1, _ = cluster_split_records(records, seed=7)
    t2, v2, _ = cluster_split_records(records, seed=7)
    assert [r["sequence"] for r in t1] == [r["sequence"] for r in t2]
    assert [r["sequence"] for r in v1] == [r["sequence"] for r in v2]


# ---------------------------------------------------------------------------
# A4.3 — hemolysis label builder
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, header: list[str], rows: list[list[str]]) -> Path:
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return p


@pytest.fixture()
def hemo_world(tmp_path):
    peptides = _write(
        tmp_path,
        "peptides.csv",
        ["id", "sequence"],
        [["1", "KLLKLLKKLLKL"], ["2", "GIGKFLHSAKKFGKAFVGEIMNS"], ["3", "RRWWVIKW"]],
    )
    raw = _write(
        tmp_path,
        "hemolysis_raw.csv",
        ["peptide_id", "kind", "target", "value", "unit", "note", "raw"],
        [
            # peptide 1: 50-60% lysis at 6 µM → RISKY (worst=6)
            ["1", "50-60% Hemolysis", "Human erythrocytes", "6", "µM", "", "{}"],
            # same peptide, benign at high conc (band 0-10% @ 400 µM)
            ["1", "0-10% Hemolysis", "Human erythrocytes", "400", "µM", "", "{}"],
            # peptide 2: only benign-at-high → SAFE
            ["2", "0-10% Hemolysis", "Human erythrocytes", "300", "µM", "", "{}"],
            # cytotoxicity against non-RBC cells → excluded by --targets
            ["2", "50% Cell death", "HeLa", "10", "µM", "", "{}"],
            # peptide 3: IC50 (counts as 50% band) given in µg/ml → converted, risky
            ["3", "IC50", "Human erythrocytes", "64", "µg/ml", "", "{}"],
            # unknown peptide id
            ["99", "50-60% Hemolysis", "Human erythrocytes", "10", "µM", "", "{}"],
            # unconvertible value
            ["1", "50-60% Hemolysis", "Human erythrocytes", "n/a", "µM", "", "{}"],
            # band-less measure
            ["3", "Visual inspection", "Human erythrocytes", "10", "µM", "", "{}"],
        ],
    )
    return peptides, raw


def test_band_midpoint_parsing():
    from build_hemolysis_labels import band_midpoint

    assert band_midpoint("50-60% Hemolysis") == 55.0
    assert band_midpoint("0-10% Hemolysis") == 5.0
    assert band_midpoint("90-100% Hemolysis") == 95.0
    assert band_midpoint("50% Cell death") == 50.0
    assert band_midpoint("IC50") == 50.0
    assert band_midpoint("MHC") == 50.0
    assert band_midpoint("Visual inspection") is None
    assert band_midpoint("") is None


def test_hemolysis_builder_band_rule(hemo_world, tmp_path):
    from build_hemolysis_labels import main as hemo_main

    peptides, raw = hemo_world
    out = tmp_path / "hemolysis_labels.csv"
    hemo_main(["--raw", str(raw), "--peptides", str(peptides), "--out", str(out)])

    with open(out, newline="") as f:
        rows = {r["sequence"]: r for r in csv.DictReader(f)}
    # peptide 1: risky band at 6 µM dominates its benign high-conc row
    assert rows["KLLKLLKKLLKL"]["label"] == "active"
    assert float(rows["KLLKLLKKLLKL"]["hc50_um"]) == 6.0
    # peptide 2: benign at 300 µM (≥ ceiling) → safe, HeLa row excluded
    assert rows["GIGKFLHSAKKFGKAFVGEIMNS"]["label"] == "inactive"
    assert float(rows["GIGKFLHSAKKFGKAFVGEIMNS"]["hc50_um"]) == 300.0
    # peptide 3: IC50 64 µg/ml → µM via MW → risky
    from build_ranking_labels import peptide_mw

    expected = 64 * 1000.0 / peptide_mw("RRWWVIKW")
    assert rows["RRWWVIKW"]["label"] == "active"
    assert float(rows["RRWWVIKW"]["hc50_um"]) == pytest.approx(expected, rel=1e-3)
    assert "99" not in rows  # unmatched ids produce no label

    # report mode writes nothing and summarizes
    hemo_main(["--raw", str(raw), "--peptides", str(peptides), "--report"])
    assert not (tmp_path / "hemolysis_labels_report.csv").exists()


def test_hemolysis_builder_knobs_change_rule(hemo_world, tmp_path):
    from build_hemolysis_labels import main as hemo_main

    peptides, raw = hemo_world
    out = tmp_path / "strict.csv"
    # risk-band 70: peptide 1's 55% band no longer counts as risky →
    # falls through to its benign-at-400µM row → safe.
    hemo_main(
        ["--raw", str(raw), "--peptides", str(peptides), "--risk-band", "70", "--out", str(out)]
    )
    with open(out, newline="") as f:
        rows = {r["sequence"]: r for r in csv.DictReader(f)}
    assert rows["KLLKLLKKLLKL"]["label"] == "inactive"
    assert float(rows["KLLKLLKKLLKL"]["hc50_um"]) == 400.0


# ---------------------------------------------------------------------------
# A4.3 — HemoScorer wiring
# ---------------------------------------------------------------------------


def test_hemo_scorer_graceful_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(score_mod, "REWARD_HEMO_DIR", tmp_path)
    assert score_mod.HemoScorer.load(device="cpu") is None


def test_safety_weight_zero_is_byte_compatible(tmp_path, monkeypatch):
    ref = ["KLLKLLKKLLKL", "GIGKFLHSAKKFGKAFVGEIMNS", "RRWWVIKW"] * 4
    monkeypatch.setattr(score_mod, "REWARD_DIR", tmp_path)
    monkeypatch.setattr(score_mod, "REWARD_HEMO_DIR", tmp_path)
    assert DEFAULT_WEIGHTS["safety"] == 0.0
    legacy = build_composite_scorer(ref, w_activity=0.0, w_precision=0.0, device="cpu")
    with_flag = build_composite_scorer(
        ref, w_activity=0.0, w_precision=0.0, w_safety=0.0, device="cpu"
    )
    assert legacy.names == with_flag.names == ["conformity"]
    seqs = ["KLLKLLKKLLKL", "RRWWVIKWGIGK"]
    assert (legacy.score(seqs)[0] == with_flag.score(seqs)[0]).all()


def test_safety_flag_requests_component_or_drops(tmp_path, monkeypatch):
    ref = ["KLLKLLKKLLKL"] * 4
    monkeypatch.setattr(score_mod, "REWARD_HEMO_DIR", tmp_path)
    monkeypatch.setattr(score_mod, "REWARD_DIR", tmp_path)
    # artifact absent → component dropped; nothing else requested → None
    assert (
        build_composite_scorer(
            ref, w_activity=0.0, w_precision=0.0, w_conformity=0.0, w_safety=1.0, device="cpu"
        )
        is None
    )


def test_binary_trainer_smoke_uneven_batches(tmp_path):
    """Binary mode must survive a ragged last val batch (regression: the
    panel refactor briefly made it np.array over per-batch arrays)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import csv as _csv

    labels_csv = tmp_path / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["sequence", "label"])
        # 10 sequences, batch_size 4 → val batches of 4/4/... with the last
        # one ragged after the 80/20 split.
        for i in range(5):
            w.writerow([("KLLKLLKKLL" * 2)[: 10 + i], "active"])
            w.writerow([("AADDGGVVWW" * 2)[: 10 + i], "inactive"])

    from train_reward_classifier import train

    train(
        labels_csv,
        esm_model="facebook/esm2_t6_8M_UR50D",
        unfreeze_layers=0,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        seed=42,
        device="cpu",
        out_dir=tmp_path / "out",
        ensemble_size=1,
        calibrate=True,
        panel=False,
    )
    assert (tmp_path / "out" / "classifier.pt").exists()
