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
        ["peptide_id", "kind", "target", "value", "unit"],
        [
            ["1", "HC50", "hRBC", "50", "µM"],  # risky (≤128)
            ["1", "HC50", "hRBC", "200", "µM"],  # same seq, higher reading
            ["2", "HC50", "hRBC", "400", "µM"],  # safe (>128)
            ["2", "EC50", "HeLa", "10", "µM"],  # cytotoxic, not hemolysis → excluded
            ["3", "Hemolysis", "hRBC", "64", "µg/ml"],  # converted via MW → µM
            ["99", "HC50", "hRBC", "10", "µM"],  # unknown peptide id
            ["1", "HC50", "hRBC", "n/a", "µM"],  # unparseable
        ],
    )
    return peptides, raw


def test_hemolysis_builder_rule_and_report(hemo_world, tmp_path):
    from build_hemolysis_labels import main as hemo_main

    peptides, raw = hemo_world
    out = tmp_path / "hemolysis_labels.csv"
    hemo_main(
        ["--raw", str(raw), "--peptides", str(peptides), "--ceiling", "128", "--out", str(out)]
    )

    with open(out, newline="") as f:
        rows = {r["sequence"]: r for r in csv.DictReader(f)}
    # peptide 1: min(50, 200) = 50 ≤ 128 → risky
    assert rows["KLLKLLKKLLKL"]["label"] == "active"
    assert float(rows["KLLKLLKKLLKL"]["hc50_um"]) == 50.0
    # peptide 2: only hemolysis reading is 400 → safe
    assert rows["GIGKFLHSAKKFGKAFVGEIMNS"]["label"] == "inactive"
    # peptide 3: 64 µg/ml → µM via MW (23-mer, ~2.4 kDa → ~26 µM) → risky
    from build_ranking_labels import peptide_mw

    expected = 64 * 1000.0 / peptide_mw("RRWWVIKW")
    assert float(rows["RRWWVIKW"]["hc50_um"]) == pytest.approx(expected, rel=1e-3)
    assert rows["RRWWVIKW"]["label"] == "active"
    assert "99" not in rows  # unmatched ids produce no label

    # report mode writes nothing and summarizes
    hemo_main(["--raw", str(raw), "--peptides", str(peptides), "--report"])


def test_hemolysis_builder_kind_filter_selective(hemo_world, tmp_path):
    from build_hemolysis_labels import main as hemo_main

    peptides, raw = hemo_world
    out = tmp_path / "strict.csv"
    # Strict regex: only literal "HC50" kinds — the "Hemolysis" row drops.
    hemo_main(
        ["--raw", str(raw), "--peptides", str(peptides), "--kinds", "^hc50$", "--out", str(out)]
    )
    with open(out, newline="") as f:
        seqs = {r["sequence"] for r in csv.DictReader(f)}
    assert seqs == {"KLLKLLKKLLKL", "GIGKFLHSAKKFGKAFVGEIMNS"}


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
