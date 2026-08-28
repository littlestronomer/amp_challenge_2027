"""Panel-aware ranker: genus multi-hot training + breadth scoring + wiring.

Hermetic tests run everywhere (label aggregation, graceful absence, zero-weight
byte-compat, MDR genera derivation). The torch-gated block exercises the real
training path (tiny ESM, synthetic labels, CPU, seconds) and the PanelScorer
end-to-end — it skips in the torch-less dev venv and runs in the box suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from amp_challenge_2027 import score as score_mod
from amp_challenge_2027.config import (
    MDR_PANEL_GENERA,
    PANEL_GENERA,
    PANEL_GENUS_TO_GRAM,
)
from amp_challenge_2027.pipeline import DEFAULT_WEIGHTS, build_composite_scorer

# ---------------------------------------------------------------------------
# Config derivation (single-sourced taxonomy)
# ---------------------------------------------------------------------------


def test_panel_genera_derivation_stable():
    assert PANEL_GENERA == [
        "A. baumannii",
        "E. cloacae",
        "E. coli",
        "K. pneumoniae",
        "P. aeruginosa",
        "S. enterica",
        "B. subtilis",
        "S. aureus",
        "E. faecalis",
        "E. faecium",
    ]
    assert len(set(PANEL_GENERA)) == 10
    assert len(MDR_PANEL_GENERA) == 7  # 8 MDR strains collapse to 7 genera
    assert "E. cloacae" not in MDR_PANEL_GENERA and "S. enterica" not in MDR_PANEL_GENERA
    assert set(PANEL_GENUS_TO_GRAM) == set(PANEL_GENERA)


# ---------------------------------------------------------------------------
# Panel label aggregation (trainer loader, pure functions)
# ---------------------------------------------------------------------------


def _write_full_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "organism", "mic_um", "band", "label"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_panel_data_aggregation_and_masking(tmp_path):
    from train_reward_classifier import load_panel_data

    csv_path = _write_full_csv(
        tmp_path / "activity_labels_full.csv",
        [
            {
                "sequence": "KLLKLLKKLLKL",
                "organism": "E. coli",
                "mic_um": "2.0",
                "band": "potent",
                "label": "active",
            },
            {
                "sequence": "KLLKLLKKLLKL",
                "organism": "P. aeruginosa",
                "mic_um": "40.0",
                "band": "inactive",
                "label": "inactive",
            },
            {
                "sequence": "KLLKLLKKLLKL",
                "organism": "S. enterica",
                "mic_um": "20.0",
                "band": "weak",
                "label": "",
            },  # masked
            {
                "sequence": "GLFSLL",
                "organism": "E. coli",
                "mic_um": "3.0",
                "band": "potent",
                "label": "active",
            },  # invalid len
            {
                "sequence": "GLLSGINAASKK",
                "organism": "Vibrio splendidus",
                "mic_um": "4.0",
                "band": "potent",
                "label": "active",
            },  # unmapped
        ],
    )
    records = load_panel_data(csv_path)
    assert [r["sequence"] for r in records] == [
        "KLLKLLKKLLKL"
    ]  # short seq + unmapped genus dropped

    r = records[0]
    i_ec, i_pa = PANEL_GENERA.index("E. coli"), PANEL_GENERA.index("P. aeruginosa")
    assert r["targets"][i_ec] == 1.0 and r["mask"][i_ec] == 1.0
    assert r["targets"][i_pa] == 0.0 and r["mask"][i_pa] == 1.0
    masked = r["mask"] == 0
    assert (r["targets"][masked] == 0.0).all()  # zero-filled; mask is source of truth


def test_load_panel_data_conflict_resolves_active(tmp_path):
    from train_reward_classifier import load_panel_data

    csv_path = _write_full_csv(
        tmp_path / "full.csv",
        [
            {
                "sequence": "KLLKLLKKLLKL",
                "organism": "E. coli",
                "mic_um": "2.0",
                "band": "potent",
                "label": "inactive",
            },
            {
                "sequence": "KLLKLLKKLLKL",
                "organism": "Escherichia coli K-12",
                "mic_um": "1.0",
                "band": "potent",
                "label": "active",
            },
        ],
    )
    records = load_panel_data(csv_path)
    assert records[0]["targets"][PANEL_GENERA.index("E. coli")] == 1.0


def test_panel_scorer_graceful_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(score_mod, "REWARD_DIR", tmp_path)
    assert score_mod.PanelScorer.load(device="cpu") is None
    # config present but wrong task → None as well
    (tmp_path / "config.json").write_text(json.dumps({"task": "binary"}))
    (tmp_path / "classifier_panel.pt").write_bytes(b"junk")
    assert score_mod.PanelScorer.load(device="cpu") is None


# ---------------------------------------------------------------------------
# Composite wiring: zero-weight byte-compat + component activation
# ---------------------------------------------------------------------------


def test_default_weights_zero_for_new_components():
    assert DEFAULT_WEIGHTS["breadth"] == 0.0 and DEFAULT_WEIGHTS["mdr"] == 0.0
    # legacy keys untouched
    assert DEFAULT_WEIGHTS["activity"] == 1.0
    assert DEFAULT_WEIGHTS["conformity"] == 0.5
    assert DEFAULT_WEIGHTS["precision"] == 0.5


def test_pipeline_zero_weight_matches_legacy_component_set(tmp_path, monkeypatch):
    ref = ["KLLKLLKKLLKL", "GIGKFLHSAKKFGKAFVGEIMNS", "RRWWVIKW"] * 4
    monkeypatch.setattr(score_mod, "REWARD_DIR", tmp_path)  # no classifiers anywhere

    legacy = build_composite_scorer(ref, w_activity=0.0, w_precision=0.0, device="cpu")
    new = build_composite_scorer(
        ref, w_activity=0.0, w_precision=0.0, w_breadth=0.0, w_mdr=0.0, device="cpu"
    )
    assert legacy is not None and new is not None
    assert legacy.names == new.names == ["conformity"]

    seqs = ["KLLKLLKKLLKL", "RRWWVIKWGIGK"]
    c_legacy, p_legacy = legacy.score(seqs)
    c_new, p_new = new.score(seqs)
    assert np.array_equal(c_legacy, c_new)
    assert set(p_legacy) == set(p_new)


def test_pipeline_breadth_flags_request_components(tmp_path, monkeypatch):
    ref = ["KLLKLLKKLLKL"] * 4
    monkeypatch.setattr(score_mod, "REWARD_DIR", tmp_path)
    scorer = build_composite_scorer(
        ref,
        w_activity=0.0,
        w_precision=0.0,
        w_conformity=0.0,
        w_breadth=1.0,
        w_mdr=0.5,
        device="cpu",
    )
    assert scorer is None  # panel artifact absent → both dropped → nothing left


# ---------------------------------------------------------------------------
# Torch-gated: real loader/model/loss/scorer round trip (runs on the GPU box)
# ---------------------------------------------------------------------------


def test_panel_training_smoke_and_scorer_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from train_reward_classifier import (
        load_panel_data,
        masked_bce,
        train,
    )

    # Synthetic long-format labels: a few clearly-separated patterns.
    rows = []
    for i in range(24):
        pos = "KKLLKKLLKKLL"[: 8 + i % 5]
        neg = "AADDAA" * 2
        rows.append(
            {
                "sequence": pos,
                "organism": "E. coli",
                "mic_um": "1",
                "band": "potent",
                "label": "active",
            }
        )
        rows.append(
            {
                "sequence": neg + "K",
                "organism": "P. aeruginosa",
                "mic_um": "64",
                "band": "inactive",
                "label": "inactive",
            }
        )
    data_csv = _write_full_csv(tmp_path / "activity_labels_full.csv", rows)
    records = load_panel_data(data_csv)
    assert records, "synthetic panel data should aggregate"

    # Masked BCE math sanity: perfect predictions → ~0 loss; mask excludes entries.
    import torch

    logits = torch.tensor([[5.0, -5.0], [5.0, 5.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (1,*) slot masked below
    mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    loss_all = masked_bce(logits, targets, mask)
    assert loss_all.item() < 0.1

    out_dir = tmp_path / "reward"
    train(
        data_csv,
        esm_model="facebook/esm2_t6_8M_UR50D",  # small + cached on the box
        unfreeze_layers=0,  # keep the smoke fast: head-only training
        epochs=2,
        batch_size=4,
        lr=1e-4,
        seed=42,
        device="cpu",
        out_dir=out_dir,
        ensemble_size=1,
        calibrate=True,
        panel=True,
    )

    art = out_dir / "classifier_panel.pt"
    assert art.exists()
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["task"] == "panel" and cfg["genera"] == PANEL_GENERA
    assert not (out_dir / "classifier.pt").exists()  # binary artifact untouched

    # Scorer round trip from the promoted artifact.
    monkeypatch.setattr(score_mod, "REWARD_DIR", out_dir)
    scorer = score_mod.PanelScorer.load(device="cpu")
    assert scorer is not None and scorer.genera == PANEL_GENERA
    b = scorer.breadth(["KKLLKKLL", "AADDAAAA"])
    m = scorer.mdr_breadth(["KKLLKKLL", "AADDAAAA"])
    assert b.shape == (2,) and m.shape == (2,)
    assert (b >= 0).all() and (b <= 1).all()
    assert "E. coli" in scorer.mdr_genera and "S. enterica" not in scorer.mdr_genera

    # transformers imported transitively above keeps the torch-gate honest
    assert transformers is not None
