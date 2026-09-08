"""Exact head identity and missing-provenance behavior; no backbone needed."""

from __future__ import annotations

import json

import audit_reward_artifacts as audit
import pytest
from experiment_utils import sha256, write_json


def setup_artifacts(tmp_path):
    torch = pytest.importorskip("torch")
    reward, hemo = tmp_path / "reward", tmp_path / "hemo"
    recorded = {}
    for name, root, stem, task in (("activity", reward, "classifier", "binary"),
                                   ("panel", reward, "classifier_panel", "panel"),
                                   ("hemolysis", hemo, "classifier", "binary")):
        cfg = {"esm_model": audit.baseline.BACKBONE, "unfreeze_layers": 0, "checkpoint_format": "head-only",
               "task": task, "temperature": 0.95}
        if task == "panel":
            cfg["genera"] = audit.baseline.PANEL_GENERA
        write_json(root / f"{stem}_config.json", cfg)
        outputs = 10 if task == "panel" else 1
        state = {"dense.weight": torch.zeros(480, 480), "dense.bias": torch.zeros(480),
                 "classifier.weight": torch.zeros(outputs, 480), "classifier.bias": torch.zeros(outputs)}
        if name == "hemolysis":
            state["dense.bias"][0] = 1
        torch.save(state, root / f"{stem}.pt")
        member = root / "member7"
        member.mkdir(exist_ok=True)
        # Different torch archive filename makes serialization differ, tensors remain identical.
        intermediate = member / "different-name.pt"
        torch.save(state, intermediate)
        intermediate.rename(member / f"{stem}.pt")
        recorded[name] = {"files": {p.name: sha256(p) for p in (root / f"{stem}.pt", root / f"{stem}_config.json")}}
    write_json(tmp_path / "source/run.json", {"kind": "paired_top100_v1", "classifiers": recorded})
    return ["--reward-dir", str(reward), "--hemo-dir", str(hemo), "--source", str(tmp_path / "source"),
            "--out", str(tmp_path / "inventory")]


def test_inventory_matches_tensors_without_guessing_seed_or_validation(tmp_path):
    argv = setup_artifacts(tmp_path)
    audit.main(argv)
    report = json.loads((tmp_path / "inventory/inventory.json").read_text())
    assert not report["training_provenance_verified"]
    for item in report["artifacts"].values():
        assert item["status"] == "provenance_incomplete"
        assert item["identity_status"] == "unique_tensor_match"
        assert not item["matches"][0]["byte_identical"]
        assert item["matches"][0]["tensor_identical"]
        assert not item["validation_reproduced"]
        assert "seed" not in item
    before = (tmp_path / "inventory/inventory.json").read_bytes()
    audit.main(argv)
    assert before == (tmp_path / "inventory/inventory.json").read_bytes()


def test_missing_explicit_config_never_uses_shared_fallback(tmp_path):
    argv = setup_artifacts(tmp_path)
    config = tmp_path / "hemo/classifier_config.json"
    config.rename(config.with_name("config.json"))
    with pytest.raises(SystemExit) as error:
        audit.main(argv)
    assert error.value.code == 2
    report = json.loads((tmp_path / "inventory/inventory.json").read_text())
    assert report["artifacts"]["hemolysis"]["status"] == "invalid_artifact"


def test_changed_head_or_partial_head_fails_closed(tmp_path):
    torch = pytest.importorskip("torch")
    argv = setup_artifacts(tmp_path)
    torch.save({"classifier.bias": torch.zeros(1)}, tmp_path / "reward/classifier.pt")
    with pytest.raises(SystemExit):
        audit.main(argv)
    report = json.loads((tmp_path / "inventory/inventory.json").read_text())
    item = report["artifacts"]["activity"]
    assert item["status"] == "invalid_artifact"
    assert any("hashes differ" in issue for issue in item["issues"])
    assert any("complete head" in issue for issue in item["issues"])


def test_invalid_or_wrong_dimensional_metadata():
    assert audit.metadata_issue({"esm_model": "facebook/esm2_t48_15B_UR50D"}, "binary")
    cfg = {"esm_model": audit.baseline.BACKBONE, "unfreeze_layers": 0, "checkpoint_format": "head-only",
           "task": "binary", "temperature": float("nan")}
    assert audit.metadata_issue(cfg, "binary")
