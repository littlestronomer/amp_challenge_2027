"""Real offline snapshot -> split -> train -> sample -> score integration."""
# ruff: noqa: E402

import hashlib
import json
import random
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.conditional_data import fetch_sources, prepare_dataset, verify_snapshot
from amp_challenge_2027.conditional_research import (
    compare_experiments,
    sample_experiment,
    verify,
)
from amp_challenge_2027.conditional_training import _read_dataset, train_experiment
from amp_challenge_2027.config import PANEL_GENERA, PROJECT_ROOT


def _xlsx(path, rows):
    lines = ['<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for row_index, row in enumerate(rows, 1):
        lines.append(f'<row r="{row_index}">')
        for column, value in enumerate(row):
            lines.append(f'<c r="{chr(65 + column)}{row_index}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        lines.append("</row>")
    lines.append("</sheetData></worksheet>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", "".join(lines))


def _sources(tmp_path):
    rng = random.Random(731)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = sorted({"".join(rng.choices(alphabet, k=12)) for _ in range(72)})
    marlys = tmp_path / "marlys.csv"
    marlys.write_text("sequence\n" + "\n".join(sequences) + "\n")
    cards = tmp_path / "cards"
    cards.mkdir()
    for index, sequence in enumerate(sequences):
        card = {
            "id": index + 1, "sequence": sequence, "nTerminus": "H", "cTerminus": "OH",
            "stereochemistry": "L", "linearCyclic": "linear", "pubmed_id": str(10000 + index),
            "targetActivities": [{
                "targetSpecies": {"name": f"Escherichia coli ATCC {20000 + index}"},
                "activityMeasureGroup": {"name": "MIC"}, "concentration": "<=16",
                "unit": {"name": "µM"}, "medium": {"name": "MHB"},
            }],
            "hemoliticCytotoxicActivities": [{
                "targetCell": {"name": "human erythrocytes"},
                "activityMeasureForLysisGroup": {"name": "0-10%"},
                "concentration": "64", "unit": {"name": "µM"},
            }],
        }
        (cards / f"{index + 1}.json").write_text(json.dumps(card))
    dramp = tmp_path / "dramp.xlsx"
    _xlsx(dramp, [["Sequence", "DRAMP_ID"], [sequences[0], "fixture1"]])
    return marlys, cards, dramp


def test_real_offline_pipeline_and_training_provenance(tmp_path):
    marlys, cards, dramp = _sources(tmp_path)
    snapshot = tmp_path / "snapshot"
    def forbid_network(url):
        raise AssertionError(f"Offline fixture unexpectedly requested {url}")

    fetched = fetch_sources(snapshot, marlys=marlys, dbaasp_dir=cards,
                            dramp=dramp, opener=forbid_network)
    assert fetched == verify_snapshot(snapshot)
    data_dir = tmp_path / "prepared"
    dataset = prepare_dataset(snapshot, data_dir, seed=2027)
    assert {row["split"] for row in dataset["examples"]} == {
        "train", "validation", "calibration", "test",
    }
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = tmp_path / "trained"
        result = train_experiment(
            data_dir, out, "conditional", preset="A", seed=42, device="cpu",
            epochs=1, batch_size=4, examples_per_epoch=8,
            model_overrides=dict(hidden_size=16, num_heads=2, num_layers=2,
                                 ffn_inner=32, dropout=0.0),
        )
        assert result["status"] == "complete"
        verify(out, ["run.json", "checkpoint/model.pt", "checkpoint/config.json"])
        run = json.loads((out / "run.json").read_text())
        integrity = run["training_input_integrity"]
        assert integrity["status"] == "prepared_manifest_verified"
        assert integrity["snapshot_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
        assert integrity["parser_sha256"] == dataset["recipe"]["parser_sha256"]
        assert len(run["input_hashes"]) == 5  # prepared manifest + four sealed outputs
        schema = json.loads((out / "checkpoint/config.json").read_text())["assay_schema"]
        train_strains = {condition["strain"] for row in dataset["examples"]
                        if row["split"] == "train" for condition in row["conditions"]
                        if condition.get("strain")}
        calibration_strains = {condition["strain"] for row in dataset["examples"]
                              if row["split"] == "calibration" for condition in row["conditions"]
                              if condition.get("strain")}
        assert train_strains and calibration_strains
        assert set(schema["vocabularies"]["strain"]) == train_strains
        assert not calibration_strains & set(schema["vocabularies"]["strain"])
        example = next(row for row in dataset["examples"] if row["split"] == "train")
        request = tmp_path / "request.json"
        request.write_text(json.dumps({"conditions": example["conditions"]}))
        reference = tmp_path / "reference.fasta"
        reference.write_text(">reference\nRRRRRRRRRRRR\n")
        sample_dir = tmp_path / "draws"
        summary = sample_experiment(out, sample_dir, label="tiny", seed=42, draws=5,
                                    batch_size=2, device="cpu", conditions=request, reference=reference)
        assert summary["raw_draws"] == 5
        verify(sample_dir, ["run.json", "draws.csv", "raw.fasta"])

        class StubOracles:
            identity = {"kind": "offline_fixture_not_biological"}

            def score(self, sequences):
                return (np.full(len(sequences), .9), np.full(len(sequences), .1),
                        np.full((len(sequences), len(PANEL_GENERA)), .8))

        comparison_dir = tmp_path / "comparison"
        compared = compare_experiments([sample_dir], comparison_dir, baseline="tiny",
                                       reference=reference, device="cpu", scorer=StubOracles())
        assert compared["rows"][0]["raw_draws"] == 5
        assert compared["decisions"] == []
        verify(comparison_dir, ["comparison.json", "comparison.csv", "REPORT.md"])
    finally:
        torch.set_num_threads(old_threads)
    # Mutating even an ancillary prepared table is detected before model fitting.
    (data_dir / "observations.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Prepared dataset hash mismatch"):
        _read_dataset(data_dir)


def test_training_refuses_new_directory_inside_deployed_artifacts(tmp_path):
    sequences = ["ACDEFGHIKLMN", "PQRSTVWYACDE"]
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps({"schema_version": 1, "examples": [
        {"sequence": sequence, "family": str(i), "split": split, "conditions": [],
         "supervision_eligible": False}
        for i, (sequence, split) in enumerate(zip(sequences, ("train", "validation")))
    ]}))
    output = PROJECT_ROOT / "checkpoint/generator/__rejected_training_fixture__"
    assert not output.exists()
    with pytest.raises(ValueError, match="deployed artifacts"):
        train_experiment(data, output, "unconditional", preset="A", device="cpu", epochs=1)
    assert not output.exists()


def test_manual_fixture_is_explicitly_unverified(tmp_path):
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps({"schema_version": 1, "examples": [
        {"sequence": "ACDEFGHIKLMN", "family": "train", "split": "train", "conditions": [],
         "supervision_eligible": False},
        {"sequence": "PQRSTVWYACDE", "family": "val", "split": "validation", "conditions": [],
         "supervision_eligible": False},
    ]}))
    _, payload, _ = _read_dataset(data)
    assert payload["training_input_integrity"]["status"] == "unverified_manual_dataset"
