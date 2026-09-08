import json

import pytest
from experiment_utils import sha256, write_json, write_summary
from reconcile_observation_metadata import (
    conversion_diagnosis,
    index_peptides,
    join_status,
    main,
    metadata_status,
)


def sample(assay, concentration, sequence="AKKKKKKK"):
    return {"task": "activity", "sequence": sequence,
            "raw_row": json.dumps({"assay": assay, "concentration": concentration})}


def test_molar_operator_and_conversion():
    result = conversion_diagnosis(sample(">16 µM", "16 µM"))
    assert result["operator_changed"]
    assert result["diagnosis"] == "molar_consistent"
    assert conversion_diagnosis(sample("1000 nM", "1 µM"))["diagnosis"] == "molar_consistent"
    assert conversion_diagnosis(sample("1 mM", "1 µM"))["diagnosis"] == "molar_disagrees"


def test_mass_does_not_repair_or_guess():
    result = conversion_diagnosis(sample(">64 µg/ml", ">718.342 µM"))
    assert result["diagnosis"] == "sequence_mass_estimate_disagrees"
    assert 89 < result["implied_mw_da"] < 90
    assert conversion_diagnosis(sample(">64 µg/ml", ">718.342 µM", ""))["diagnosis"] == "mass_without_standard_sequence"


def test_unsupported_range():
    assert conversion_diagnosis(sample("0-32 µM", "0 µM"))["diagnosis"] == "non_scalar_or_missing"


def test_aliases_never_strip_prefixes_or_choose_ambiguous():
    index = index_peptides([{"id": "3", "dbaasp_id": "DBAASPR_3", "sequence": "AAA"}])
    assert join_status("DBAASPR_3", index)[0] == "exact_alias_candidate"
    assert join_status("DBAASP_3", index)[0] == "unmatched"
    assert join_status("3", index)[0] == "exact_id"
    index = index_peptides([{"id": "3", "dbaasp_id": "X"}, {"id": "4", "dbaasp_id": "X"}])
    assert join_status("X", index) == ("ambiguous_alias", {})


def test_missing_metadata_not_modified():
    assert metadata_status({})["n_terminus"] == "missing"
    assert metadata_status({"n_terminus": "free"})["n_terminus"] == "recorded_uninterpreted"


def test_end_to_end_source_verification(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    peptides = tmp_path / "peptides.csv"
    write_summary(peptides, [{"id": "1", "sequence": "AKKKKKKK"}])
    write_json(source / "report.json", {"kind": "label_observation_audit_v1",
               "inputs": {str(peptides): sha256(peptides)}})
    for task in ("activity", "hemolysis"):
        row = sample("1 mM", "1 µM")
        row.update(task=task, source_line=2, peptide_id="1", issues="normalized_conversion_disagrees")
        write_summary(source / f"{task}_observations.csv", [row])
    write_json(source / "complete.json", {"files": {p.name: sha256(p) for p in source.iterdir()}})
    before = peptides.read_bytes()
    argv = ["--source", str(source), "--peptides", str(peptides), "--out", str(out)]
    main(argv)
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["activity"]["prior_disagreements"] == {"molar_disagrees": 1}
    assert peptides.read_bytes() == before
    with pytest.raises(ValueError):
        main(argv)
    peptides.write_text("id,sequence\n2,AAAA\n")
    with pytest.raises(ValueError, match="snapshot"):
        main(argv[:-1] + [str(tmp_path / "other")])
