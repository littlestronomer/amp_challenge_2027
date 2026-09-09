from build_molar_candidates import build, decision


def test_full_build_preserves_inputs_and_reports_unmapped(tmp_path):
    import json

    import pytest
    from build_molar_candidates import main
    from experiment_utils import sha256, write_json, write_summary

    source, prepared, out = [tmp_path / p for p in ("source", "prepared", "out")]
    source.mkdir()
    prepared.mkdir()
    write_json(source / "report.json", {"kind": "label_observation_audit_v1"})
    for task in ("activity", "hemolysis"):
        write_summary(source / f"{task}_observations.csv", [row(), row(sequence="AKKKKKKA")])
    write_summary(source / "label_comparison.csv", [{"task": task, "sequence": "AKKKKKKK", "genus": "E. coli" if task == "panel" else "", "old_label": "active"} for task in ("activity", "panel", "hemolysis")])
    write_summary(prepared / "families.csv", [{"sequence": "AKKKKKKK", "family": "f1", "split": "train"}])
    for root in (source, prepared):
        write_json(root / "complete.json", {"files": {p.name: sha256(p) for p in root.iterdir()}})
    before = {p: sha256(p) for root in (source, prepared) for p in root.iterdir()}
    argv = ["--source", str(source), "--prepared", str(prepared), "--out", str(out)]
    main(argv)
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["panel"]["coverage"]["unmapped"]["rows"] == 1
    assert report["summary"]["panel"]["common_unambiguous_old_keys"] == 1
    assert all(sha256(p) == digest for p, digest in before.items())
    with pytest.raises(ValueError):
        main(argv)


def row(value=2, label="active", **kwargs):
    return {"sequence": "AKKKKKKK", "source_line": 2, "panel_genus": "E. coli", "target": "E. coli strain1",
            "eligible_target": "True", "syntax": "exact", "conversion": "molar", "operator": "=",
            "estimated_um": value, "evidence": label, **kwargs}


def test_task_thresholds():
    assert decision(row(8), "activity")[0] == "ambiguous"
    assert decision(row(8), "panel")[0] == "active"


def test_conflicts_and_ambiguous_mask():
    assert not build([row(2), row(64)], "panel")[0]
    assert not build([row(2), row(24)], "panel")[0]
    assert build([row(2), row(2)], "panel")[0] == [{"sequence": "AKKKKKKK", "organism": "E. coli", "label": "active"}]


def test_exclusions_and_provenance():
    candidates, groups, events = build([row(), row(conversion="estimated_mass")], "panel")
    assert len(candidates) == 1
    assert events[1]["reason"] == "not_direct_molar"
    assert groups[0]["targets"] == '["E. coli strain1"]'
    assert decision(row(sequence="AXKKKKKK"), "activity")[1] == "invalid_sequence"
    assert decision(row(syntax="range"), "activity")[1] == "non_scalar_or_missing"


def test_hemolysis_uses_risk_direction():
    assert build([row(label="inactive")], "hemolysis")[0][0]["label"] == "inactive"
    assert not build([row(label="inactive"), row(label="active")], "hemolysis")[0]
