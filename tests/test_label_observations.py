import csv
import json

import pytest
from audit_label_observations import main, measurement, observation


@pytest.mark.parametrize("text,kind", [("49±9.0", "uncertainty"), ("0-32.5", "range"),
                                      (">66", "censored"), ("NA", "missing"), ("10", "exact")])
def test_syntax(text, kind):
    assert measurement(text)["syntax"] == kind


def test_range_truncation_and_missing_sequence():
    row = observation({"assay": "0-32.5 µM", "concentration": "0 µM", "target_organism": "Escherichia coli"}, {}, "activity", 2)
    assert "range_truncated_to_zero" in row["issues"]
    assert row["evidence"] == "ambiguous"


def test_conversion_disagreement_and_unverified_mass():
    row = observation({"assay": ">64 µg/ml", "concentration": ">718.342 µM", "target_organism": "Escherichia coli"},
                      {"sequence": "AKKKKKKK"}, "activity", 2)
    assert "normalized_conversion_disagrees" in row["issues"]
    assert row["evidence"] == "ambiguous"
    assert row["conversion"] == "estimated_mass_unverified_modifications"


@pytest.mark.parametrize("value,target,expected", [("50", "Human erythrocytes", "active"),
    (">66", "Human erythrocytes", "ambiguous"), ("49±9", "Human erythrocytes", "ambiguous"),
    ("50", "CEM-SS cells", "ambiguous")])
def test_hemo(value, target, expected):
    row = observation({"value": value, "unit": "µM", "kind": "50-60% Hemolysis", "target": target},
                      {"sequence": "AKKKKKKK"}, "hemolysis", 2)
    assert row["evidence"] == expected


def test_complete_readonly_run(tmp_path):
    raw, processed, out = [tmp_path / n for n in ("raw", "processed", "out")]
    raw.mkdir()
    processed.mkdir()
    def write(path, row):
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    write(raw / "peptides.csv", {"id": "1", "sequence": "AKKKKKKK"})
    write(raw / "activity.csv", {"peptide_id": "1", "assay": "8 µM", "concentration": "8 µM", "target_organism": "Escherichia coli"})
    write(raw / "hemolysis_raw.csv", {"peptide_id": "1", "kind": "50-60% Hemolysis", "target": "Human erythrocytes", "value": ">66", "unit": "µM"})
    for name in ("activity_labels.csv", "activity_labels_full.csv", "hemolysis_labels.csv"):
        write(processed / name, {"sequence": "AKKKKKKK", "organism": "E. coli", "label": "active"})
    originals = {p: p.read_bytes() for root in (raw, processed) for p in root.iterdir()}
    argv = ["--raw-dir", str(raw), "--processed-dir", str(processed), "--out", str(out)]
    main(argv)
    report = json.loads((out / "report.json").read_text())
    assert report["comparison"]["activity"] == {"ambiguous_only": 1}
    assert report["comparison"]["panel"] == {"agrees": 1}
    assert all(p.read_bytes() == content for p, content in originals.items())
    with pytest.raises(ValueError):
        main(argv)
