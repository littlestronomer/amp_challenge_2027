import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from amp_challenge_2027.conditional_data import (
    concentration_bounds,
    deduplicate_observations,
    fetch_sources,
    import_dbaasp_card,
    import_dramp_rows,
    import_hemolytik_rows,
    molecular_identity,
    parse_measurement,
    peptide_mw,
    prepare_dataset,
    read_xlsx_rows,
    verify_snapshot,
)


def chemistry():
    return {"nTerminus": "H", "cTerminus": "OH", "stereochemistry": "L", "linearCyclic": "linear"}


def card(pid=1, sequence="ACDEFGHIKLMN"):
    return {"id": pid, "sequence": sequence, **chemistry(), "pubmed_id": "12345",
            "targetActivities": [{"targetSpecies": {"name": "Escherichia coli ATCC 25922"},
                                  "activityMeasureGroup": {"name": "MIC"}, "concentration": "<=16",
                                  "unit": {"name": "µM"}, "medium": {"name": "MHB"}}],
            "hemoliticCytotoxicActivities": [{"targetCell": {"name": "human erythrocytes"},
                "activityMeasureForLysisGroup": {"name": "0-10%"}, "concentration": "64", "unit": {"name": "µM"}}]}


def xlsx(path, rows):
    xml = ['<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for index, row in enumerate(rows, 1):
        xml.append(f'<row r="{index}">')
        for column, value in enumerate(row):
            if not value:
                continue
            # These fixtures deliberately use inline strings and sparse cells.
            xml.append(f'<c r="{chr(65 + column)}{index}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml.append('</row>')
    xml.append('</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", "".join(xml))
    return path


@pytest.mark.parametrize("raw,low,high,op,uncertainty", [
    ("1e3", 1000, 1000, "exact", None), ("0.5-2", .5, 2, "range", None),
    ("1e-3–2e-2", .001, .02, "range", None), ("≤ 16", None, 16, "<=", None),
    (">64", 64, None, ">", None), ("16 ± 2", 16, 16, "exact", 2),
    ("4 +/- 1", 4, 4, "exact", 1), ("0.5 to 2", .5, 2, "range", None),
])
def test_bound_parser_preserves_censoring_and_uncertainty(raw, low, high, op, uncertainty):
    parsed = parse_measurement(raw)
    assert (parsed["value_lower"], parsed["value_upper"], parsed["operator"], parsed["uncertainty"]) == (low, high, op, uncertainty)


@pytest.mark.parametrize("raw", ["5 mg/ml extra", "1 or 5", "NaN", "3-1", "1e999", "1e999 ± 1", "1-1e999", None])
def test_ambiguous_values_not_truncated(raw):
    assert parse_measurement(raw)["parse_status"] == "unparsed"


def test_molar_and_mass_units_require_correct_chemistry():
    assert concentration_bounds("1", "nmol/mL")["value_lower"] == 1
    assert concentration_bounds("1", "nmol/L")["value_lower"] == .001
    assert concentration_bounds("1", "mM")["value_lower"] == 1000
    assert concentration_bounds("0.5-2", "µg/ml")["conversion_status"] == "unknown_molecular_identity"
    converted = concentration_bounds("0.5-2", "µg/ml", sequence="ACDEFGHIKLMN", chemistry_compatible=True)
    assert converted["value_lower"] == pytest.approx(.5 * 1000 / peptide_mw("ACDEFGHIKLMN"))
    assert converted["value_upper"] == pytest.approx(2 * 1000 / peptide_mw("ACDEFGHIKLMN"))
    assert converted["operator"] == "range"


def test_missing_chemistry_is_not_unmodified():
    assert not molecular_identity("ACDEFGHIKLMN", {})["competition_compatible"]
    assert molecular_identity("ACDEFGHIKLMN", chemistry())["competition_compatible"]
    assert not molecular_identity("ACDEFGHIKLMN", {**chemistry(), "cTerminus": "NH2"})["competition_compatible"]
    assert not molecular_identity("AcDEFGHIKLMN", chemistry())["competition_compatible"]
    assert not molecular_identity("ACDEFGHIKLMN", {**chemistry(), "intrachainBonds": ["2-4"]})["competition_compatible"]


def test_dbaasp_measurements_preserve_endpoint_dose_and_context():
    molecules, observations = import_dbaasp_card(card())
    assert len(molecules) == 1
    assert len(observations) == 2
    mic, lysis = [row["condition"] for row in observations]
    assert mic["endpoint"] == "MIC"
    assert mic["target"] == "E. coli"
    assert mic["strain"] == "ATCC 25922"
    assert observations[0]["original_target"] == "Escherichia coli ATCC 25922"
    assert mic["medium"] == "MHB"
    assert mic["value_lower"] is None
    assert mic["value_upper"] == 16
    assert lysis["endpoint"] == "%lysis"
    assert lysis["value_lower"] == 0
    assert lysis["value_upper"] == 10
    assert lysis["dose_lower"] == lysis["dose_upper"] == 64
    assert lysis["rbc_species"] == "human erythrocytes"
    assert lysis["publication_id"] == "pmid:12345"
    assert all(row["supervision_eligible"] for row in observations)


def test_unknown_mhc_does_not_become_hc50_or_safety_label():
    raw = card()
    raw["hemoliticCytotoxicActivities"][0]["activityMeasureForLysisGroup"]["name"] = "MHC"
    _, rows = import_dbaasp_card(raw)
    assert rows[1]["condition"]["endpoint"] == "MHC"
    assert not rows[1]["supervision_eligible"]
    raw["hemoliticCytotoxicActivities"][0]["activityMeasureForLysisGroup"]["name"] = "MHC10"
    _, rows = import_dbaasp_card(raw)
    assert rows[1]["condition"]["endpoint"] == "MHC10"


def test_missing_measurements_and_predictions_are_not_training_labels():
    raw = card()
    raw["targetActivities"][0]["evidence_type"] = "predicted"
    raw["hemoliticCytotoxicActivities"][0]["concentration"] = None
    _, rows = import_dbaasp_card(raw)
    assert not any(row["supervision_eligible"] for row in rows)
    assert rows[1]["condition"]["dose_lower"] is None


def test_real_api_reference_is_an_article_ordinal_and_percent_has_suffix():
    raw = card()
    del raw["pubmed_id"]
    raw["articles"] = [{"pubmed": {"pubmedId": "11297740"}}, {"pubmed": {"pubmedId": "18597491"}}]
    raw["targetActivities"][0]["reference"] = "2"
    raw["hemoliticCytotoxicActivities"][0].update(reference="1", activityMeasureForLysisGroup={"name": "0-10% Hemolysis"})
    _, rows = import_dbaasp_card(raw)
    assert rows[0]["condition"]["publication_id"] == "pmid:18597491"
    assert rows[1]["condition"]["publication_id"] == "pmid:11297740"
    assert rows[1]["condition"]["value_upper"] == 10
    raw["targetActivities"][0]["reference"] = "30"
    _, rows = import_dbaasp_card(raw)
    assert rows[0]["condition"]["publication_id"] is None


def test_nonphysical_lysis_and_non_rbc_cells_do_not_become_safety_evidence():
    raw = card()
    raw["hemoliticCytotoxicActivities"][0]["activityMeasureForLysisGroup"]["name"] = "110% Hemolysis"
    _, rows = import_dbaasp_card(raw)
    assert not rows[1]["supervision_eligible"]
    raw["hemoliticCytotoxicActivities"][0]["activityMeasureForLysisGroup"]["name"] = "10% Hemolysis"
    raw["hemoliticCytotoxicActivities"][0]["targetCell"]["name"] = "human fibroblasts"
    _, rows = import_dbaasp_card(raw)
    assert rows[1]["condition"]["rbc_species"] is None


def test_real_dramp_columns_alias_identity_and_distinct_targets():
    raw = {"Sequence": "ACDEFGHIKLMN", "DRAMP_ID": "DRAMP1", "N-terminal_Modification": "Free",
           "C-terminal_Modification": "Free", "Stereochemistry": "L", "Other_Modifications": "None",
           "Linear/Cyclic/Branched": "Linear ", "Pubmed_ID": "12345",
           "Target_Organism": "Escherichia coli ATCC 25922 (MIC=16 µM), Staphylococcus aureus (MIC=2 µM)"}
    molecules, rows = import_dramp_rows([raw])
    assert molecules[0]["identity"]["competition_compatible"]
    assert [row["condition"]["target"] for row in rows] == ["E. coli", "S. aureus"]
    db_molecules, _ = import_dbaasp_card(card())
    assert molecules[0]["molecule_id"] == db_molecules[0]["molecule_id"]


def test_real_hemolytik_csv_headers_and_activity_text():
    raw = {"id": "1001", "pmid": "10660589", "seq": "ALWMTLLKKVLKAAAKAALNAVLVGANA",
           "cter": "Free", "nter": "Free", "lyn_cyc": "Linear", "ldmix": "L", "non_nat": "None",
           "activity": "LC50 = 1.4±0.2 µM", "source": "Human", "non_hem": "NA"}
    molecules, rows = import_hemolytik_rows([raw])
    assert molecules[0]["identity"]["competition_compatible"]
    assert rows[0]["supervision_eligible"]
    assert rows[0]["condition"]["endpoint"] == "LC50"  # Never relabel as HC50.
    assert rows[0]["condition"]["value_lower"] == 1.4
    assert rows[0]["condition"]["uncertainty"] == .2
    assert rows[0]["condition"]["publication_id"] == "pmid:10660589"
    assert rows[0]["condition"]["rbc_species"] == "Human"
    _, qualitative = import_hemolytik_rows([{**raw, "activity": "Non-hemolytic", "non_hem": "Yes"}])
    assert not qualitative[0]["supervision_eligible"]
    assert qualitative[0]["condition"]["value_lower"] is None


def test_dramp_reads_numerical_target_column_and_percent_lysis():
    rows = [{"Sequence": "ACDEFGHIKLMN", "DRAMP_ID": "D1", **chemistry(),
             "Activity": "Antibacterial", "Target_Organism": "E. coli: MIC 1e-1–2 µM",
             "Hemolytic_activity": "human: <=10% hemolysis at 64 µM", "Pubmed_ID": "12345"}]
    _, observations = import_dramp_rows(rows)
    assert len(observations) == 2
    assert observations[0]["condition"]["value_lower"] == .1
    assert observations[0]["condition"]["value_upper"] == 2
    assert observations[1]["condition"]["dose_upper"] == 64
    assert observations[1]["condition"]["value_upper"] == 10


def test_dedup_keeps_conflicts_and_unknown_studies():
    _, rows = import_dbaasp_card(card())
    copy = json.loads(json.dumps(rows[0]))
    copy["source"] = "dramp"
    copy["source_id"] = "D2"
    copy["sources"] = [{"source": "dramp", "source_id": "D2", "ordinal": 0}]
    assert len(deduplicate_observations([rows[0], copy])) == 1
    copy["condition"]["value_upper"] = 64
    assert len(deduplicate_observations([rows[0], copy])) == 2
    copy["condition"]["publication_id"] = None
    rows[0]["condition"]["publication_id"] = None
    copy["condition"]["value_upper"] = 16
    assert len(deduplicate_observations([rows[0], copy])) == 2


def test_distinct_same_study_replicates_in_one_database_remain_separate():
    raw = card()
    raw["targetActivities"].append(dict(raw["targetActivities"][0]))
    _, rows = import_dbaasp_card(raw)
    assert len(deduplicate_observations(rows)) == 3


def test_xlsx_inline_sparse_and_xml_entities(tmp_path):
    path = xlsx(tmp_path / "table.xlsx", [["Sequence", "Unused", "Target_Organism"], ["ACDEFGHIKLMN", "", "E. coli & MIC <=16 µM"]])
    assert read_xlsx_rows(path) == [{"Sequence": "ACDEFGHIKLMN", "Unused": "", "Target_Organism": "E. coli & MIC <=16 µM"}]


def fixture_sources(tmp_path):
    marlys = tmp_path / "corpus.csv"
    marlys.write_text("sequence\nACDEFGHIKLMN\nVYWTSRQPNMLK\nDDDEEEGGGHHH\nWWWWFFFFYYYY\nPPPRRRSSSTTT\nAAAACCCCGGGG\nKKKKLLLLMMMM\nVVVVNNNNQQQQ\nIYIFWTVNDSQA\nGMAVNPWTEDYF\n")
    dramp = xlsx(tmp_path / "dramp.xlsx", [["Sequence", "DRAMP_ID"], ["ACDEFGHIKLMN", "D1"]])
    return marlys, dramp


def test_snapshot_fetch_resume_and_integrity(tmp_path):
    marlys, dramp = fixture_sources(tmp_path)
    out = tmp_path / "snapshot"
    calls = Counter()
    state = {"fail": True}
    def get(url):
        calls[url] += 1
        if "?" in url:
            return json.dumps({"data": [{"id": 1}, {"id": 2}], "total": 2}).encode()
        pid = int(url.rsplit("/", 1)[1])
        if pid == 2 and state["fail"]:
            raise OSError("simulated disconnection")
        return json.dumps(card(pid)).encode()
    with pytest.raises(OSError, match="disconnection"):
        fetch_sources(out, marlys=marlys, dramp=dramp, opener=get, workers=1)
    state["fail"] = False
    manifest = fetch_sources(out, marlys=marlys, dramp=dramp, opener=get, workers=1)
    assert manifest["complete"]
    assert calls["https://dbaasp.org/peptides/1"] == 1
    assert calls["https://dbaasp.org/peptides?limit=200&offset=0"] == 1
    assert not manifest["hemolytik_enabled"]
    assert verify_snapshot(out) == manifest
    (out / "dbaasp/cards/1.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_snapshot(out)


def test_prepare_family_union_missingness_and_immutable_output(tmp_path):
    marlys, dramp = fixture_sources(tmp_path)
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "1.json").write_text(json.dumps(card()))
    modified = card(2)
    modified["cTerminus"] = "NH2"
    (cards / "2.json").write_text(json.dumps(modified))
    snapshot = tmp_path / "snapshot"
    fetch_sources(snapshot, marlys=marlys, dbaasp_dir=cards, dramp=dramp)
    dataset = prepare_dataset(snapshot, tmp_path / "prepared")
    examples = {row["sequence"]: row for row in dataset["examples"]}
    assert set(row["split"] for row in examples.values()) == {"train", "validation", "calibration", "test"}
    assert len(examples["ACDEFGHIKLMN"]["conditions"]) == 2
    assert examples["VYWTSRQPNMLK"]["conditions"] == []
    molecules = [json.loads(line) for line in (tmp_path / "prepared/molecules.jsonl").read_text().splitlines()]
    seq_rows = [row for row in molecules if row["sequence"] == "ACDEFGHIKLMN"]
    assert len({row["family"] for row in seq_rows}) == len({row["split"] for row in seq_rows}) == 1
    assert prepare_dataset(snapshot, tmp_path / "prepared") == dataset
    with pytest.raises(ValueError, match="inputs differ"):
        prepare_dataset(snapshot, tmp_path / "prepared", seed=1)


def test_incomplete_and_optional_sources_require_explicit_flags(tmp_path):
    marlys, dramp = fixture_sources(tmp_path)
    snapshot = tmp_path / "snapshot"
    def get(url):
        return json.dumps({"data": [{"id": 1}], "total": 10} if "?" in url else card()).encode()
    fetch_sources(snapshot, marlys=marlys, dramp=dramp, max_records=1, opener=get)
    with pytest.raises(ValueError, match="incomplete"):
        prepare_dataset(snapshot, tmp_path / "prepared")
    with pytest.raises(ValueError, match="disabled"):
        fetch_sources(tmp_path / "optional", marlys=marlys, dramp=dramp, hemolytik=Path("x.csv"))


def test_explicit_hemolytik_fetch_preserves_archive_and_terms(tmp_path):
    marlys, dramp = fixture_sources(tmp_path)
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "1.json").write_text(json.dumps(card()))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("release/Hemolytik2_complete_data.csv", "id,seq,activity\n1,ACDEFGHIKLMN,Unknown\n")
        archive.writestr("../not_extracted.txt", "must not be extracted")
    def get(url):
        if "api/records" in url:
            return json.dumps({"files": [{"key": "Hemolytik.zip", "links": {"self": "https://example.org/archive.zip"}}]}).encode()
        return buffer.getvalue()
    snapshot = tmp_path / "snapshot"
    manifest = fetch_sources(snapshot, marlys=marlys, dramp=dramp, dbaasp_dir=cards,
                             include_hemolytik=True, opener=get)
    assert manifest["release_terms_unresolved"]
    assert (snapshot / "hemolytik/archive.zip").exists()
    assert (snapshot / "hemolytik/table.csv").read_text().startswith("id,seq")
    assert not (snapshot / "not_extracted.txt").exists()
    assert "GPL-3.0" in next(row for row in manifest["artifacts"] if row["source"] == "hemolytik")["terms"]
