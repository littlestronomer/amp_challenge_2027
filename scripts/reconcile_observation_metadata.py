"""Read-only conversion and ID reconciliation of a completed observation audit."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from audit_label_observations import measurement, rows, split_unit
from build_ranking_labels import RESIDUE_AVG_MW
from compare_top100 import checked_stage
from experiment_utils import code_identity, sha256, write_json, write_summary


def sequence_status(sequence):
    seq = sequence.strip().upper()
    return "missing" if not seq else "nonstandard" if set(seq) - set(RESIDUE_AVG_MW) else "standard"


def index_peptides(peptides):
    index = defaultdict(list)
    for row in peptides:
        # Exact identifier aliases only; never strip prefixes or guess numeric IDs.
        for key in {row.get("id", ""), row.get("dbaasp_id", ""), row.get("dbaaspId", "")} - {""}:
            if row not in index[key]:
                index[key].append(row)
    return index


def join_status(identifier, index):
    candidates = index.get(identifier, [])
    if not candidates:
        return "unmatched", {}
    if len(candidates) != 1:
        return "ambiguous_alias", {}
    return "exact_id" if candidates[0].get("id") == identifier else "exact_alias_candidate", candidates[0]


def conversion_diagnosis(row):
    raw = json.loads(row["raw_row"])
    seq = row["sequence"].strip().upper()
    result = {"diagnosis": "not_activity", "implied_mw_da": "", "sequence_mw_da": "",
              "expected_um": "", "relative_difference": "", "operator_changed": False}
    if row["task"] != "activity":
        return result
    value, unit = split_unit(raw.get("assay", ""))
    nv, nu = split_unit(raw.get("concentration", ""))
    original, normalized = measurement(value), measurement(nv)
    if original["syntax"] not in ("exact", "censored") or normalized["syntax"] not in ("exact", "censored"):
        result["diagnosis"] = "non_scalar_or_missing"
        return result
    factors = {"µM": 1., "μM": 1., "uM": 1., "nM": .001, "mM": 1000.}
    if nu not in factors:
        result["diagnosis"] = "normalized_unit_unknown"
        return result
    actual = normalized["value"] * factors[nu]
    if not math.isfinite(actual) or actual <= 0:
        result["diagnosis"] = "nonfinite_conversion"
        return result
    result["operator_changed"] = original["operator"] != normalized["operator"]
    if unit in factors:
        expected = original["value"] * factors[unit]
        prefix = "molar"
    elif unit.replace("µ", "u").replace("μ", "u").lower() == "ug/ml":
        implied = original["value"] * 1000. / actual
        if not math.isfinite(implied):
            result["diagnosis"] = "nonfinite_conversion"
            return result
        result["implied_mw_da"] = implied
        if sequence_status(seq) != "standard":
            result["diagnosis"] = "mass_without_standard_sequence"
            return result
        mw = sum(RESIDUE_AVG_MW[aa] for aa in seq) + 18.01528
        result["sequence_mw_da"] = mw
        expected = original["value"] * 1000. / mw
        prefix = "sequence_mass_estimate"
    else:
        result["diagnosis"] = "original_unit_unknown"
        return result
    if not all(math.isfinite(v) and v > 0 for v in (actual, expected)):
        result["diagnosis"] = "nonfinite_conversion"
        return result
    result.update(expected_um=expected, relative_difference=abs(actual - expected) / expected)
    result["diagnosis"] = prefix + ("_consistent" if math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-6) else "_disagrees")
    return result


def metadata_status(peptide):
    # Inventory only: nonempty metadata is not a declaration of an unmodified molecule.
    return {field: "missing" if not str(peptide.get(field, "")).strip() else "recorded_uninterpreted"
            for field in ("n_terminus", "c_terminus", "complexity")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("sweep_results/label-observations-v1"))
    parser.add_argument("--peptides", type=Path, default=Path("data/raw/dbaasp/peptides.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out, source = args.out.resolve(), args.source.resolve()
    if out.exists() or out == source or out in source.parents or source in out.parents:
        raise ValueError("Use a new output directory separate from the source audit")
    marker = checked_stage(source, "complete.json", {"report.json", "activity_observations.csv", "hemolysis_observations.csv"})
    previous = json.loads((source / "report.json").read_text())
    if previous.get("kind") != "label_observation_audit_v1":
        raise ValueError("Unsupported source audit")
    digest = sha256(args.peptides)
    expected = [v for k, v in previous["inputs"].items() if Path(k).name == "peptides.csv"]
    if expected != [digest]:
        raise ValueError("Peptide snapshot differs from source audit; no cross-snapshot joins allowed")
    index = index_peptides(rows(args.peptides, ["id", "sequence"]))
    summaries, reports = {}, {}
    for task in ("activity", "hemolysis"):
        output = []
        for row in rows(source / f"{task}_observations.csv", ["task", "source_line", "peptide_id", "sequence", "raw_row", "issues"]):
            join, peptide = join_status(row["peptide_id"], index)
            converted = conversion_diagnosis(row)
            metadata = metadata_status(peptide)
            output.append({"source_line": row["source_line"], "peptide_id": row["peptide_id"],
                           "join": join, "original_sequence_status": sequence_status(row["sequence"]),
                           "candidate_sequence_status": sequence_status(peptide.get("sequence", "")),
                           "candidate_id": peptide.get("id", ""),
                           "candidate_sequence": peptide.get("sequence", ""),
                           "candidate_metadata": json.dumps(peptide, sort_keys=True),
                           "metadata_coverage": json.dumps(metadata, sort_keys=True),
                           "prior_conversion_disagreement": "normalized_conversion_disagrees" in row["issues"].split("|"),
                           **converted})
        reports[task] = output
        summaries[task] = {key: dict(Counter(str(r[key]) for r in output)) for key in
                           ("join", "diagnosis", "original_sequence_status", "operator_changed")}
        summaries[task]["prior_disagreements"] = dict(Counter(r["diagnosis"] for r in output if r["prior_conversion_disagreement"]))
        summaries[task]["metadata_coverage"] = {
            field: dict(Counter(json.loads(r["metadata_coverage"])[field] for r in output))
            for field in ("n_terminus", "c_terminus", "complexity")}
    if sha256(args.peptides) != digest or checked_stage(source, "complete.json", set(marker["files"])) != marker:
        raise ValueError("Inputs changed during reconciliation")
    out.mkdir(parents=True, exist_ok=False)
    for task, output in reports.items():
        write_summary(out / f"{task}_reconciliation.csv", output)
    write_json(out / "report.json", {"kind": "observation_reconciliation_v1", "source": str(source),
               "source_marker": marker, "peptides_sha256": digest, "code": code_identity(), "summary": summaries,
               "limitations": ["No sequences, metadata or labels automatically replaced",
                               "Exact aliases are candidates, not verified molecular equivalence",
                               "Mass agreement does not verify modifications or measured molecular weight",
                               "Missing termini stay unknown; no external metadata fetched",
                               "Reference/donor/strain conflicts remain unresolved"]})
    write_json(out / "complete.json", {"files": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
