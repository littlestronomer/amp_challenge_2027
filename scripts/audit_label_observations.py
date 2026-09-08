"""Versioned observation audit, NOT a training-label builder. No inputs are modified."""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_ranking_labels import RESIDUE_AVG_MW
from experiment_utils import code_identity, sha256, write_json, write_summary

from amp_challenge_2027.data import SPECIES_TO_PANEL
from amp_challenge_2027.measurement_bounds import Bound, activity_label, parse_bound

NUMBER = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


def measurement(value):
    """Recognize syntax without assigning semantics to ranges or ± errors."""
    text = value.strip()
    try:
        bound = parse_bound(text)
        return {"syntax": "exact" if bound.operator == "=" else "censored",
                "operator": bound.operator, "value": bound.value}
    except ValueError:
        pass
    for pattern, category in [(rf"({NUMBER})\s*(?:±|\+/-)\s*({NUMBER})", "uncertainty"),
                              (rf"({NUMBER})\s*[-–]\s*({NUMBER})", "range")]:
        match = re.fullmatch(pattern, text)
        if match:
            a, b = map(float, match.groups())
            if not all(math.isfinite(v) for v in (a, b)):
                break
            return {"syntax": category, "first": a, "second": b}
    return {"syntax": "missing" if text.upper() in ("", "NA", "N/A", "-") else "unsupported"}


def split_unit(text):
    match = re.fullmatch(r"\s*(.+?)\s+(µM|μM|uM|nM|mM|µg/ml|μg/ml|ug/ml|µg/mL|μg/mL|ug/mL)\s*", text)
    return (match[1], match[2]) if match else (text, "")


def convert(parsed, unit, peptide):
    """Approximate mass conversion only for known standard unmodified sequences."""
    if parsed["syntax"] not in ("exact", "censored"):
        return None, "non_scalar"
    factors = {"µM": 1., "μM": 1., "uM": 1., "nM": .001, "mM": 1000.}
    if unit in factors:
        return Bound(parsed["operator"], parsed["value"] * factors[unit]), "molar"
    if unit.replace("μ", "u").replace("µ", "u").lower() != "ug/ml":
        return None, "unknown_unit"
    seq = peptide.get("sequence", "").strip().upper()
    if not seq or set(seq) - set(RESIDUE_AVG_MW):
        return None, "missing_or_nonstandard_sequence"
    # Missing modification metadata is not evidence of an unmodified molecule.
    termini = (peptide.get("n_terminus"), peptide.get("c_terminus"))
    mw = sum(RESIDUE_AVG_MW[aa] for aa in seq) + 18.01528
    result = Bound(parsed["operator"], parsed["value"] * 1000. / mw)
    if termini != ("free", "free") or peptide.get("complexity", "").lower() not in ("", "linear"):
        return result, "estimated_mass_unverified_modifications"
    return result, "estimated_mass"


def rows(path, required):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {required}; found {reader.fieldnames}")
        return list(reader)


def observation(row, peptide, task, line):
    issues = []
    raw_json = {}
    if row.get("raw"):
        try:
            raw_json = json.loads(row["raw"])
            if not isinstance(raw_json, dict):
                raise ValueError("not an object")
        except (ValueError, TypeError):
            issues.append("invalid_raw_json")
            raw_json = {}
    if task == "activity":
        original = row.get("assay", "")
        value, unit = split_unit(original)
        if not original.strip():
            issues.append("original_measurement_missing")
        target = row.get("target_organism", "")
    else:
        value, unit = row.get("value", ""), row.get("unit", "")
        target = row.get("target", "")
    parsed = measurement(value)
    bound, conversion = convert(parsed, unit, peptide)
    sequence = peptide.get("sequence", "").strip().upper()
    if not sequence:
        issues.append("sequence_missing")
    elif set(sequence) - set(RESIDUE_AVG_MW):
        issues.append("nonstandard_sequence")
    genus = next((g for k, g in SPECIES_TO_PANEL.items() if k.lower() in target.lower()), "")
    eligible = bool(genus) if task == "activity" else bool(re.search(r"erythrocyt|\bRBC\b", target, re.I))
    evidence = "ambiguous"
    if bound and conversion in ("molar", "estimated_mass") and eligible and sequence and not set(sequence) - set(RESIDUE_AVG_MW):
        if task == "activity":
            evidence = activity_label(bound)
        else:
            # Require an explicit hemolysis percentage. Do not equate IC50/MHC
            # or cytotoxicity endpoints with a percent-lysis observation.
            kind = row.get("kind", "")
            band = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*%\s*Hemolysis", kind, re.I)
            if band:
                lo, hi = float(band[1]), float(band[2] or band[1])
                if 0 <= lo <= hi <= 100:
                    if lo >= 40 and bound.definitely_le(128):
                        evidence = "active"
                    elif hi <= 30 and bound.definitely_ge(128):
                        evidence = "inactive"
            else:
                issues.append("endpoint_not_explicit_hemolysis_band")
    if not eligible:
        issues.append("outside_task_targets")
    if parsed["syntax"] == "missing" and row.get("note"):
        issues.append("qualitative_note_only")
    if task == "activity":
        normal_value, normal_unit = split_unit(row.get("concentration", ""))
        normalized = measurement(normal_value)
        normalized_bound, _ = convert(normalized, normal_unit, peptide)
        if parsed["syntax"] in ("range", "uncertainty") and normalized["syntax"] in ("exact", "censored"):
            issues.append("upstream_structure_loss")
        if parsed["syntax"] == "range" and normal_value.strip() == "0":
            issues.append("range_truncated_to_zero")
        if bound and normalized_bound and (bound.operator != normalized_bound.operator or
                not math.isclose(bound.value, normalized_bound.value, rel_tol=1e-4, abs_tol=1e-6)):
            issues.append("normalized_conversion_disagrees")
    return {"task": task, "source_line": line, "peptide_id": row.get("peptide_id", ""),
            "sequence": sequence, "target": target, "panel_genus": genus,
            "eligible_target": eligible, "measurement": json.dumps(parsed, sort_keys=True),
            "syntax": parsed["syntax"], "unit": unit, "conversion": conversion,
            "estimated_um": bound.value if bound else "", "operator": bound.operator if bound else "",
            "evidence": evidence, "issues": "|".join(issues),
            "reference_local_to_peptide": str(raw_json.get("reference", "")),
            "raw_observation_id": str(raw_json.get("id", "")),
            "raw_row": json.dumps(row, sort_keys=True), "peptide_metadata": json.dumps(peptide, sort_keys=True)}


def summarize(observations):
    return {key: dict(Counter(str(row[key]) for row in observations))
            for key in ("syntax", "conversion", "evidence", "eligible_target")} | {
        "issues": dict(Counter(issue for row in observations for issue in row["issues"].split("|") if issue)),
        "rows": len(observations)}


def comparisons(observations, path, task):
    """Evidence comparison only: disagreement is not proof the old label is wrong."""
    by_key = defaultdict(set)
    for row in observations:
        if row["eligible_target"] and row["sequence"]:
            key = (row["sequence"], row["panel_genus"] if task == "panel" else "")
            by_key[key].add(row["evidence"])
    output = []
    for line, row in enumerate(rows(path, ["sequence", "label"]), 2):
        genus = row.get("organism", row.get("genus", row.get("target_organism", ""))) if task == "panel" else ""
        if task == "panel" and not genus:
            raise ValueError(f"{path}: panel comparison needs organism/genus/target_organism")
        evidence = by_key.get((row["sequence"].strip().upper(), genus), set())
        definite = evidence & {"active", "inactive"}
        status = ("no_matched_evidence" if not evidence else "conflicting_evidence" if len(definite) > 1 else
                  "ambiguous_only" if not definite else "mixed_with_ambiguous" if "ambiguous" in evidence else
                  "agrees" if row["label"] in definite else "disagrees")
        output.append({"task": task, "source_line": line, "sequence": row["sequence"], "genus": genus,
                       "old_label": row["label"], "status": status, "observed_evidence": "|".join(sorted(evidence))})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/dbaasp"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = [args.raw_dir / name for name in ("peptides.csv", "activity.csv", "hemolysis_raw.csv")]
    sources += [args.processed_dir / name for name in ("activity_labels.csv", "activity_labels_full.csv", "hemolysis_labels.csv")]
    if args.out.exists() or any(args.out.resolve() == p.resolve() or args.out.resolve() in p.resolve().parents for p in sources):
        raise ValueError("Output must be a new directory separate from inputs")
    hashes = {str(p): sha256(p) for p in sources}
    peptides = {}
    for peptide in rows(sources[0], ["id", "sequence"]):
        key = peptide["id"]
        if key in peptides and peptides[key] != peptide:
            raise ValueError(f"Conflicting peptide metadata: {key}")
        peptides[key] = peptide
    datasets = {}
    for task, path, fields in [("activity", sources[1], ["peptide_id", "assay", "concentration", "target_organism"]),
                               ("hemolysis", sources[2], ["peptide_id", "kind", "target", "value", "unit"])]:
        datasets[task] = [observation(row, peptides.get(row["peptide_id"], {}), task, line)
                          for line, row in enumerate(rows(path, fields), 2)]
    compared = []
    for task, path, obs in [("activity", sources[3], datasets["activity"]), ("panel", sources[4], datasets["activity"]),
                            ("hemolysis", sources[5], datasets["hemolysis"])]:
        compared.extend(comparisons(obs, path, task))
    # The activity binary classifier historically uses a different positive cutoff.
    # Recompute that comparison at 4 µM, without changing panel evidence.
    binary = []
    for row in datasets["activity"]:
        copied = dict(row)
        if copied["evidence"] != "ambiguous":
            copied["evidence"] = activity_label(Bound(row["operator"], row["estimated_um"]), success=4)
        binary.append(copied)
    compared = [r for r in compared if r["task"] != "activity"] + comparisons(binary, sources[3], "activity")
    if hashes != {str(p): sha256(p) for p in sources}:
        raise ValueError("Inputs changed during audit; no report written")
    args.out.mkdir(parents=True, exist_ok=False)
    for task, observations in datasets.items():
        write_summary(args.out / f"{task}_observations.csv", observations)
    write_summary(args.out / "label_comparison.csv", compared)
    report = {"kind": "label_observation_audit_v1", "inputs": hashes, "code": code_identity(),
              "tasks": {task: summarize(obs) for task, obs in datasets.items()},
              "comparison": {task: dict(Counter(r["status"] for r in compared if r["task"] == task))
                             for task in ("activity", "panel", "hemolysis")},
              "limitations": ["Diagnostic evidence only, not replacement labels or validated generalization",
                              "Ranges and uncertainty masked; ± is not a guaranteed interval",
                              "Mass conversions with unknown modifications masked; estimates retained for audit",
                              "Reference identifiers are peptide-local, not globally unique study IDs",
                              "No qualitative note is promoted to a safety label",
                              "Binary comparisons pool genera and do not reproduce historical aggregation",
                              "Existing test remains inspected; no new holdout created"]}
    write_json(args.out / "report.json", report)
    write_json(args.out / "complete.json", {"files": {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}})
    print(json.dumps({"tasks": report["tasks"], "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
