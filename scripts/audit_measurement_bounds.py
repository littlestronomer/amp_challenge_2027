"""Read-only screen of raw concentration fields; not a replacement label builder."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from experiment_utils import sha256, write_json

from amp_challenge_2027.measurement_bounds import Bound, activity_label, parse_bound


def audit_file(path, column):
    counts, findings = Counter(), []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing {column}; available: {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            raw = row[column].strip()
            # Only normalized micromolar strings can be threshold-classified.
            match = re.fullmatch(r"(.+?)\s+(?:µM|μM|uM)", raw)
            numeric = match[1] if match else raw
            try:
                bound = parse_bound(numeric)
                status = "exact" if bound.operator == "=" else "censored"
                counts[status] += 1
                detail = {"line": line, "status": status, "raw_row": row}
                if match:
                    detail["mic_label_if_exact"] = activity_label(Bound("=", bound.value))
                    detail["mic_bound_label"] = activity_label(bound)
                    counts["mic_threshold_label_changes"] += int(detail["mic_label_if_exact"] != detail["mic_bound_label"])
                if status != "exact":
                    findings.append(detail)
            except ValueError:
                counts["unsupported"] += 1
                findings.append({"line": line, "status": "unsupported", "raw_row": row})
    return {"path": str(path), "sha256": sha256(path), "counts": dict(counts), "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--column", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new output directory")
    report = audit_file(args.input, args.column)
    report["limitations"] = ["No datasets modified", "No sequence aggregation or conflict resolution",
                            "MIC threshold comparison applies only to explicit micromolar strings",
                            "Hemolysis requires target and lysis-band interpretation; no safety labels inferred",
                            "Already discarded operators cannot be reconstructed"]
    write_json(args.out / "report.json", report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
