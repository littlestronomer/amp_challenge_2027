"""Collect local training evidence on CPU without loading models or rebuilding data.

Snapshot matches, matching model copies and reconstructed splits never establish
historical training lineage automatically. Outputs contain hashes and summaries,
not peptide rows, raw JSON/log contents, credentials, or model tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from audit_authorship_readiness import source_inventory
from experiment_utils import REPO_ROOT, mark_files, prepare_run, sha256, write_json


def local_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside repository: {name}")
    return path


def file_record(root: Path, name: str, expected: str | None = None) -> dict:
    path = local_path(root, name)
    result = {"path": name, "status": "missing", "expected_sha256": expected}
    if path.is_file():
        digest = sha256(path)
        result.update(sha256=digest, bytes=path.stat().st_size,
                      status="match" if digest == expected else "mismatch" if expected else "present")
    return result


def csv_summary(path: Path) -> dict:
    counts: dict[str, Counter] = {}
    sequences = set()
    total = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for column in ("source_db", "source_dbs"):
            if column in columns:
                counts[column] = Counter()
        for row in reader:
            total += 1
            if row.get("sequence"):
                sequences.add(row["sequence"].strip().upper())
            for column, counter in counts.items():
                for value in (row.get(column) or "").split("|"):
                    if value.strip():
                        counter[value.strip()] += 1
    return {"columns": columns, "rows": total,
            "unique_sequences": len(sequences) if "sequence" in columns else None,
            "sequence_set_sha256": sequence_set_hash(sequences) if "sequence" in columns else None,
            "declared_source_counts": {k: dict(sorted(v.items())) for k, v in counts.items()}}


def sequence_set_hash(sequences) -> str:
    """Canonical uppercase unique-sequence set, not a raw-file checksum."""
    return hashlib.sha256("\n".join(sorted(set(sequences))).encode()).hexdigest()


def fasta_set_hash(path: Path) -> str:
    sequences, parts = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
            parts = []
        elif line.strip():
            parts.append(line.strip().upper())
    if parts:
        sequences.append("".join(parts))
    return sequence_set_hash(sequences)


def collect(root: Path, out: Path, ledger: dict, baseline: dict) -> dict:
    root, out = root.resolve(), out.resolve()
    # Protect all evidence roots, including nested outputs that a later scan
    # might otherwise mistake for original training evidence.
    for name in ("data", "checkpoint", "runs", "logs", "src", "docs"):
        if out == root or out.is_relative_to(root / name):
            raise ValueError("Place output outside input/evidence directories")
    datasets = {}
    models = []
    reference = local_path(root, "data/antibacterial.fasta")
    reference_set = fasta_set_hash(reference) if reference.is_file() else None
    for item in ledger["models"]:
        artifact = file_record(root, item["artifact"], item["artifact_sha256"])
        data = file_record(root, item["candidate_data"], item["candidate_data_sha256"])
        if data["status"] != "missing":
            data["summary"] = csv_summary(local_path(root, item["candidate_data"]))
            candidate_set = data["summary"]["sequence_set_sha256"]
            data["summary"]["same_sequence_set_as_reference"] = (
                candidate_set == reference_set if candidate_set is not None and reference_set is not None else None
            )
        datasets[item["candidate_data"]] = data
        models.append({"id": item["id"], "artifact": artifact, "candidate_data": data,
                       "historical_training_provenance_verified": False,
                       "declaration_status": item["training_link_status"]})

    # Inventory source inputs and training manifests without copying contents.
    records = {}
    for base in ("data/raw", "data/processed", "checkpoint", "runs", "logs"):
        directory = root / base
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            is_data = base.startswith("data/") and path.suffix in {".csv", ".fasta", ".fa", ".json"}
            is_record = path.suffix == ".json" and any(
                term in path.name.lower() for term in ("manifest", "split", "ensemble", "config")
            )
            if is_data or is_record:
                name = str(path.relative_to(root))
                records[name] = file_record(root, name)

    # Original member copies can strengthen artifact identity only. Read bytes,
    # never unpickle checkpoints. Different serializations may not match.
    sizes = {local_path(root, m["artifact"]["path"]).stat().st_size
             for m in models if m["artifact"]["status"] != "missing"}
    copies = []
    if (root / "checkpoint").is_dir():
        for path in sorted((root / "checkpoint").rglob("*.pt")):
            if (path.name not in {"model.pt", "classifier.pt", "classifier_panel.pt"}
                    or not path.resolve().is_relative_to(root) or path.stat().st_size not in sizes):
                continue
            name = str(path.relative_to(root))
            digest = sha256(path)
            matches = [m["id"] for m in models if m["artifact"].get("sha256") == digest
                       and m["artifact"]["path"] != name]
            if matches:
                copies.append({"path": name, "sha256": digest, "matches_deployed": matches,
                               "training_data_link_verified": False})

    runtime = [file_record(root, name, digest) for name, digest in baseline["runtime_files"].items()]
    release_outputs = [file_record(root, f"submission/release-default-v1/generate/{name}", digest)
                       for name, digest in baseline["outputs"].items()]
    report = {
        "kind": "training_lineage_collection_v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "historical_training_provenance_verified": False,
        "models": models, "datasets": list(datasets.values()), "records": list(records.values()),
        "reference_sequence_set_sha256": reference_set,
        "byte_identical_model_copies": copies, "download_sources": source_inventory(root),
        "baseline_commit": baseline["validated_commit"], "runtime_files": runtime,
        "runtime_matches_validated_baseline": all(r["status"] == "match" for r in runtime),
        "fresh_clone_outputs": release_outputs,
        "limitations": [
            "Current CSV hashes are compared to reported/pinned candidate snapshots, not original training attestations.",
            "CSV source columns are declarations and may omit upstream database membership.",
            "Matching model copies do not bind a data snapshot or split to training.",
            "Original logs/manifests need review; reconstructed splits are not original evidence.",
            "No model loading, inference, downloads, training, or public upload performed.",
        ],
    }
    prepare_run(out, report)
    write_json(out / "lineage.json", report)
    lines = ["# Training lineage evidence", "", f"Commit: `{report['commit']}`.", "",
             "Historical snapshot-to-checkpoint linkage: **not certified**.", "",
             "| Model | Weight vs release | Candidate CSV vs recorded snapshot |",
             "|---|---|---|"]
    lines += [f"| {m['id']} | {m['artifact']['status']} | {m['candidate_data']['status']} |" for m in models]
    lines += ["", "## Dataset summaries", ""]
    for record in datasets.values():
        lines.append(f"- `{record['path']}`: {record['status']}; SHA-256 `{record.get('sha256', 'unavailable')}`")
        if "summary" in record:
            lines.append(f"  - {json.dumps(record['summary'], sort_keys=True)}")
    lines += ["", "## Original artifact copies", ""]
    lines += [f"- `{r['path']}`: byte match to {', '.join(r['matches_deployed'])}" for r in copies]
    if not copies:
        lines.append("- No byte-identical copies found in checkpoint/. Tensor equivalence was not tested.")
    lines += ["", "## Baseline checks", "",
              f"- Runtime files match validated baseline: {report['runtime_matches_validated_baseline']}"]
    lines += [f"- `{r['path']}`: {r['status']}" for r in release_outputs]
    lines += ["", "## Records available for review", ""]
    lines += [f"- `{r['path']}`: SHA-256 `{r['sha256']}`" for r in records.values()]
    lines += ["", "## Limits", ""] + [f"- {line}" for line in report["limitations"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    mark_files(out, "complete.json", ["run.json", "lineage.json", "REPORT.md"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release = args.root / "docs/release"
    result = collect(args.root, args.out,
                     json.loads((release / "TRAINING_LINEAGE.json").read_text()),
                     json.loads((release / "BASELINE.json").read_text()))
    print(f"Lineage evidence: {args.out / 'REPORT.md'}")
    print(f"Runtime matches validated baseline: {result['runtime_matches_validated_baseline']}")
    print("Historical training lineage remains unverified; inspect original records before sign-off.")


if __name__ == "__main__":
    main()
