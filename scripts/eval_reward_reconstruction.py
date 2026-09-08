"""Hash-pinned, explicitly provisional validation reconstruction for deployed heads.

No original split manifests were recovered. Results never certify historical
validation, a new holdout, or biological safety. No temperature fitting occurs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import compare_top100 as baseline
import numpy as np
from audit_reward_artifacts import metadata_issue, tensor_fingerprint
from experiment_utils import (
    REPO_ROOT,
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from selection_cache import separate_output
from train_reward_classifier import load_activity_data, load_panel_data

from amp_challenge_2027.config import AMINO_ACIDS
from amp_challenge_2027.props import compute_properties
from amp_challenge_2027.reward_audit import (
    cross_split_similarity,
    prediction_report,
    reconstruct_split,
    sigmoid,
)

DEFAULT_PROTOCOL = REPO_ROOT / "experiments/reward_reconstruction_v1.json"
OUTPUT_FILES = ["split_assignments.csv", "split_audit.json", "validation_predictions.csv",
                "predictions.npz", "backbone.json", "metrics.json", "per_output_metrics.csv", "reliability_bins.csv"]


def resolve(path: str | Path) -> Path:
    return (REPO_ROOT / path).resolve()


def verify_inputs(inputs: dict[str, str]) -> None:
    for filename, expected in inputs.items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned input changed or missing: {filename}")


def prepare_inputs(args) -> tuple[dict, list[dict]]:
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get("kind") != "reward_reconstruction_protocol_v1"
            or protocol.get("historical_validation_verified") is not False or not protocol.get("assumptions")
            or set(protocol.get("tasks", {})) != {"activity", "panel", "hemolysis"}):
        raise ValueError("Require an explicitly unverified reconstruction protocol for the three heads")
    audit = args.artifact_audit.resolve()
    baseline.checked_stage(audit, "complete.json", {"inventory.json"})
    inventory = json.loads((audit / "inventory.json").read_text())
    audit_run = json.loads((audit / "run.json").read_text())
    if audit_run.get("kind") != "reward_artifact_inventory_v1":
        raise ValueError("Unexpected artifact audit kind")
    source = args.source.resolve()
    source_run = json.loads((source / "run.json").read_text())
    if source_run.get("kind") != "paired_top100_v1":
        raise ValueError("Expected the original top-100 scoring manifest")
    if audit_run["inputs"].get(str(source / "run.json")) != sha256(source / "run.json"):
        raise ValueError("Artifact audit and scoring source differ")
    baseline.checked_stage(source, "reference_cache.json", {"backbones.json", "reference_embeddings.npy"})
    revisions = json.loads((source / "backbones.json").read_text())
    paths = [args.protocol.resolve(), audit / "run.json", audit / "complete.json", audit / "inventory.json",
             source / "run.json", source / "backbones.json", source / "reference_cache.json"]
    expected_files = {str(p): sha256(p) for p in paths}
    protected_roots = [audit, source]
    cells = []
    for name in args.tasks:
        spec = protocol["tasks"][name]
        item = inventory["artifacts"][name]
        if item["issues"] or item["status"] == "invalid_artifact":
            raise ValueError(f"Artifact audit failed: {name}")
        if item["files"] != source_run["classifiers"][name]["files"]:
            raise ValueError(f"Artifact audit and original scorer hashes differ: {name}")
        root, stem = Path(item["directory"]), item["stem"]
        cfg_path, head = root / f"{stem}_config.json", root / f"{stem}.pt"
        expected_files.update({str(root / key): value for key, value in item["files"].items()})
        verify_inputs({str(root / key): value for key, value in item["files"].items()})
        cfg = json.loads(cfg_path.read_text())
        if (metadata_issue(cfg, spec["task"]) or cfg["temperature"] != spec["temperature"]
                or sha256(head) != spec["head_sha256"]):
            raise ValueError(f"Deployed head/config differs from the pinned protocol: {name}")
        training = resolve(spec["training_run"])
        member_path = training / f"member{spec['member']}" / f"{stem}.pt"
        matches = [m for m in item["matches"] if Path(m["path"]).resolve() == member_path and m["tensor_identical"]]
        if len(matches) != 1 or sha256(member_path) != matches[0]["sha256"]:
            raise ValueError(f"Expected matched member is absent or changed: {name}")
        expected_files[str(member_path)] = matches[0]["sha256"]
        training_config, members_path = training / "config.json", training / "members.json"
        for path in (training_config, members_path):
            if inventory["records"].get(str(path)) != sha256(path):
                raise ValueError(f"Training metadata changed since artifact audit: {path}")
            expected_files[str(path)] = inventory["records"][str(path)]
        training_cfg = json.loads(training_config.read_text())
        if (metadata_issue(training_cfg, spec["task"]) or training_cfg["temperature"] != spec["temperature"]
                or training_cfg.get("val_auroc") != spec["recorded_val_auroc"]):
            raise ValueError(f"Frozen training config disagrees with protocol: {name}")
        members = json.loads(members_path.read_text())
        selected = [m for m in members if m["member"] == spec["member"]]
        if len(selected) != 1 or any(selected[0].get(k) != spec[v] for k, v in (
                ("seed", "seed"), ("temperature", "temperature"), ("val_auroc", "recorded_val_auroc"))):
            raise ValueError(f"Matched member metadata disagrees with protocol: {name}")
        path = resolve(spec["data"])
        if sha256(path) != spec["data_sha256"]:
            raise ValueError(f"Label CSV hash differs from the pinned user-provided snapshot: {path}")
        expected_files[str(path)] = spec["data_sha256"]
        panel = spec["task"] == "panel"
        records = load_panel_data(path) if panel else load_activity_data(path)
        train, val = reconstruct_split(records, panel=panel, seed=spec["seed"])
        targets = np.stack([r["targets"] for r in records]) if panel else np.array([[r["label"]] for r in records], dtype=np.float32)
        masks = np.stack([r["mask"] for r in records]) if panel else np.ones_like(targets)
        outputs = list(baseline.PANEL_GENERA) if panel else [spec["positive_label"]]
        revision = revisions.get(name)
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"Missing recorded scoring backbone revision: {name}")
        paths.extend([path, head, cfg_path, member_path, training_config, members_path])
        protected_roots.extend([root, training])
        cells.append({"name": name, "spec": spec, "artifact": item, "config": cfg, "data": path,
                      "records": records, "train": train, "val": val, "targets": targets, "masks": masks,
                      "outputs": outputs, "revision": revision})
    separate_output(args.out, [*protected_roots, *paths])
    verify_inputs(expected_files)
    return {"protocol": protocol, "files": expected_files}, cells


class PinnedPredictor:
    """Production forward architecture, explicit complete head, pinned scoring backbone."""

    def __init__(self, cell: dict, *, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from amp_challenge_2027.score import _build_activity_module
        from amp_challenge_2027.training import enable_determinism

        enable_determinism(42)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        outputs = len(cell["outputs"])
        path = Path(cell["artifact"]["directory"]) / f"{cell['artifact']['stem']}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        baseline.validate_head(state, outputs)
        if tensor_fingerprint(state) != cell["artifact"]["tensor_sha256"]:
            raise ValueError("Loaded tensors differ from the matched artifact audit")
        Classifier, _, _ = _build_activity_module(outputs)
        self.model = Classifier(480, outputs)
        self.model.load_state_dict(state, strict=True)  # Before attaching the pretrained backbone.
        model_id, revision = cell["config"]["esm_model"], cell["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model.esm = AutoModel.from_pretrained(model_id, revision=revision)
        if self.model.esm.config.hidden_size != 480 or self.model.esm.config._commit_hash != revision:
            raise ValueError("Resolved scoring backbone revision/dimension differs")
        self.model.float().to(device).eval()
        self.device, self.outputs = device, outputs
        self.info = {"model": model_id, "requested_revision": revision,
                     "resolved_revision": self.model.esm.config._commit_hash,
                     "revision_origin": "top100 comparison, NOT verified original training revision",
                     "dtype": "float32", "max_token_length": 52, "pooling": "production masked mean incl special tokens"}

    def predict(self, sequences: list[str], batch_size: int) -> np.ndarray:
        import torch

        values = np.empty((len(sequences), self.outputs), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start:start + batch_size]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=52)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                raw = self.model(enc["input_ids"], enc["attention_mask"])
                values[start:start + len(batch)] = raw.cpu().numpy().reshape(len(batch), self.outputs)
                if start == 0 or (start // batch_size + 1) % 25 == 0:
                    print(f"[reward-reconstruction] inferred {min(start + batch_size, len(sequences))}/{len(sequences)}", flush=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite model predictions")
        return values


def label_inventory(path: Path, records: list[dict]) -> dict:
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    evidence = defaultdict(set)
    for row in raw:
        seq, organism = row.get("sequence", "").strip().upper(), row.get("organism", "").strip()
        label = row.get("label", "").strip().lower()
        if label in {"active", "1", "1.0", "true"}:
            evidence[(seq, organism)].add(1)
        elif label in {"inactive", "0", "0.0", "false"}:
            evidence[(seq, organism)].add(0)
    sequences = [r["sequence"] for r in records]
    return {"raw_csv_rows": len(raw), "loaded_records": len(records), "loaded_unique_sequences": len(set(sequences)),
            "loaded_sequences_longer_than_50": sum(len(s) > 50 for s in sequences),
            "loaded_sequences_with_nonstandard_alphabet": sum(bool(set(s) - set(AMINO_ACIDS)) for s in sequences),
            "raw_sequence_organism_label_conflicts": sum(len(v) > 1 for v in evidence.values()),
            "loader_note": "Current production training loaders retained without cleaning/dedup changes; panel genus conflicts use last applicable CSV row. Raw conflict count is before genus mapping."}


def write_split(directory: Path, cell: dict) -> None:
    train_ids = set(cell["train"])
    rows = []
    for i, record in enumerate(cell["records"]):
        row = {"loaded_record_index": i, "sequence": record["sequence"], "split": "train" if i in train_ids else "validation"}
        row.update({f"target:{name}": float(cell["targets"][i, j]) for j, name in enumerate(cell["outputs"])})
        row.update({f"observed:{name}": int(cell["masks"][i, j]) for j, name in enumerate(cell["outputs"])})
        rows.append(row)
    write_summary(directory / "split_assignments.csv", rows)


def write_predictions(directory: Path, cell: dict, logits: np.ndarray, nearest: list[float]) -> None:
    raw, calibrated = sigmoid(logits), sigmoid(logits.astype(np.float64) / cell["spec"]["temperature"])
    rows = []
    for i, index in enumerate(cell["val"]):
        seq = cell["records"][index]["sequence"]
        props = compute_properties(seq) if not (set(seq) - set(AMINO_ACIDS)) else None
        for j, output in enumerate(cell["outputs"]):
            rows.append({"validation_order": i, "loaded_record_index": int(index), "sequence": seq, "output": output,
                         "target": float(cell["targets"][index, j]), "observed": int(cell["masks"][index, j]),
                         "raw_logit": float(logits[i, j]), "uncalibrated_probability": float(raw[i, j]),
                         "stored_temperature_probability": float(calibrated[i, j]),
                         "nearest_reconstructed_training_similarity": nearest[i],
                         "length": len(seq), "charge": props.charge if props else None,
                         "hydrophobicity_kd": props.hydrophobicity_kd if props else None,
                         "properties_status": "computed" if props else "nonstandard_alphabet"})
    write_summary(directory / "validation_predictions.csv", rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--artifact-audit", type=Path, default=Path("sweep_results/reward-artifact-audit-v2"))
    parser.add_argument("--source", type=Path, default=Path("sweep_results/epoch58-top100-v1"))
    parser.add_argument("--tasks", nargs="+", choices=["hemolysis", "activity", "panel"], default=["hemolysis", "activity", "panel"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--list", action="store_true", help="Hash/metadata/split preflight; no writes or neural inference")
    parser.add_argument("--accept-reconstruction", action="store_true", help="Acknowledge the protocol's unverified historical data/split assumptions")
    args = parser.parse_args(argv)
    args.tasks = list(dict.fromkeys(args.tasks))
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.list and not args.accept_reconstruction:
        parser.error("Use --accept-reconstruction to acknowledge this is NOT verified historical validation")
    inputs, cells = prepare_inputs(args)
    for assumption in inputs["protocol"]["assumptions"]:
        print(f"[ASSUMPTION] {assumption}", flush=True)
    for cell in cells:
        print(f"[reward-reconstruction] {cell['name']}: member{cell['spec']['member']}, seed{cell['spec']['seed']}, "
              f"train={len(cell['train'])}, validation={len(cell['val'])}, T={cell['spec']['temperature']}", flush=True)
    if args.list:
        return
    recipe = {"kind": "reward_validation_reconstruction_v1", "code": code_identity(), "inputs": inputs,
              "tasks": args.tasks, "device": args.device, "batch_size": args.batch_size,
              "historical_validation_verified": False, "independent_test": False,
              "rank_metric_space": "raw_logits; fixed positive temperature preserves ordering",
              "reliability": "10 equal-width probability bins; Wilson intervals assume iid rows, NOT independent peptide families",
              "baseline_log_loss_clip": 1e-7, "auroc_discrepancy_investigation_trigger": .001}
    prepare_run(args.out, recipe)
    if (args.out / "complete.json").exists():
        baseline.checked_stage(args.out, "complete.json", {"results.csv"})
    summaries = []
    for cell in cells:
        directory = args.out / cell["name"]
        if (directory / "complete.json").exists():
            baseline.checked_stage(directory, "complete.json", set(OUTPUT_FILES))
            measured = json.loads((directory / "metrics.json").read_text())
            summaries.append(measured["summary"])
            continue
        directory.mkdir(parents=True, exist_ok=True)
        val, train = cell["val"], cell["train"]
        sequences = [cell["records"][i]["sequence"] for i in val]
        if (directory / "inference.json").exists():
            baseline.checked_stage(directory, "inference.json", {"predictions.npz", "backbone.json"})
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                if set(saved.files) != {"logits"}:
                    raise ValueError("Invalid cached inference schema")
                logits = saved["logits"]
        else:
            print(f"[reward-reconstruction] loading pinned 35M predictor: {cell['name']}", flush=True)
            predictor = PinnedPredictor(cell, device=args.device)
            logits = predictor.predict(sequences, args.batch_size)
            np.savez_compressed(directory / "predictions.npz", logits=logits)
            write_json(directory / "backbone.json", {**predictor.info, "batch_size": args.batch_size})
            del predictor
            gc.collect()
            if args.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
            verify_inputs(inputs["files"])
            mark_files(directory, "inference.json", ["predictions.npz", "backbone.json"])
        metrics, rows, bins = prediction_report(logits, cell["targets"][val], cell["masks"][val],
                                                cell["targets"][train], cell["masks"][train],
                                                temperature=cell["spec"]["temperature"], outputs=cell["outputs"])
        print(f"[reward-reconstruction] exact reconstructed cross-split similarity: {cell['name']} (CPU)", flush=True)
        leakage, nearest = cross_split_similarity([cell["records"][i]["sequence"] for i in train], sequences)
        write_json(directory / "split_audit.json", {**leakage, "labels": label_inventory(cell["data"], cell["records"])})
        write_split(directory, cell)
        write_predictions(directory, cell, logits, nearest)
        write_summary(directory / "per_output_metrics.csv", rows)
        write_summary(directory / "reliability_bins.csv", bins)
        current = metrics["stored_temperature"]
        auc = current["macro_auroc"]
        delta = auc - cell["spec"]["recorded_val_auroc"] if auc is not None else None
        summary = {"task": cell["name"], "member": cell["spec"]["member"], "seed": cell["spec"]["seed"],
                   "temperature": cell["spec"]["temperature"], "recorded_val_auroc": cell["spec"]["recorded_val_auroc"],
                   **current, "auroc_delta_from_record": delta,
                   "discrepancy_requires_investigation": delta is None or abs(delta) > .001,
                   "historical_validation_verified": False, "independent_test": False, **leakage}
        # JSON manifests sort keys; keep fresh/resumed CSV column order identical.
        summary = dict(sorted(summary.items()))
        write_json(directory / "metrics.json", {"summary": summary, "modes": metrics, "assumptions": inputs["protocol"]["assumptions"]})
        verify_inputs(inputs["files"])
        mark_files(directory, "complete.json", OUTPUT_FILES)
        summaries.append(summary)
        write_summary(args.out / "results.csv", summaries)
        print(f"[reward-reconstruction] complete {cell['name']}: AUROC={auc}, delta={delta}; historical validation remains unverified", flush=True)
    verify_inputs(inputs["files"])
    write_summary(args.out / "results.csv", summaries)
    mark_files(args.out, "complete.json", ["results.csv"])


if __name__ == "__main__":
    main()
