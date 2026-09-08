"""Train/seal fresh predictors, then explicitly unlock the family-held-out test.

No production promotion. Linear and production-shaped MLP controls use identical
frozen embeddings and joint splits. All three seeds are retained and ensembled.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from compare_top100 import checked_stage
from experiment_utils import (
    REPO_ROOT,
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    write_json,
    write_summary,
)
from prepare_reward_generalization import FILES, validate_protocol, verify_hashes
from selection_cache import separate_output

from amp_challenge_2027.generalization import TASKS, arrays, calibrate_temperature, family_bootstrap
from amp_challenge_2027.reward_audit import prediction_report, sigmoid
from amp_challenge_2027.reward_benchmark import FrozenEncoder, fit_head, predict_head

HEAD_FILES = {"head.pt", "config.json", "history.csv", "validation_logits.npy", "calibration_logits.npy"}


def load_prepared(path):
    root = path.resolve()
    checked_stage(root, "complete.json", FILES | {"run.json"})
    recipe = json.loads((root / "run.json").read_text())
    if recipe.get("kind") != "reward_generalization_data_v1":
        raise ValueError("Not a prepared generalization benchmark")
    validate_protocol(recipe["protocol"])
    verify_hashes(recipe["inputs"])
    preflight = json.loads((root / "preflight.json").read_text())
    if not preflight["eligible_for_training"] or preflight["boundary"]["violations"] != 0:
        raise ValueError("Prepared dataset failed split/support preflight")
    data = json.loads((root / "dataset.json").read_text())["tasks"]
    # Validate the serialized schema as well as its hash, before model work.
    assignments, families = {}, {}
    for task in TASKS:
        records, outputs = data[task]["records"], data[task]["outputs"]
        if not records or len({r["sequence"] for r in records}) != len(records):
            raise ValueError("Prepared sequences must be nonempty and unique per task")
        for rec in records:
            if (len(rec["targets"]) != len(outputs) or len(rec["mask"]) != len(outputs)
                    or not set(rec["targets"]) <= {0, 1} or not set(rec["mask"]) <= {0, 1}
                    or not any(rec["mask"]) or rec["split"] not in {"train", "validation", "calibration", "test"}):
                raise ValueError("Invalid prepared target/mask/split schema")
            for mapping, key in ((assignments, rec["sequence"]), (families, rec["family"])):
                if mapping.setdefault(key, rec["split"]) != rec["split"]:
                    raise ValueError("Sequence/family crosses tasks or partitions")
    return recipe, data


def freeze_prepared_inputs(root):
    return {str((root / name).resolve()): sha256(root / name) for name in FILES | {"complete.json", "run.json"}}


def embedding_cache(root, data, splits, recipe, *, device, batch_size):
    """Only requested splits are encoded. Training never requests test features."""
    encoder, result = None, {}
    for split in splits:
        sequences = sorted({r["sequence"] for task in data.values() for r in task["records"] if r["split"] == split})
        dest = root / "embeddings" / split
        if (dest / "complete.json").exists():
            checked_stage(dest, "complete.json", {"features.npy", "sequences.json", "backbone.json"})
            if json.loads((dest / "sequences.json").read_text())["sequences"] != sequences:
                raise ValueError("Cached embedding order differs")
            info = json.loads((dest / "backbone.json").read_text())
            if info.get("model") != recipe["backbone"] or info.get("revision") != recipe["revision"]:
                raise ValueError("Cached embedding backbone differs")
            features = np.load(dest / "features.npy", allow_pickle=False)
        else:
            if encoder is None:
                encoder = FrozenEncoder(recipe["backbone"], recipe["revision"], device=device)
            dest.mkdir(parents=True, exist_ok=True)
            features = encoder.encode(sequences, batch_size)
            if features.shape != (len(sequences), 480) or features.dtype != np.float32 or not np.isfinite(features).all():
                raise ValueError("Invalid frozen feature cache")
            np.save(dest / "features.npy", features, allow_pickle=False)
            write_json(dest / "sequences.json", {"sequences": sequences})
            write_json(dest / "backbone.json", encoder.info)
            mark_files(dest, "complete.json", ["features.npy", "sequences.json", "backbone.json"])
        if features.shape != (len(sequences), 480) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("Invalid frozen feature cache")
        result[split] = ({s: i for i, s in enumerate(sequences)}, features)
    del encoder
    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    return result


def task_arrays(task, split, embeddings):
    y, mask, sequences = arrays(task["records"], split)
    index, features = embeddings[split]
    return features[[index[s] for s in sequences]], y, mask


def verify_training(root, protocol):
    checked_stage(root, "sealed.json", {"run.json", "training_summary.csv", "model_inventory.json", "embedding_inventory.json"})
    features = json.loads((root / "embedding_inventory.json").read_text())
    if set(features) != {"train", "validation", "calibration"}:
        raise ValueError("Training feature seal must exclude test and include all development partitions")
    for split, digest in features.items():
        path = root / "embeddings" / split
        if sha256(path / "complete.json") != digest:
            raise ValueError("Training feature marker changed")
        checked_stage(path, "complete.json", {"features.npy", "sequences.json", "backbone.json"})
    inventory = json.loads((root / "model_inventory.json").read_text())
    expected = {f"{task}/{arch}/seed{seed}" for task in TASKS for arch in protocol["architectures"]
                for seed in protocol["training_seeds"]}
    expected |= {f"{task}/{arch}/ensemble" for task in TASKS for arch in protocol["architectures"]}
    if set(inventory) != expected:
        raise ValueError("Training seal is missing a predeclared task/architecture/seed")
    for cell, digest in inventory.items():
        path = root / cell
        if sha256(path / "complete.json") != digest:
            raise ValueError("Frozen training marker changed")
        required = {"calibration.json"} if cell.endswith("/ensemble") else HEAD_FILES
        checked_stage(path, "complete.json", required)
    return inventory


def run_train(args, recipe, data):
    import torch

    protocol = recipe["protocol"]
    inputs = freeze_prepared_inputs(args.prepared)
    run = {"kind": "reward_generalization_training_v1", "code": code_identity(), "inputs": inputs,
           "data_recipe": recipe, "device": args.device, "embedding_batch_size": args.batch_size,
           "test_used": False, "ensemble": "unweighted mean of all three raw-logit heads; no best-seed promotion"}
    if args.list:
        print(json.dumps({"protocol": protocol, "test_used": False, "out": str(args.out)}, indent=2))
        return
    prepare_run(args.out, run)
    if (args.out / "sealed.json").exists():
        verify_training(args.out, protocol)
        print("[generalization] verified sealed training; test remains separate")
        return
    embeddings = embedding_cache(args.out, data, ["train", "validation", "calibration"], recipe,
                                 device=args.device, batch_size=args.batch_size)
    verify_hashes(inputs)
    summary, inventory = [], {}
    for task in TASKS:
        tx, ty, tm = task_arrays(data[task], "train", embeddings)
        vx, vy, vm = task_arrays(data[task], "validation", embeddings)
        cx, cy, cm = task_arrays(data[task], "calibration", embeddings)
        for arch in protocol["architectures"]:
            calibration_logits = []
            for seed in protocol["training_seeds"]:
                cell = f"{task}/{arch}/seed{seed}"
                dest = args.out / cell
                cfg = {"task": task, "architecture": arch, "seed": seed, "outputs": data[task]["outputs"],
                       "backbone": recipe["backbone"], "revision": recipe["revision"], "hidden_size": 480,
                       "unfreeze_layers": 0, "checkpoint_format": "head-only", "deployment_approved": False}
                if (dest / "complete.json").exists():
                    checked_stage(dest, "complete.json", HEAD_FILES)
                    if json.loads((dest / "config.json").read_text()) != cfg:
                        raise ValueError("Cached head config differs")
                    logits = np.load(dest / "calibration_logits.npy", allow_pickle=False)
                else:
                    print(f"[generalization] training {cell}", flush=True)
                    state, history, val_logits = fit_head(tx, ty, tm, vx, vy, vm, protocol=protocol,
                                                         seed=seed, architecture=arch, device=args.device)
                    logits = predict_head(state, cx, len(data[task]["outputs"]), arch, device=args.device)
                    dest.mkdir(parents=True, exist_ok=True)
                    torch.save(state, dest / "head.pt")
                    write_json(dest / "config.json", cfg)
                    write_summary(dest / "history.csv", history)
                    np.save(dest / "validation_logits.npy", val_logits, allow_pickle=False)
                    np.save(dest / "calibration_logits.npy", logits, allow_pickle=False)
                    verify_hashes(inputs)
                    mark_files(dest, "complete.json", sorted(HEAD_FILES))
                if logits.shape != cy.shape or not np.isfinite(logits).all():
                    raise ValueError("Invalid calibration predictions")
                calibration_logits.append(logits)
                inventory[cell] = sha256(dest / "complete.json")
            ensemble = np.mean(calibration_logits, axis=0, dtype=np.float64)
            calibration = calibrate_temperature(ensemble, cy, cm)
            cell = f"{task}/{arch}/ensemble"
            dest = args.out / cell
            if (dest / "complete.json").exists():
                checked_stage(dest, "complete.json", {"calibration.json"})
                if json.loads((dest / "calibration.json").read_text()) != calibration:
                    raise ValueError("Frozen ensemble calibration differs")
            else:
                write_json(dest / "calibration.json", calibration)
                mark_files(dest, "complete.json", ["calibration.json"])
            inventory[cell] = sha256(dest / "complete.json")
            summary.append({"task": task, "architecture": arch, "temperature": calibration["temperature"],
                            "calibration_status": calibration["status"], "test_used": False,
                            "seeds": ",".join(map(str, protocol["training_seeds"]))})
    verify_hashes(inputs)
    write_summary(args.out / "training_summary.csv", summary)
    write_json(args.out / "model_inventory.json", inventory)
    write_json(args.out / "embedding_inventory.json", {
        split: sha256(args.out / "embeddings" / split / "complete.json")
        for split in ("train", "validation", "calibration")})
    mark_files(args.out, "sealed.json", ["run.json", "training_summary.csv", "model_inventory.json", "embedding_inventory.json"])
    print("[generalization] ALL fits sealed. No test embeddings or test predictions produced.", flush=True)


def run_test(args, recipe, data):
    import torch

    training = args.training.resolve()
    protocol = recipe["protocol"]
    verify_training(training, protocol)
    train_run = json.loads((training / "run.json").read_text())
    inputs = freeze_prepared_inputs(args.prepared)
    if (train_run.get("kind") != "reward_generalization_training_v1" or train_run.get("inputs") != inputs
            or train_run.get("data_recipe") != recipe or train_run.get("test_used") is not False):
        raise ValueError("Training belongs to a different prepared dataset/protocol")
    if train_run.get("code") != code_identity():
        raise ValueError("Test requires the sealed training code/runtime; do not change the inference implementation after fitting")
    inputs[str(training / "sealed.json")] = sha256(training / "sealed.json")
    run = {"kind": "reward_generalization_test_v1", "code": code_identity(), "inputs": inputs,
           "training": str(training), "device": args.device, "embedding_batch_size": args.batch_size,
           "scope": "internal family-held-out for freshly trained heads only; old deployed heads have seen overlapping data",
           "warning": "Once inspected, this test cannot be reused as fresh evidence for subsequent tuned methods"}
    if args.list:
        print(json.dumps(run, indent=2))
        return
    prepare_run(args.out, run)
    if (args.out / "complete.json").exists():
        checked_stage(args.out, "complete.json", {"run.json", "results.csv", "paired_architecture_deltas.json"})
        for task in TASKS:
            for arch in protocol["architectures"]:
                checked_stage(args.out / task / arch, "complete.json",
                              {"predictions.npz", "predictions.csv", "metrics.json", "per_output_metrics.csv", "reliability_bins.csv"})
        print(f"[generalization] verified completed test: {args.out}")
        return
    embeddings = embedding_cache(args.out, data, ["test"], recipe, device=args.device, batch_size=args.batch_size)
    summaries, deltas = [], {}
    for task in TASKS:
        x, y, mask = task_arrays(data[task], "test", embeddings)
        ty, tm, _ = arrays(data[task]["records"], "train")
        records = [r for r in data[task]["records"] if r["split"] == "test"]
        families = [r["family"] for r in records]
        ensemble_by_arch = {}
        for arch in protocol["architectures"]:
            dest = args.out / task / arch
            if (dest / "complete.json").exists():
                checked_stage(dest, "complete.json",
                              {"predictions.npz", "predictions.csv", "metrics.json", "per_output_metrics.csv", "reliability_bins.csv"})
                saved = json.loads((dest / "metrics.json").read_text())
                with np.load(dest / "predictions.npz", allow_pickle=False) as cache:
                    if (cache["logits"].shape != y.shape or not np.array_equal(cache["targets"], y)
                            or not np.array_equal(cache["masks"], mask)):
                        raise ValueError("Completed test predictions have a different target order")
                    ensemble_by_arch[arch] = cache["logits"]
                summaries.append(saved["summary"])
                continue
            predictions = []
            for seed in protocol["training_seeds"]:
                state = torch.load(training / task / arch / f"seed{seed}" / "head.pt", map_location="cpu", weights_only=True)
                predictions.append(predict_head(state, x, len(data[task]["outputs"]), arch, device=args.device))
            logits = np.mean(predictions, axis=0, dtype=np.float64)
            ensemble_by_arch[arch] = logits
            calibration = json.loads((training / task / arch / "ensemble/calibration.json").read_text())
            temp = calibration["temperature"]
            measured, rows, bins = prediction_report(logits, y, mask, ty, tm, temperature=temp, outputs=data[task]["outputs"])
            ci = family_bootstrap(logits, y, mask, families, replicates=protocol["bootstrap_replicates"])
            seed_results = {}
            for seed, raw in zip(protocol["training_seeds"], predictions):
                modes, _, _ = prediction_report(raw, y, mask, ty, tm, temperature=1., outputs=data[task]["outputs"])
                seed_results[str(seed)] = modes["uncalibrated"]
            for row in rows:
                j = data[task]["outputs"].index(row["output"])
                observed = np.flatnonzero(mask[:, j] == 1)
                row["observed_families"] = len({families[i] for i in observed})
                row["positive_families"] = len({families[i] for i in observed if y[i, j] == 1})
                row["negative_families"] = len({families[i] for i in observed if y[i, j] == 0})
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / "predictions.npz", logits=logits, per_seed_logits=np.stack(predictions), targets=y, masks=mask)
            prob = sigmoid(logits / temp)
            write_summary(dest / "predictions.csv", [
                {"sequence": rec["sequence"], "family": rec["family"], "output": output, "target": int(y[i, j]),
                 "observed": int(mask[i, j]), "raw_logit": float(logits[i, j]), "calibrated_probability": float(prob[i, j])}
                for i, rec in enumerate(records) for j, output in enumerate(data[task]["outputs"])])
            write_summary(dest / "per_output_metrics.csv", rows)
            write_summary(dest / "reliability_bins.csv", bins)
            summary = dict(sorted({"task": task, "architecture": arch, "sequences": len(records), "families": len(set(families)),
                                   **measured["stored_temperature"], "auroc_ci_lower": ci.get("percentile_95_lower"),
                                   "auroc_ci_upper": ci.get("percentile_95_upper"), "interval_status": ci["status"]}.items()))
            write_json(dest / "metrics.json", {"modes": measured, "family_auroc_interval": ci, "per_seed_uncalibrated": seed_results,
                                                "calibration": calibration, "scope": run["scope"], "summary": summary})
            mark_files(dest, "complete.json", ["predictions.npz", "predictions.csv", "metrics.json", "per_output_metrics.csv", "reliability_bins.csv"])
            summaries.append(summary)
            print(f"[generalization] test {task}/{arch}: AUROC={measured['stored_temperature']['macro_auroc']}", flush=True)
        deltas[task] = {"comparison": "mlp minus linear, paired family resampling",
                        **family_bootstrap(ensemble_by_arch["mlp"], y, mask, families,
                                           comparator=ensemble_by_arch["linear"], replicates=protocol["bootstrap_replicates"])}
    verify_hashes(inputs)
    verify_training(training, protocol)
    write_summary(args.out / "results.csv", summaries)
    write_json(args.out / "paired_architecture_deltas.json", deltas)
    mark_files(args.out, "complete.json", ["run.json", "results.csv", "paired_architecture_deltas.json"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "test"])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--training", type=Path, help="Sealed training output; test stage only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="Frozen embedding batch size; head batch size is in protocol")
    parser.add_argument("--list", action="store_true", help="Read-only preflight, no models or output writes")
    parser.add_argument("--unlock-test", action="store_true", help="Acknowledge final test exposure after all predeclared fits are sealed")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.stage == "test" and args.training is None):
        parser.error("Positive batch size and --training for test are required")
    if args.stage == "test" and not args.list and not args.unlock_test:
        parser.error("Final test requires --unlock-test; no fitting is allowed afterward on this test")
    if args.stage == "train" and (args.training or args.unlock_test):
        parser.error("--training and --unlock-test are test-only")
    separate_output(args.out, [args.prepared, REPO_ROOT / "checkpoint", *([args.training] if args.training else [])])
    recipe, data = load_prepared(args.prepared)
    if args.stage == "train":
        run_train(args, recipe, data)
    else:
        run_test(args, recipe, data)


if __name__ == "__main__":
    main()
