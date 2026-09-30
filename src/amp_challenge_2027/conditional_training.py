"""Offline, measured-assay generator experiments, isolated from release generation.

The training unit is a unique sequence, followed by one observation of that
sequence. Missing measurements are never converted into favorable labels.
Validation losses count valid target tokens (including EOS), not batches.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from amp_challenge_2027 import tokenizer
from amp_challenge_2027.assay_conditioning import (
    collate_assay_conditions,
    fit_condition_schema,
)
from amp_challenge_2027.model import (
    DecoderConfig,
    build_model,
    is_assay_parameter,
    load_model,
    save_model,
    warm_start_model,
)
from amp_challenge_2027.training import build_cosine_scheduler, seed_everything

PRESETS = {
    "A": {"residual": "standard", "ffn": "dense"},
    "B": {"residual": "block_attnres", "ffn": "dense"},
    "C": {"residual": "standard", "ffn": "moe"},
    "D": {"residual": "block_attnres", "ffn": "moe"},
}
ARMS = ("frozen", "unconditional", "selective", "conditional")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _prepared_provenance(path: Path, payload: dict) -> dict:
    """Verify prepared bytes and their embedded raw-manifest/parser identities."""
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return {"status": "unverified_manual_dataset", "input_hashes": {str(path): _hash(path)}}
    manifest = json.loads(manifest_path.read_text())
    outputs = manifest.get("outputs")
    recipe = manifest.get("recipe")
    if not isinstance(outputs, dict) or path.name not in outputs or not isinstance(recipe, dict):
        raise ValueError("Prepared dataset manifest lacks required outputs or recipe")
    hashes = {str(manifest_path): _hash(manifest_path)}
    for name, digest in outputs.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe prepared dataset output path")
        file = path.parent / rel
        if not file.is_file() or _hash(file) != digest:
            raise ValueError(f"Prepared dataset hash mismatch: {file}")
        hashes[str(file)] = digest
    if payload.get("recipe") != recipe:
        raise ValueError("Prepared dataset recipe differs from its manifest")
    source = payload.get("source_manifest")
    if not isinstance(source, dict) or not isinstance(source.get("artifacts"), list):
        raise ValueError("Prepared dataset lacks the raw source manifest")
    canonical = (json.dumps(source, indent=2, sort_keys=True, ensure_ascii=False,
                            allow_nan=False) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != recipe.get("snapshot_sha256"):
        raise ValueError("Prepared dataset raw-source manifest identity differs")
    for digest in [recipe.get("parser_sha256"), *[row.get("sha256") for row in source["artifacts"]]]:
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Prepared dataset requires valid parser and raw source hashes")
    return {
        "status": "prepared_manifest_verified", "input_hashes": hashes,
        "snapshot_sha256": recipe["snapshot_sha256"], "parser_sha256": recipe["parser_sha256"],
        "raw_source_artifacts": len(source["artifacts"]),
        "scope": "Prepared bytes and embedded identities verified; raw files need not remain co-located",
    }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def qualifies_for_selective(conditions: list[dict[str, Any]]) -> bool:
    """Conservative measured positive: MIC upper<=16uM and <=10% human lysis at64uM.

    The data builder already restricts molecular chemistry. Here both outcomes
    must additionally identify the same nonempty publication. A lower-bound
    hemolysis measurement, unknown species, or HC50 is not a negative label.
    """
    activity_sources: set[str] = set()
    safe_sources: set[str] = set()
    unresolved_safety_sources: set[str] = set()
    for item in conditions:
        source = str(item.get("publication_id") or "").strip()
        if not source:
            continue
        endpoint = str(item.get("endpoint", "")).lower()
        upper = _number(item.get("value_upper"))
        unit = str(item.get("unit", "")).replace("µ", "u").replace("μ", "u")
        has_upper = upper is not None and upper >= 0 and item.get("operator") not in {">", ">="}
        if endpoint == "mic" and unit == "uM" and has_upper and upper <= 16:
            activity_sources.add(source)
        elif endpoint in {"%lysis", "lysis_percent", "percent_lysis", "hemolysis_percent"}:
            species = str(item.get("rbc_species", "")).lower().strip()
            lower_dose = _number(item.get("dose_lower"))
            upper_dose = _number(item.get("dose_upper"))
            if (
                item.get("unit") in {"%", "percent"}
                and (species in {"human", "homo sapiens"}
                     or species.startswith(("human ", "homo sapiens ")))
                and str(item.get("dose_unit", "")).replace("µ", "u").replace("μ", "u") == "uM"
                and lower_dose == upper_dose == 64.0
                and item.get("dose_operator") in {None, "exact", "=", "range"}
            ):
                if has_upper and upper <= 10.0:
                    safe_sources.add(source)
                else:
                    unresolved_safety_sources.add(source)
    return bool(activity_sources & (safe_sources - unresolved_safety_sources))


def _read_dataset(path: Path) -> tuple[Path, dict, dict[str, dict[str, list[dict]]]]:
    path = path / "dataset.json" if path.is_dir() else path
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported conditional dataset schema_version")
    provenance = _prepared_provenance(path, payload)
    payload = dict(payload, training_input_integrity=provenance)
    rows = payload.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset requires nonempty examples")
    grouped = {split: defaultdict(list)
               for split in ("train", "validation", "calibration", "test")}
    seen_sequences: dict[str, str] = {}
    seen_families: dict[str, str] = {}
    for row in rows:
        split = row.get("split")
        if split == "val":
            split = "validation"
        if split not in grouped:
            raise ValueError(f"Unknown dataset split: {split!r}")
        sequence = row.get("sequence", "")
        if not isinstance(sequence, str) or not 8 <= len(sequence) <= 50:
            raise ValueError("Dataset sequences must be 8-50 canonical residues")
        tokenizer.encode(sequence)
        family = str(row.get("family", ""))
        if not family:
            raise ValueError("Dataset examples require sequence family IDs")
        for key, table in ((sequence, seen_sequences), (family, seen_families)):
            if key in table and table[key] != split:
                raise ValueError("Sequence/family leakage across dataset partitions")
            table[key] = split
        if not isinstance(row.get("conditions"), list):
            raise ValueError("Every example requires an explicit conditions list")
        if any(not isinstance(item, dict) for item in row["conditions"]):
            raise ValueError("Conditions must be objects")
        grouped[split][sequence].append(dict(row, split=split))
    if not grouped["train"] or not grouped["validation"]:
        raise ValueError("Nonempty train and validation partitions are required")
    return path, payload, grouped


def _usable_conditions(row: dict) -> list[dict]:
    return row["conditions"] if row.get("supervision_eligible") is True else []


def _observation_groups(rows: list[dict]) -> list[list[dict]]:
    """Choose one publication's measured requests, without inventing a joint assay.

    Records with unknown publication IDs remain individual observations; they
    must not be combined into synthetic paired outcomes.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    unknown = 0
    for row in rows:
        for item in _usable_conditions(row):
            source = str(item.get("publication_id") or "").strip()
            if not source:
                unknown += 1
                source = f"__unknown_{unknown}"
            groups[source].append(item)
    return list(groups.values()) or [[]]


def _sample_epoch(
    groups: dict[str, list[dict]],
    favored: list[str],
    count: int,
    rng: random.Random,
) -> list[tuple[str, list[dict]]]:
    """75% observed sequences, 25% unconditionally replayed sequences.

    Draw sequence first, then uniformly choose one eligible observation. This
    prevents the number of assay rows from setting a sequence's sampling mass.
    """
    all_sequences = sorted(groups)
    samples: list[tuple[str, list[dict]]] = []
    annotated_count = round(0.75 * count) if favored else 0
    for _ in range(annotated_count):
        sequence = rng.choice(favored)
        observations = _observation_groups(groups[sequence])
        samples.append((sequence, rng.choice(observations)))
    for _ in range(count - annotated_count):
        samples.append((rng.choice(all_sequences), []))
    rng.shuffle(samples)
    return samples


def _tokens(sequences: list[str], device: str) -> torch.Tensor:
    encoded = [tokenizer.encode(sequence) for sequence in sequences]
    ids = torch.full(
        (len(encoded), max(map(len, encoded))), tokenizer.PAD_ID,
        dtype=torch.long, device=device,
    )
    for index, row in enumerate(encoded):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return ids


def causal_loss(logits: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-example NLL sums and valid target counts, excluding padding."""
    targets = ids[:, 1:]
    loss = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), targets,
        ignore_index=tokenizer.PAD_ID, reduction="none",
    )
    return loss.sum(-1), targets.ne(tokenizer.PAD_ID).sum(-1)


def _condition_kwargs(condition_lists, schema, device, *, dropout=0.0, generator=None):
    if schema is None:
        return {}
    return {"assay_conditions": collate_assay_conditions(
        condition_lists, schema, device=device, dropout=dropout, generator=generator,
    )}


def evaluate_model(model, groups, schema, *, device="cpu", batch_size=128, autocast=nullcontext):
    """Evaluate all observations, weighting every unique sequence equally per token."""
    observations = []
    for sequence, rows in sorted(groups.items()):
        groups_for_sequence = _observation_groups(rows)
        observations.extend((sequence, conditions, 1.0 / len(groups_for_sequence))
                            for conditions in groups_for_sequence)
    total_loss = total_tokens = aux_total = 0.0
    batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start:start + batch_size]
            ids = _tokens([row[0] for row in batch], device)
            kwargs = _condition_kwargs([row[1] for row in batch], schema, device)
            with autocast():
                output = model(ids, **kwargs)
            sums, counts = causal_loss(output.logits, ids)
            weights = torch.tensor([row[2] for row in batch], device=device)
            total_loss += (sums * weights).sum().item()
            total_tokens += (counts * weights).sum().item()
            aux_total += float(output.aux_loss.detach())
            batches += 1
    if total_tokens <= 0:
        raise ValueError("Validation has no valid target tokens")
    return {
        "token_nll": total_loss / total_tokens,
        "valid_tokens": total_tokens,
        "aux_loss": aux_total / max(batches, 1),
        "unique_sequences": len(groups),
    }


def configure_optimizer(model, *, freeze_backbone: bool, new_lr: float, base_lr: float):
    """Rebuild optimizer at stage boundaries; warmup cannot mutate old weights."""
    groups = {"new": [], "base": []}
    for name, parameter in model.named_parameters():
        is_new = is_assay_parameter(name)
        parameter.requires_grad_(is_new or not freeze_backbone)
        if parameter.requires_grad:
            groups["new" if is_new else "base"].append(parameter)
    param_groups = [
        {"params": parameters, "lr": new_lr if name == "new" else base_lr, "name": name}
        for name, parameters in groups.items() if parameters
    ]
    if not param_groups:
        raise ValueError("No trainable parameters in requested stage")
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


def _routing(output) -> list[dict]:
    result = []
    for layer in getattr(output, "routing_metrics", None) or []:
        result.append({
            key: value.detach().float().cpu().tolist() if torch.is_tensor(value) else value
            for key, value in layer.items()
        })
    return result


def _finish(out: Path, metrics: dict) -> dict:
    run = json.loads((out / "run.json").read_text())
    for name, digest in run["input_hashes"].items():
        if _hash(Path(name)) != digest:
            raise ValueError(f"Training input changed during the run: {name}")
    _write_json(out / "metrics.json", metrics)
    files = sorted(path for path in out.rglob("*") if path.is_file())
    _write_json(out / "complete.json", {
        "schema_version": 1, "status": metrics["status"],
        "files": {str(path.relative_to(out)): _hash(path) for path in files},
    })
    return metrics


def train_experiment(
    data: Path,
    out: Path,
    arm: str,
    preset: str | None = None,
    checkpoint: Path | None = None,
    seed: int = 42,
    device: str = "cuda",
    epochs: int | None = None,
    batch_size: int = 128,
    model_overrides: dict | None = None,
    warmup_epochs: int = 1,
    patience: int = 3,
    examples_per_epoch: int | None = None,
    min_selective: int = 32,
) -> dict:
    """Train one registered arm; source files and release checkpoints are read-only.

    Warm experiments train at most ten joint epochs after the default one-epoch
    condition-module warmup. Fresh presets use 100 full epochs by default and
    preserve an equal example budget instead of early stopping individual arms.
    Test overrides are explicit and recorded in the run manifest.
    """
    data, out = Path(data), Path(out)
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    if arm not in ARMS or (preset is not None and preset not in PRESETS):
        raise ValueError("Unknown arm or architecture preset")
    if checkpoint is not None and preset is not None:
        raise ValueError("Warm starts preserve checkpoint architecture; omit preset")
    if checkpoint is not None and model_overrides:
        raise ValueError("Model overrides cannot change warm-start architecture")
    if arm in {"frozen", "selective"} and checkpoint is None:
        raise ValueError(f"{arm} requires an existing checkpoint")
    if batch_size < 1 or patience < 1 or warmup_epochs < 0 or min_selective < 1:
        raise ValueError("Invalid batch, patience, warmup, or selective minimum")
    fresh = checkpoint is None
    epochs = (100 if fresh else 10) if epochs is None else epochs
    if epochs < 1 or (not fresh and epochs > 10):
        raise ValueError("Epochs must be positive; warm-start joint budget is at most ten")
    data_path, payload, splits = _read_dataset(data)
    groups = splits["train"]
    count = len(groups) if examples_per_epoch is None else examples_per_epoch
    if count < 1:
        raise ValueError("examples_per_epoch must be positive")
    if out.exists():
        raise ValueError("Use a new output directory; existing experiments are preserved")
    train_rows = [row for rows in groups.values() for row in rows]
    favored = sorted(sequence for sequence, rows in groups.items()
                     if any(_usable_conditions(row) for row in rows))
    qualified = sorted(sequence for sequence, rows in groups.items()
                       if qualifies_for_selective([item for row in rows
                                                   for item in _usable_conditions(row)]))
    schema = fit_condition_schema([dict(row, conditions=_usable_conditions(row)) for row in train_rows]) \
        if arm == "conditional" else None
    input_hashes = dict(payload["training_input_integrity"]["input_hashes"])
    if checkpoint is not None:
        input_hashes.update({str(checkpoint / name): _hash(checkpoint / name)
                             for name in ("model.pt", "config.json")})
    code_files = [Path(__file__).with_name(name) for name in (
        "conditional_training.py", "assay_conditioning.py", "model.py", "tokenizer.py",
        "training.py", "conditional_data.py", "generalization.py",
    )]
    run = {
        "schema_version": 1, "arm": arm, "preset": preset or ("A" if fresh else None),
        "checkpoint": str(checkpoint) if checkpoint else None, "seed": seed,
        "device": device, "joint_epochs": epochs, "warmup_epochs": warmup_epochs,
        "batch_size": batch_size, "examples_per_epoch": count,
        "min_selective": min_selective, "patience": patience,
        "model_overrides": model_overrides or {}, "input_hashes": input_hashes,
        "code_hashes": {path.name: _hash(path) for path in code_files},
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(device)
            if str(device).startswith("cuda") and torch.cuda.is_available() else platform.processor() or "CPU",
        },
        "optimization": {
            "optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "backbone_lr": 3e-4 if fresh else 1e-5,
            "condition_modules_lr": 3e-4 if fresh else 1e-4,
            "condition_token_dropout": 0.2, "annotated_sampling_fraction": 0.75,
            "replay_fraction": 0.25, "fresh_schedule": "linear_warmup_then_cosine" if fresh else None,
            "fresh_warmup_steps": max(1, math.ceil(0.05 * epochs * math.ceil(count / batch_size)))
            if fresh else 0,
            "fresh_early_stopping": False,
            "effective_condition_warmup_epochs": warmup_epochs if not fresh and arm == "conditional" else 0,
            "expected_fresh_train_examples": epochs * count if fresh else None,
            "exposure_comparison": "Same seed, dataset and count reproduce sequence draws across fresh presets; actual tokens reported",
        },
        "sampling": "unique sequence then observation; 75% annotated + 25% unconditional replay",
        "training_objective": "mean of per-sequence valid-token NLL; MoE auxiliary added separately",
        "validation": "observation-averaged per unique sequence, valid-token-weighted NLL",
        "source_manifest": payload.get("sources", payload.get("source_manifest", {})),
        "training_input_integrity": payload["training_input_integrity"],
        "split_unique_sequences": {name: len(rows) for name, rows in splits.items()},
        "annotated_train_sequences": len(favored),
        "selective_train_sequences": len(qualified),
        "pretraining_family_exposure": "unverified for historical warm-start checkpoints" if not fresh
                                       else "project-controlled training uses train partition only",
    }
    from amp_challenge_2027.conditional_research import new_output

    protected_inputs = [data_path.parent if (data_path.parent / "manifest.json").exists() else data_path]
    if checkpoint is not None:
        protected_inputs.append(checkpoint)
    out = new_output(out, protected_inputs)
    _write_json(out / "run.json", run)
    if arm == "selective" and len(qualified) < min_selective:
        return _finish(out, {
            "status": "insufficient_joint_labels", "qualified_sequences": len(qualified),
            "minimum_required": min_selective, "checkpoint_written": False,
        })
    if arm == "conditional" and not favored:
        return _finish(out, {"status": "insufficient_condition_labels", "checkpoint_written": False})
    if arm == "selective":
        favored = qualified
    seed_everything(seed)
    rng = random.Random(seed)
    dropout_rng = torch.Generator(device="cpu").manual_seed(seed + 1009)
    if checkpoint is not None:
        model, cfg = (warm_start_model(checkpoint, schema) if arm == "conditional"
                      else load_model(checkpoint))
    else:
        config = dict(hidden_size=384, num_layers=6, num_heads=6, ffn_inner=1536,
                      moe_num_experts=4, moe_num_active=2, moe_expert_inner=768,
                      residual_impl_version=2, attnres_block_layers=2, attnres_num_blocks=3,
                      assay_schema=schema)
        config.update(PRESETS[preset or "A"])
        config.update(model_overrides or {})
        model, cfg = build_model(DecoderConfig(**config))
    model.to(device)
    use_bf16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    run["precision"] = "bf16" if use_bf16 else "fp32"
    run["model_config"] = cfg.to_dict()
    run["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    expert_inner = cfg.moe_expert_inner or cfg.inner_size
    run["active_ffn_parameters_estimate"] = cfg.num_layers * (
        cfg.moe_num_active * (2 * cfg.hidden_size * expert_inner + expert_inner + cfg.hidden_size)
        if cfg.ffn == "moe" else
        2 * cfg.hidden_size * cfg.inner_size + cfg.inner_size + cfg.hidden_size
    )
    _write_json(out / "run.json", run)
    started = time.perf_counter()
    baseline = evaluate_model(model, splits["validation"], schema, device=device,
                              batch_size=batch_size, autocast=autocast)
    best_nll, best_epoch, best_step = baseline["token_nll"], 0, 0
    save_model(model, out / "checkpoint", config=cfg)
    if arm == "frozen":
        _write_json(out / "history.json", [])
        (out / "history.csv").write_text("epoch,stage,validation_nll\n")
        return _finish(out, {
            "status": "complete", "best_epoch": 0, "epochs_completed": 0,
            "best_validation_nll": best_nll, "baseline_validation": baseline,
            "wall_seconds": time.perf_counter() - started, "checkpoint_written": True,
        })
    stages = []
    if not fresh and arm == "conditional" and warmup_epochs:
        stages.append(("conditions_only", warmup_epochs, True))
    stages.append(("fresh" if fresh else "joint", epochs, False))
    history, epoch_index, step, tokens_seen, bad_epochs = [], 0, 0, 0, 0
    stop = False
    for stage, stage_epochs, freeze_backbone in stages:
        optimizer = configure_optimizer(
            model, freeze_backbone=freeze_backbone,
            new_lr=3e-4 if fresh else 1e-4, base_lr=3e-4 if fresh else 1e-5,
        )
        stage_steps = stage_epochs * math.ceil(count / batch_size)
        scheduler = (build_cosine_scheduler(optimizer, total_steps=stage_steps,
                                           warmup_steps=max(1, math.ceil(0.05 * stage_steps)))
                     if fresh else None)
        for _ in range(stage_epochs):
            epoch_index += 1
            model.train()
            samples = _sample_epoch(groups, favored, count, rng)
            epoch_start = time.perf_counter()
            nll_sum, token_count, aux_sum, batches = 0.0, 0, 0.0, 0
            routing = []
            for start in range(0, count, batch_size):
                batch = samples[start:start + batch_size]
                ids = _tokens([row[0] for row in batch], device)
                kwargs = _condition_kwargs([row[1] for row in batch], schema, device,
                                           dropout=0.2, generator=dropout_rng)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    output = model(ids, **kwargs)
                    sums, counts = causal_loss(output.logits, ids)
                    data_loss = (sums / counts.clamp_min(1)).mean()
                    loss = data_loss + output.aux_loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite generator loss; no successful completion marker written")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                nll_sum += sums.detach().sum().item()
                token_count += int(counts.sum().item())
                aux_sum += float(output.aux_loss.detach())
                batches += 1
                step += 1
                routing = _routing(output)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - epoch_start
            tokens_seen += token_count
            validation = evaluate_model(model, splits["validation"], schema, device=device,
                                        batch_size=batch_size, autocast=autocast)
            val_nll = validation["token_nll"]
            if not math.isfinite(val_nll):
                raise ValueError("Nonfinite validation NLL; checkpoint not promoted")
            improved = val_nll < best_nll
            if improved:
                best_nll, best_epoch, best_step = val_nll, epoch_index, step
                save_model(model, out / "checkpoint", config=cfg)
            if stage != "conditions_only":
                bad_epochs = 0 if improved else bad_epochs + 1
            record = {
                "epoch": epoch_index, "stage": stage, "step": step,
                "train_nll": nll_sum / token_count, "train_aux_loss": aux_sum / batches,
                "validation_nll": val_nll, "validation_aux_loss": validation["aux_loss"],
                "valid_train_tokens": token_count, "train_examples": count,
                "train_seconds": train_seconds, "tokens_per_second": token_count / train_seconds,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "backbone_frozen": freeze_backbone, "improved": improved,
                "optimizer_lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "routing_last_batch": routing,
            }
            history.append(record)
            _write_json(out / "history.json", history)
            print(f"[conditional-train] epoch={epoch_index} stage={stage} "
                  f"train_nll={record['train_nll']:.5f} val_nll={val_nll:.5f}", flush=True)
            if not fresh and stage != "conditions_only" and bad_epochs >= patience:
                stop = True
                break
        if stop:
            break
    with (out / "history.csv").open("w", newline="") as handle:
        columns = [key for key in history[0] if key not in {"optimizer_lrs", "routing_last_batch"}]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    return _finish(out, {
        "status": "complete", "epochs_completed": epoch_index,
        "best_epoch": best_epoch, "best_step": best_step,
        "best_validation_nll": best_nll, "baseline_validation": baseline,
        "early_stopped": stop, "wall_seconds": time.perf_counter() - started,
        "valid_train_tokens": tokens_seen, "checkpoint_written": True,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda") else 0,
        "total_parameters": run["total_parameters"],
        "active_ffn_parameters_estimate": run["active_ffn_parameters_estimate"],
        "scientific_claim": "Language-model training result; no biological efficacy or safety claim",
    })
