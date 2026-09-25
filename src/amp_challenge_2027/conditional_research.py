"""Fresh-draw evaluation for conditional generators; never promotes a release.

The legacy activity/hemolysis heads are instruments, not biological ground truth.
Sampling retains every draw; filtering never changes a raw-yield denominator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import (
    AMINO_ACIDS,
    BOS_ID,
    EOS_ID,
    MAX_LENGTH,
    MIN_LENGTH,
    PAD_ID,
    PANEL_GENERA,
    PROJECT_ROOT,
    REWARD_DIR,
    REWARD_HEMO_DIR,
)
from amp_challenge_2027.data import iter_fasta


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    paths += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(sha256(path).encode())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"commit": commit, "source_sha256": digest.hexdigest(), "python": platform.python_version()}


def new_output(out, inputs=()):
    out = Path(out).resolve()
    for item in inputs:
        item = Path(item).resolve()
        if out == item or out in item.parents or item in out.parents:
            raise ValueError("Research output must be separate from its inputs")
    for protected in ("checkpoint/generator", "checkpoint/generator_blend", "checkpoint/reward",
                      "checkpoint/reward_hemo", "data/processed"):
        root = (PROJECT_ROOT / protected).resolve()
        if out == root or root in out.parents or out in root.parents:
            raise ValueError(f"Refusing output overlapping deployed artifacts: {out}")
    if out.exists():
        raise ValueError(f"Use a new --out directory; preserving {out}")
    out.mkdir(parents=True)
    return out


def seal(out, names):
    write_json(Path(out) / "complete.json", {"files": {n: sha256(Path(out) / n) for n in names}})


def verify(out, required=()):
    out = Path(out)
    files = json.loads((out / "complete.json").read_text())["files"]
    if not set(required) <= set(files):
        raise ValueError(f"Incomplete research artifact: {out}")
    for name, digest in files.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or sha256(out / p) != digest:
            raise ValueError(f"Research artifact hash mismatch: {out}/{name}")
    return files


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def valid_sequence(sequence):
    return MIN_LENGTH <= len(sequence) <= MAX_LENGTH and not set(sequence) - set(AMINO_ACIDS)


def read_request(path):
    if path is None:
        return []
    value = json.loads(Path(path).read_text())
    value = value.get("conditions") if isinstance(value, dict) else value
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("Conditions JSON must contain a list of assay tokens")
    return value


def sample_experiment(checkpoint, out, *, label, seed=42, draws=10000, batch_size=64,
                      device="cuda", conditions=None, temperature=1.0, reference=None):
    import torch

    from amp_challenge_2027.assay_conditioning import collate_assay_conditions, condition_support
    from amp_challenge_2027.model import load_model
    from amp_challenge_2027.tokenizer import RESIDUE_TO_ID, decode

    if not label or draws < 1 or batch_size < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Require label, positive draws/batch size and finite positive temperature")
    if str(device).startswith("cuda"):
        # cuBLAS reads this on initialization, before any checkpoint reaches CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    checkpoint = Path(checkpoint)
    if (checkpoint / "checkpoint" / "model.pt").exists():
        checkpoint = checkpoint / "checkpoint"
    request = read_request(conditions)
    training = None
    training_root = checkpoint.parent
    if (training_root / "complete.json").exists() and (training_root / "run.json").exists():
        verify(training_root, ["run.json", "checkpoint/model.pt", "checkpoint/config.json"])
        training = json.loads((training_root / "run.json").read_text())
    model, cfg = load_model(checkpoint, map_location=device)
    model = model.to(device).eval()
    if request and cfg.assay_schema is None:
        raise ValueError("This checkpoint has no biological conditioning; omit --conditions for controls")
    inputs = {str(checkpoint / n): sha256(checkpoint / n) for n in ("model.pt", "config.json")}
    if conditions:
        inputs[str(Path(conditions).resolve())] = sha256(conditions)
    if reference:
        inputs[str(Path(reference).resolve())] = sha256(reference)
    encoded = None
    support_warnings = []
    if cfg.assay_schema is not None:
        # Validate before creating outputs or starting an expensive run.
        encoded = collate_assay_conditions([request], cfg.assay_schema, device=device)
        support_warnings = condition_support(request, cfg.assay_schema)
        for warning in support_warnings:
            print(f"[conditional-sample] request support: {warning}", flush=True)
    out = new_output(out, [checkpoint, *([conditions] if conditions else [])])
    rng = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    run = {"kind": "conditional_raw_draws_v1", "label": label, "seed": seed,
           "checkpoint": str(checkpoint.resolve()), "inputs": inputs, "conditions": request,
           "training": training, "condition_support_warnings": support_warnings,
           "reference_sha256": sha256(reference) if reference else None,
           "sampling": {"draws": draws, "batch_size": batch_size, "temperature": temperature,
                        "min_length": MIN_LENGTH, "max_length": MAX_LENGTH,
                        "top_k": None, "top_p": None, "repetition_penalty": 1.0,
                        "decoding": "full_prefix_masked_categorical"},
           "code": source_identity(), "torch": str(torch.__version__), "device": device,
           "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
           "limitations": ["Condition requests are targets, not measured output properties",
                           "No automatic selection or release promotion"]}
    write_json(out / "run.json", run)
    started = time.monotonic()
    rows = []
    with (out / "raw.fasta").open("w") as fasta, torch.inference_mode():
        for start in range(0, draws, batch_size):
            n = min(batch_size, draws - start)
            tokens = torch.full((n, 1), BOS_ID, dtype=torch.long, device=device)
            finished = torch.zeros(n, dtype=torch.bool, device=device)
            batch_conditions = None
            if encoded is not None:
                batch_conditions = collate_assay_conditions([request] * n, cfg.assay_schema, device=device)
            for step in range(MAX_LENGTH + 1):
                logits = model(tokens, assay_conditions=batch_conditions).logits[:, -1] / temperature
                allowed = torch.zeros_like(logits, dtype=torch.bool)
                if step < MAX_LENGTH:
                    allowed[:, list(RESIDUE_TO_ID.values())] = True
                if step >= MIN_LENGTH:
                    allowed[:, EOS_ID] = True
                probs = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
                if not torch.isfinite(probs).all():
                    raise ValueError("Nonfinite generation probabilities")
                nxt = torch.multinomial(probs, 1, generator=rng).squeeze(1)
                nxt = torch.where(finished, PAD_ID, nxt)
                tokens = torch.cat([tokens, nxt[:, None]], 1)
                finished |= nxt == EOS_ID
                if bool(finished.all()):
                    break
            for j, sequence in enumerate([decode(row) for row in tokens.cpu().tolist()]):
                draw_id = start + j + 1
                fasta.write(f">draw_{draw_id}\n{sequence}\n")
                rows.append({"draw_id": draw_id, "sequence": sequence})
            print(f"[conditional-sample] {start + n}/{draws} raw draws", flush=True)
    write_csv(out / "draws.csv", rows)
    refs = {s for _, s in iter_fasta(reference)} if reference else set()
    library = list(dict.fromkeys(r["sequence"] for r in rows if valid_sequence(r["sequence"]) and r["sequence"] not in refs))
    with (out / "library.fasta").open("w") as handle:
        for i, sequence in enumerate(library, 1):
            handle.write(f">candidate_{i}\n{sequence}\n")
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError("Input changed during sampling")
    summary = {"raw_draws": len(rows), "unique_clean_candidates": len(library),
               "elapsed_seconds": time.monotonic() - started, "reference_filter_applied": reference is not None}
    write_json(out / "summary.json", summary)
    seal(out, ["run.json", "raw.fasta", "draws.csv", "library.fasta", "summary.json"])
    return summary


def wilson_interval(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0., center - half), min(1., center + half)]


def mean_distance(sequences, *, maximum=256, seed=2027):
    from Levenshtein import ratio

    if len(sequences) < 2:
        return None
    ids = np.random.default_rng(seed).choice(len(sequences), min(maximum, len(sequences)), replace=False)
    sample = [sequences[i] for i in ids]
    return float(np.mean([1 - ratio(s, t) for i, s in enumerate(sample) for t in sample[:i]]))


def raw_metrics(sequences, activity, risk, panel, reference, *, activity_floor=.8, risk_ceiling=.5):
    """One row per draw; NaN scores are allowed ONLY for invalid sequences."""
    n = len(sequences)
    if n < 1 or not 0 <= activity_floor <= 1 or not 0 <= risk_ceiling <= 1:
        raise ValueError("Invalid raw draw count or probability gates")
    valid = np.array([valid_sequence(s) for s in sequences])
    activity, risk, panel = map(np.asarray, (activity, risk, panel))
    if activity.shape != (n,) or risk.shape != (n,) or panel.shape != (n, len(PANEL_GENERA)):
        raise ValueError("Score shape differs from draw inventory/panel order")
    for values in (activity[valid], risk[valid], panel[valid]):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Invalid predictor probabilities")
    novel = np.array([s not in reference for s in sequences])
    passes = valid & novel & (activity >= activity_floor) & (risk <= risk_ceiling)
    unique_valid = set(s for s, ok in zip(sequences, valid, strict=True) if ok)
    unique_novel = unique_valid - set(reference)
    joint = {s for s, ok in zip(sequences, passes, strict=True) if ok}
    def m(a):
        return float(np.mean(a)) if len(a) else None
    result = {"raw_draws": n, "valid_draws": int(valid.sum()), "unique_valid": len(unique_valid),
              "unique_novel": len(unique_novel), "joint_unique_count": len(joint),
              "joint_unique_yield_per_1000": 1000 * len(joint) / n,
              "valid_unique_novel_fraction": len(unique_novel) / n,
              "raw_joint_pass_fraction": float(passes.mean()),
              "raw_joint_pass_wilson95": wilson_interval(int(passes.sum()), n),
              "raw_activity_mean": m(activity[valid]), "raw_risk_mean": m(risk[valid]),
              "raw_risk_p95": float(np.quantile(risk[valid], .95)) if valid.any() else None,
              "raw_panel_mean": m(panel[valid]),
              "raw_pairwise_distance_256": mean_distance([s for s, ok in zip(sequences, valid, strict=True) if ok]),
              "novelty_definition": "No exact match in the supplied competition reference; top selection separately applies its similarity cap",
              "diversity_scope": "Fixed-seed sample of up to 256 valid draws, retaining duplicates",
              "confidence_interval_scope": "Binomial raw pass fraction at fixed model; not unique yield or biological success",
              "panel_means": {g: m(panel[valid, j]) for j, g in enumerate(PANEL_GENERA)}}
    strata = []
    lengths = np.array([len(s) for s in sequences])
    for lo, hi in ((8, 15), (16, 25), (26, 35), (36, 50)):
        for alo, ahi in ((0., .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)):
            mask = valid & (lengths >= lo) & (lengths <= hi) & (activity >= alo) & (activity < ahi)
            strata.append({"length_min": lo, "length_max": hi, "activity_min": alo,
                           "activity_max_exclusive": ahi, "n": int(mask.sum()),
                           "risk_mean": m(risk[mask])})
    result["risk_strata"] = strata
    return result


class LegacyOracles:
    """Fail closed on missing heads; record the actual backbone revisions."""

    def __init__(self, device="cuda", revision=None):
        from amp_challenge_2027.inference_metadata import load_config
        from amp_challenge_2027.score import ActivityScorer, HemoScorer, PanelScorer

        self.identity = {"kind": "legacy_surrogate_heads", "heads": {}}
        for name, root, stem in (("activity", REWARD_DIR, "classifier"),
                                  ("hemolysis", REWARD_HEMO_DIR, "classifier"),
                                  ("panel", REWARD_DIR, "classifier_panel")):
            config = load_config(root, stem)
            config_path = root / f"{stem}_config.json"
            if not config_path.exists():
                raise ValueError("Require explicit per-artifact inference configs for research scoring")
            self.identity["heads"][name] = {"config": config, "model_sha256": sha256(root / f"{stem}.pt"),
                                            "config_sha256": sha256(config_path)}
        self.activity = ActivityScorer.load(device=device, revision=revision)
        self.risk = HemoScorer.load(device=device, revision=revision)
        self.panel = PanelScorer.load(device=device, revision=revision)
        if any(s is None for s in (self.activity, self.risk, self.panel)):
            raise RuntimeError("All three legacy scoring heads are required; no silent component fallback")
        if self.panel.genera != PANEL_GENERA:
            raise ValueError("Unexpected panel label order")
        self.identity["backbones"] = {n: getattr(s._model.esm.config, "_commit_hash", None)
                                      for n, s in (("activity", self.activity), ("hemolysis", self.risk), ("panel", self.panel))}
        if any(v is None for v in self.identity["backbones"].values()):
            raise ValueError("Cannot establish scoring backbone revision")

    def score(self, sequences):
        return self.activity.score(sequences), self.risk.p_risky(sequences), self.panel._probs(sequences)


def summarize_pairs(rows, baseline, *, minimum_draws=10000):
    """Advancement requires matched complete seed triplets, not a lone good seed."""
    by_label = {}
    for row in rows:
        cells = by_label.setdefault(row["label"], {})
        if row["seed"] in cells:
            raise ValueError("Duplicate label/seed; give condition variants distinct labels")
        cells[row["seed"]] = row
    if baseline not in by_label:
        raise ValueError("Baseline label is absent")
    output = []
    for label, cells in by_label.items():
        if label == baseline:
            continue
        base = by_label[baseline]
        paired_seeds = sorted(set(base) & set(cells))
        checks, differences = [], []
        for seed in paired_seeds:
            a, b = base[seed], cells[seed]
            distance_a, distance_b = a.get("raw_pairwise_distance_256"), b.get("raw_pairwise_distance_256")
            complete = all(x is not None for x in (a.get("raw_activity_mean"), b.get("raw_activity_mean"), distance_a, distance_b))
            passing = complete and (b["joint_unique_yield_per_1000"] > a["joint_unique_yield_per_1000"]
                and b["raw_activity_mean"] >= a["raw_activity_mean"] - .03
                and all(b["panel_means"][g] is not None and a["panel_means"][g] is not None and
                        b["panel_means"][g] >= a["panel_means"][g] - .03 for g in PANEL_GENERA)
                and distance_b >= distance_a - .02
                and b["valid_unique_novel_fraction"] >= a["valid_unique_novel_fraction"] - .05
                and b["raw_draws"] == a["raw_draws"] and b["raw_draws"] >= minimum_draws)
            delta = b["joint_unique_yield_per_1000"] - a["joint_unique_yield_per_1000"]
            differences.append(delta)
            independent_training_seed = b.get("training_seed") == seed and b.get("training_arm") != "frozen"
            checks.append({"seed": seed, "raw_criteria_pass": bool(passing), "joint_yield_delta": delta,
                           "training_seed_verified": independent_training_seed})
        ci = None
        if len(differences) >= 3 and all(c["training_seed_verified"] for c in checks):
            values = np.asarray(differences)
            bootstrap = np.random.default_rng(2027).choice(values, (2000, len(values)), replace=True).mean(1)
            ci = np.quantile(bootstrap, [.025, .975]).tolist()
        output.append({"label": label, "baseline": baseline, "seeds": checks,
                       "paired_seed_mean_yield_delta": float(np.mean(differences)) if differences else None,
                       "paired_seed_bootstrap95": ci,
                       "advance_to_50000": set(paired_seeds) == {42, 43, 44} and all(
                           c["raw_criteria_pass"] and c["training_seed_verified"] for c in checks),
                       "status": "surrogate_only_no_automatic_promotion",
                       "ci_scope": "Descriptive resampling of paired training seeds, not wet-lab confidence"})
    return output


def compare_experiments(runs, out, *, baseline, reference, device="cuda", activity_floor=.8,
                        risk_ceiling=.5, select_top=False, revision=None, scorer=None):
    runs = [Path(p) for p in runs]
    refset = {s for _, s in iter_fasta(reference)}
    refs = sorted(refset)
    if not refs:
        raise ValueError("Reference FASTA is empty")
    inventories = []
    sampling = None
    requests = {}
    profiles = {}
    data_digest = None
    reference_digest = sha256(reference)
    for directory in runs:
        files = verify(directory, ["run.json", "draws.csv", "raw.fasta"])
        run = json.loads((directory / "run.json").read_text())
        if run.get("kind") != "conditional_raw_draws_v1":
            raise ValueError("Expected raw-draw sampling artifacts")
        if run.get("reference_sha256") not in (None, reference_digest):
            raise ValueError("Sampling and comparison reference hashes differ")
        training = run.get("training") or {}
        profile = {k: training.get(k) for k in ("arm", "preset", "model_config", "optimization", "code_hashes")}
        if run["label"] in profiles and profile != profiles[run["label"]]:
            raise ValueError("Training configuration differs across seeds sharing a label")
        profiles[run["label"]] = profile
        digests = {v for k, v in training.get("input_hashes", {}).items() if Path(k).name == "dataset.json"}
        if len(digests) > 1:
            raise ValueError("Ambiguous training dataset identity")
        if digests:
            current_digest = next(iter(digests))
            if data_digest is not None and current_digest != data_digest:
                raise ValueError("Training datasets differ; matched experiments require one prepared partition")
            data_digest = current_digest
        if run["label"] in requests and run["conditions"] != requests[run["label"]]:
            raise ValueError("Condition requests differ across seeds sharing a label")
        requests[run["label"]] = run["conditions"]
        if sampling is not None and run["sampling"] != sampling:
            raise ValueError("Sampling budgets/settings differ; use matched runs")
        sampling = run["sampling"]
        with (directory / "draws.csv").open() as handle:
            draws = list(csv.DictReader(handle))
        if len(draws) != sampling["draws"] or [int(r["draw_id"]) for r in draws] != list(range(1, len(draws) + 1)):
            raise ValueError("Incomplete or reordered draw inventory")
        # iter_fasta drops empty records, so parse the writer's two-line raw format.
        raw_lines = (directory / "raw.fasta").read_text().splitlines()
        expected_lines = [line for row in draws for line in (f">draw_{row['draw_id']}", row["sequence"])]
        if raw_lines != expected_lines:
            raise ValueError("Raw FASTA and draw inventory disagree")
        inventories.append((directory, run, draws, files))
    if len({(run["label"], run["seed"]) for _, run, _, _ in inventories}) != len(inventories):
        raise ValueError("Duplicate label/seed input")
    if baseline not in {r["label"] for _, r, _, _ in inventories}:
        raise ValueError("Baseline label absent")
    out = new_output(out, [*runs, reference])
    scorer = scorer or LegacyOracles(device, revision)
    identity = scorer.identity
    write_json(out / "run.json", {"kind": "conditional_comparison_v1", "code": source_identity(),
               "inputs": {str(p.resolve()): sha256(p / "complete.json") for p in runs},
               "reference_sha256": reference_digest, "dataset_sha256": data_digest,
               "baseline": baseline, "scorers": identity,
               "activity_floor": activity_floor, "risk_ceiling": risk_ceiling, "sampling": sampling,
               "select_top": select_top,
               "limitations": ["Legacy head scores are not measured activity or hemolysis",
                                "Legacy hemolysis head is not a dose-conditioned assay predictor",
                                "Warm starts may have seen holdout sequences before fine-tuning"]})
    rows = []
    output_files = ["run.json"]
    for cell_id, (directory, run, draws, files) in enumerate(inventories):
        dest = out / f"cell-{cell_id:03d}"
        dest.mkdir()
        seqs = [r["sequence"] for r in draws]
        unique = list(dict.fromkeys(s for s in seqs if valid_sequence(s)))
        a, h, p = scorer.score(unique) if unique else (np.empty(0), np.empty(0), np.empty((0, len(PANEL_GENERA))))
        lookup = {s: i for i, s in enumerate(unique)}
        activity, risk = np.full(len(seqs), np.nan), np.full(len(seqs), np.nan)
        panel = np.full((len(seqs), len(PANEL_GENERA)), np.nan)
        for i, s in enumerate(seqs):
            if s in lookup:
                j = lookup[s]
                activity[i], risk[i], panel[i] = a[j], h[j], p[j]
        metrics = raw_metrics(seqs, activity, risk, panel, refset, activity_floor=activity_floor, risk_ceiling=risk_ceiling)
        training = run.get("training") or {}
        checkpoint_files = {Path(k).name: v for k, v in run.get("inputs", {}).items()
                            if Path(k).name in {"model.pt", "config.json"}}
        checkpoint_identity = (hashlib.sha256(json.dumps(checkpoint_files, sort_keys=True).encode()).hexdigest()
                               if set(checkpoint_files) == {"model.pt", "config.json"} else None)
        metrics.update(label=run["label"], seed=run["seed"], conditions=run["conditions"],
                       training_seed=training.get("seed"), training_arm=training.get("arm"),
                       architecture_preset=training.get("preset"),
                       checkpoint_sha256=checkpoint_identity,
                       condition_support_warnings=run.get("condition_support_warnings", []))
        score_rows = []
        for i, s in enumerate(seqs):
            ok = valid_sequence(s)
            row = {"draw_id": i + 1, "sequence": s, "valid": ok, "exact_reference": s in refset,
                   "activity_score": float(activity[i]) if ok else None,
                   "hemolysis_risk_score": float(risk[i]) if ok else None}
            row.update({f"panel:{g}": float(panel[i, j]) if ok else None for j, g in enumerate(PANEL_GENERA)})
            score_rows.append(row)
        write_csv(dest / "scores.csv", score_rows)
        if select_top:
            metrics["top100"] = select_unchanged(seqs, activity, risk, panel, refs, dest, device, seed=run["seed"])
            output_files.extend(str(p.relative_to(out)) for p in dest.iterdir()
                                if p.name.startswith("top") or p.name in {"library.fasta", "selection.json"})
        write_json(dest / "metrics.json", metrics)
        output_files.extend([str((dest / "scores.csv").relative_to(out)), str((dest / "metrics.json").relative_to(out))])
        rows.append(metrics)
        if verify(directory, list(files)) != files:
            raise ValueError("Sampling artifact changed while comparing")
        print(f"[conditional-compare] {run['label']} seed{run['seed']}: {metrics['joint_unique_yield_per_1000']:.2f} joint unique / 1000", flush=True)
    decisions = summarize_pairs(rows, baseline)
    if sha256(reference) != reference_digest:
        raise ValueError("Reference changed during comparison")
    # Top tolerances are evaluated only when both arms actually contain 100 members.
    base = {r["seed"]: r for r in rows if r["label"] == baseline}
    for decision in decisions:
        top_checks = []
        for row in rows:
            if row["label"] != decision["label"] or row["seed"] not in base:
                continue
            a, b = base[row["seed"]].get("top100"), row.get("top100")
            if not a or not b:
                continue
            passed = (a["n"] == b["n"] == 100 and a.get("library_complete") and b.get("library_complete")
                      and b["activity_mean"] >= a["activity_mean"] - .03 and b["risk_mean"] < a["risk_mean"])
            passed = passed and b["pairwise_distance_mean"] >= a["pairwise_distance_mean"] - .02
            passed = passed and all(b["panel_means"][g] >= a["panel_means"][g] - .03 for g in PANEL_GENERA)
            top_checks.append({"seed": row["seed"], "passes": bool(passed)})
        decision["top100_checks"] = top_checks
        decision["top100_tolerances_pass"] = (all(r["passes"] for r in top_checks)
            if {r["seed"] for r in top_checks} == {42, 43, 44} else None)
    from amp_challenge_2027.conditional_reporting import ablation_contrasts

    contrasts = ablation_contrasts(rows)
    write_json(out / "comparison.json", {"rows": rows, "decisions": decisions, "ablation_contrasts": contrasts})
    summary_rows = [{k: r[k] for k in ("label", "seed", "raw_draws", "joint_unique_yield_per_1000",
                    "valid_unique_novel_fraction", "raw_activity_mean", "raw_risk_mean", "raw_risk_p95",
                    "raw_panel_mean", "raw_pairwise_distance_256")} for r in rows]
    write_csv(out / "comparison.csv", summary_rows)
    report = ["# Conditional generator comparison", "", "Exploratory legacy-surrogate evaluation; no biological safety claim or automatic promotion.", "",
              "| Model | Seed | Raw draws | Joint unique / 1000 | Activity | Risk |", "|---|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    for r in rows:
        report.append(f"| {r['label']} | {r['seed']} | {r['raw_draws']} | {fmt(r['joint_unique_yield_per_1000'])} | {fmt(r['raw_activity_mean'])} | {fmt(r['raw_risk_mean'])} |")
    report += ["", "Raw denominators include duplicates, reference matches and invalid draws. Dose-specific biological adherence is not established by these heads.", ""]
    for d in decisions:
        report.append(f"- {d['label']}: advance to 50,000 comparison = {d['advance_to_50000']}; top-100 tolerances = {d['top100_tolerances_pass']}.")
    if contrasts:
        report.extend(["", "## Separate ablation contrasts", "",
                       "Candidate minus baseline; these describe surrogate changes, not established biological effects.", "",
                       "| Contrast | Candidate / baseline | Seeds | Joint yield delta / 1000 | Activity delta | Risk delta |",
                       "|---|---|---|---:|---:|---:|"])
        for contrast in contrasts:
            delta = contrast["mean_deltas"]
            report.append(f"| {contrast['kind']} | {contrast['candidate_label']} / {contrast['baseline_label']} | "
                          f"{', '.join(map(str, contrast['seeds']))} | {fmt(delta.get('joint_unique_yield_per_1000'))} | "
                          f"{fmt(delta.get('raw_activity_mean'))} | {fmt(delta.get('raw_risk_mean'))} |")
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    seal(out, [*output_files, "comparison.json", "comparison.csv", "REPORT.md"])
    return {"rows": rows, "decisions": decisions}


def select_unchanged(sequences, activity, risk, panel, reference, out, device, *, seed=42):
    """Production weights/normalization/selector; hemolysis remains audit-only."""
    from amp_challenge_2027.config import MDR_PANEL_GENERA
    from amp_challenge_2027.pipeline import DEFAULT_WEIGHTS
    from amp_challenge_2027.score import CompositeScorer, ConformityScorer, PrecisionProxyScorer
    from amp_challenge_2027.select import select_library_and_top

    references = set(reference)
    indices, seen = [], set()
    for i, s in enumerate(sequences):
        if valid_sequence(s) and s not in references and s not in seen:
            indices.append(i)
            seen.add(s)
        if len(indices) == 50000:
            break
    clean = [sequences[i] for i in indices]
    if not clean:
        return {"n": 0, "library_size": 0, "library_complete": False, "activity_mean": None, "risk_mean": None,
                "pairwise_distance_mean": None, "panel_means": {g: None for g in PANEL_GENERA}}
    precision = PrecisionProxyScorer(reference, esm_model="facebook/esm2_t6_8M_UR50D", device=device, cache=False)
    # The existing scorer writes a shared cache even with cache=False. Populate
    # its in-memory reference here so this research run cannot overwrite it.
    precision._ref_emb = precision._embed(sorted(set(reference)))
    parts = {"conformity": ConformityScorer(reference, sample=12000, seed=seed).score(clean),
             "activity": activity[indices], "breadth": (panel[indices] > .5).mean(1).astype(np.float32),
             "mdr": (panel[indices][:, [j for j, g in enumerate(PANEL_GENERA) if g in MDR_PANEL_GENERA]] > .5).mean(1).astype(np.float32),
             "precision": precision.score(clean)}
    if any(not np.isfinite(values).all() for values in parts.values()):
        raise ValueError("Nonfinite production selection scores")
    revision = getattr(precision._model.config, "_commit_hash", None)
    if revision is None:
        raise ValueError("Cannot establish precision scoring backbone revision")
    write_json(out / "selection.json", {"weights": DEFAULT_WEIGHTS, "seed": seed,
               "max_novelty_candidates": 2000, "conformity_sample": 12000,
               "precision_model": precision._esm_model, "precision_revision": revision,
               "library_policy": "first 50000 valid unique non-reference draws", "hemolysis_in_objective": False})
    scorer = CompositeScorer([(name, DEFAULT_WEIGHTS[name], lambda _seqs, v=values: v) for name, values in parts.items()])
    scores, _ = scorer.score(clean)
    result = select_library_and_top(clean, reference_set=references, scores=scores, seed=seed, max_novelty_candidates=2000)
    look = {s: indices[j] for j, s in enumerate(clean)}
    top_ids = [look[s] for s in result.top]
    for name, seqs in (("library", result.library), ("top", result.top)):
        with (out / f"{name}.fasta").open("w") as handle:
            for i, s in enumerate(seqs, 1):
                handle.write(f">{name}_{i}\n{s}\n")
    write_csv(out / "top_scores.csv", [{"rank": j + 1, "sequence": sequences[i], "activity_score": float(activity[i]),
               "hemolysis_risk_score": float(risk[i])} for j, i in enumerate(top_ids)],
              fields=["rank", "sequence", "activity_score", "hemolysis_risk_score"])
    return {"n": len(top_ids), "library_size": len(result.library), "library_complete": len(result.library) == 50000,
            "activity_mean": float(activity[top_ids].mean()) if top_ids else None,
            "risk_mean": float(risk[top_ids].mean()) if top_ids else None,
            "pairwise_distance_mean": mean_distance(result.top, maximum=100),
            "panel_means": {g: float(panel[top_ids, j].mean()) if top_ids else None for j, g in enumerate(PANEL_GENERA)}}
