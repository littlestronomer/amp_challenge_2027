"""Bounded exploratory GRPO with a matched draw/scoring-budget selection baseline."""
import argparse
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from compare_top100 import checked_stage
from experiment_utils import code_identity, mark_files, sha256, write_json, write_summary
from selection_cache import separate_output

from amp_challenge_2027.config import REWARD_DIR, REWARD_HEMO_DIR
from amp_challenge_2027.data import iter_fasta
from amp_challenge_2027.inference_metadata import load_config


def assert_strict_runtime():
    import torch

    if (not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
            or not torch.backends.cuda.math_sdp_enabled() or torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled() or torch.backends.cuda.cudnn_sdp_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError("GRPO deterministic execution settings changed; refusing to continue")


def strict_runtime(seed):
    """Reassert strict settings AFTER every seed helper; avoid fused attention backward."""
    import torch

    from amp_challenge_2027.training import enable_determinism

    enable_determinism(seed)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    assert_strict_runtime()
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "attention": "math_only", "deterministic_algorithms": True, "warn_only": False,
            "tf32": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def rewards(sequences, activity, risk, reference, floor=.6):
    if any(np.asarray(x).shape != (len(sequences),) or not np.isfinite(x).all() or
           (np.asarray(x) < 0).any() or (np.asarray(x) > 1).any() for x in (activity, risk)):
        raise ValueError("Invalid scorer output")
    counts = Counter(sequences)
    penalty = np.array([float(s in reference) + float(counts[s] > 1) for s in sequences])
    return activity - risk - 2 * np.maximum(floor - activity, 0) - penalty


def select(sequences, activity, risk, reference, top, safety=True):
    eligible = [i for i, s in enumerate(sequences) if s not in reference and activity[i] >= .6]
    eligible.sort(key=lambda i: (-(activity[i] - risk[i] if safety else activity[i]), sequences[i], i))
    selected, seen = [], set()
    for i in eligible:
        if sequences[i] not in seen:
            selected.append(i)
            seen.add(sequences[i])
        if len(selected) == top:
            break
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Exported config.json/model.pt generator")
    parser.add_argument("--parity", type=Path, default=Path("sweep_results/inference-probes-v1/report.json"))
    parser.add_argument("--evaluator", type=Path, default=Path("sweep_results/label-ablation-fit-v1"))
    parser.add_argument("--reference", type=Path, default=Path("data/antibacterial.fasta"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-draws", type=int, default=2048)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-limit", type=float, default=.05)
    parser.add_argument("--method", choices=("grpo", "raft"), default="grpo")
    parser.add_argument("--raw-evaluation", action="store_true", help="Score all fresh draws with the evaluation-only ensemble")
    parser.add_argument("--activity-floor", type=float, default=.6)
    parser.add_argument("--risk-ceiling", type=float, default=.5)
    parser.add_argument("--raft-draws", type=int, default=1024)
    parser.add_argument("--raft-retain", type=int, default=128)
    parser.add_argument("--raft-min-retain", type=int, default=8)
    parser.add_argument("--raft-replay", type=int, default=256)
    parser.add_argument("--raft-epochs", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.steps <= 100 and 1 <= args.groups <= 8 and 2 <= args.group_size <= 16 and
            args.eval_draws >= args.top > 0 and math.isfinite(args.lr) and 0 < args.lr <= 1e-5 and
            math.isfinite(args.kl_limit) and 0 < args.kl_limit <= .1):
        raise ValueError("Invalid or unbounded pilot parameters")
    if not (0 < args.activity_floor < 1 and 0 < args.risk_ceiling < 1 and
            1 <= args.raft_min_retain <= args.raft_retain <= args.raft_draws <= 8192 and
            1 <= args.raft_replay <= 2048 and 1 <= args.raft_epochs <= 4):
        raise ValueError("Invalid generator improvement parameters")
    if args.method == "raft":
        args.raw_evaluation = True
    separate_output(args.out, [args.checkpoint, args.parity.parent, args.evaluator, args.reference])
    if args.out.exists():
        raise ValueError("Use a new output directory; pilot does not resume partial training")
    parity = json.loads(args.parity.read_text())
    configs, inputs = {}, {str(p.resolve()): sha256(p) for p in (args.parity, args.reference, args.checkpoint / "model.pt", args.checkpoint / "config.json")}
    revisions = set()
    for name, root in (("activity", REWARD_DIR), ("hemolysis", REWARD_HEMO_DIR)):
        head = root / "classifier.pt"
        cfg = load_config(root, "classifier")
        item = parity["tasks"][name]
        if not item["probe_verified"] or item["head_sha256"] != sha256(head) or item["config"] != cfg:
            raise ValueError("Reward differs from verified parity artifact")
        revisions.add(item["revision"])
        configs[name] = cfg
        inputs[str(head.resolve())] = sha256(head)
    if len(revisions) != 1:
        raise ValueError("Reward backbone revisions differ")
    revision = revisions.pop()
    evaluator_marker = checked_stage(args.evaluator, "complete.json", {"run.json", "model_inventory.json"})
    erun = json.loads((args.evaluator / "run.json").read_text())
    if erun.get("kind") != "paired_label_ablation_training_v1" or erun["manifest"]["revision"] != revision:
        raise ValueError("Evaluator run/revision differs")
    if (erun["manifest"]["backbone"] != configs["activity"]["esm_model"] or
            erun["manifest"]["protocol"]["architectures"] != ["mlp"] or
            erun["manifest"]["protocol"]["training_seeds"] != [42, 43, 44]):
        raise ValueError("Evaluator architecture, backbone or seed inventory differs")
    inventory = json.loads((args.evaluator / "model_inventory.json").read_text())
    evaluator_paths = []
    for seed in (42, 43, 44):
        cell = f"hemolysis/original/seed{seed}"
        dest = args.evaluator / cell
        checked_stage(dest, "complete.json", {"head.pt"})
        if sha256(dest / "complete.json") != inventory[cell]:
            raise ValueError("Evaluator member differs")
        evaluator_paths.append(dest / "head.pt")
    estate = args.evaluator / "hemolysis/original"
    checked_stage(estate, "complete.json", {"calibration.json"})
    if sha256(estate / "complete.json") != inventory["hemolysis/original"]:
        raise ValueError("Evaluator calibration differs")
    temperature = json.loads((estate / "calibration.json").read_text())["temperature"]
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid evaluator calibration")
    for p in evaluator_paths + [estate / "calibration.json"]:
        inputs[str(p.resolve())] = sha256(p)
    manifest = {"kind": "grpo_pilot_v2", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "inputs": inputs, "configs": configs, "revision": revision, "evaluator_marker": evaluator_marker,
                "code": code_identity(), "objective": "activity-risk-2*relu(.6-activity)-batch_duplicates-exact_reference",
                "sampling": "temperature1 full masked categorical; residues8..50; forced EOS at50; default checkpoint conditioning",
                "kl_beta": .05, "clip": .2, "update_epochs": 2,
                "determinism_policy": "strict, math-only SDPA, TF32 disabled; same environment reproducibility only",
                "limitations": ["Proxy optimization, not experimental hemolysis or safety",
                                "Evaluator shares backbone/data lineage; not independent biological validation",
                                "Unconditional BOS groups; not prompt-conditioned task GRPO",
                                "Exact-reference novelty only; not submission novelty validation",
                                "No existing composite-selector reproduction; activity-only diagnostic control",
                                "Selection baseline matches total reward-scored draws; not GPU training FLOPs",
                                "Default one seed is exploratory; no deployment promotion"]}
    if args.raw_evaluation:
        manifest["kind"] = "generator_improvement_pilot_v1"
        manifest["raw_yield_definition"] = "unique non-reference sequences passing fixed probability gates per ALL raw draws"
        manifest["limitations"].extend(["Probability gates are exploratory, not validated biological thresholds",
                                        "Diversity diagnostic uses first256 draws, not full family coverage"])
    if args.method == "raft":
        manifest["objective"] = "selected masked NLL + .25 initial-policy replay NLL + .05 sampled-prefix KL"
        manifest["teaching_policy"] = "activity/risk gates; rank activity-risk; greedy .8 similarity exclusion; 1.5x raw length-bin caps"
        manifest["update_epochs"] = args.raft_epochs
        manifest["clip"] = None
        manifest["limitations"].remove("Unconditional BOS groups; not prompt-conditioned task GRPO")
        manifest["limitations"].append("Replay costs extra unscored initial-policy draws; no token-distribution teacher distillation")
    if args.list:
        print(json.dumps(manifest, indent=2))
        return
    import torch

    from amp_challenge_2027.grpo import advantages, distributions, objective, rollout
    from amp_challenge_2027.model import load_model, save_model
    from amp_challenge_2027.reward_benchmark import FrozenEncoder, build_head
    strict_runtime(args.seed)
    encoder = FrozenEncoder(configs["activity"]["esm_model"], revision, device=args.device)
    heads = {}
    for name, root in (("activity", REWARD_DIR), ("hemolysis", REWARD_HEMO_DIR)):
        head = build_head(480, 1, "mlp").to(args.device).eval()
        head.load_state_dict(torch.load(root / "classifier.pt", map_location=args.device, weights_only=True), strict=True)
        head.requires_grad_(False)
        heads[name] = head

    def score(sequences):
        features = torch.as_tensor(encoder.encode(sequences, 64), device=args.device)
        with torch.no_grad():
            return tuple(torch.sigmoid(heads[n](features).squeeze(-1) / configs[n]["temperature"]).cpu().numpy() for n in ("activity", "hemolysis"))

    policy, cfg = load_model(args.checkpoint, map_location=args.device)
    policy = policy.to(args.device).float().eval()
    frozen = copy.deepcopy(policy).requires_grad_(False).eval()
    # FrozenEncoder initializes its own RNG; reset before policy optimization.
    manifest["runtime"] = strict_runtime(args.seed)
    manifest["runtime"]["device"] = str(args.device)
    if str(args.device).startswith("cuda"):
        manifest["runtime"]["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0)
    reference = {seq for _, seq in iter_fasta(args.reference)}
    if not reference:
        raise ValueError("Empty reference")
    args.out.mkdir(parents=True)
    write_json(args.out / "run.json", manifest)
    history, training_rows, draw_count, stopped = [], [], 0, False
    extra_status = {}
    if args.method == "raft":
        from generator_improvement import train_raft

        training_rows, draw_count, stopped, extra_status = train_raft(policy, frozen, score, reference, args)
    for step in range(args.steps if args.method == "grpo" else 0):
        assert_strict_runtime()
        tokens, sequences = rollout(policy, args.groups * args.group_size, rng, args.device)
        activity, risk = score(sequences)
        reward = rewards(sequences, activity, risk, reference)
        advantage = advantages(torch.as_tensor(reward, dtype=torch.float32, device=args.device), args.group_size)
        draw_count += len(sequences)
        training_rows.extend({"step": step, "sequence": s, "activity": float(a), "risk": float(h), "reward": float(r)} for s, a, h, r in zip(sequences, activity, risk, reward, strict=True))
        with torch.no_grad():
            old, ref = distributions(policy, tokens).detach(), distributions(frozen, tokens).detach()
        for update in range(2):
            assert_strict_runtime()
            current = distributions(policy, tokens)
            loss, kl = objective(current, old, ref, tokens, advantage)
            if not torch.isfinite(loss) or not torch.isfinite(kl):
                raise ValueError("Nonfinite policy objective")
            if float(kl.detach()) > args.kl_limit:
                stopped = True
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            before = [p.detach().clone() for p in policy.parameters()]
            optimizer.step()
            with torch.no_grad():
                _, post_kl = objective(distributions(policy, tokens), old, ref, tokens, advantage)
                if not torch.isfinite(post_kl) or float(post_kl) > args.kl_limit:
                    for parameter, saved in zip(policy.parameters(), before, strict=True):
                        parameter.copy_(saved)
                    stopped = True
            if stopped:
                break
        history.append({"step": step, "reward_mean": float(reward.mean()), "activity_mean": float(activity.mean()),
                        "risk_mean": float(risk.mean()), "unique_fraction": len(set(sequences))/len(sequences),
                        "sampled_prefix_kl": float(kl.detach()), "kl_stop": stopped})
        print(f"[grpo] step={step} reward={reward.mean():.4f} risk={risk.mean():.4f} kl={float(kl.detach()):.5f}", flush=True)
        write_summary(args.out / "history.csv", history)
        if stopped:
            break
    save_model(policy, args.out / "policy", config=cfg)

    def pool(model, count, seed):
        assert_strict_runtime()
        random = torch.Generator(device=args.device).manual_seed(seed)
        result = []
        while len(result) < count:
            _, seqs = rollout(model, min(32, count-len(result)), random, args.device)
            result.extend(seqs)
        a, h = score(result)
        return result, a, h

    baseline = pool(frozen, args.eval_draws + draw_count, args.seed + 10000)
    optimized = pool(policy, args.eval_draws, args.seed + 10000)
    # Evaluation-only heads are loaded after policy updates, never called for reward.
    evaluators = []
    for path in evaluator_paths:
        head = build_head(480, 1, "mlp").to(args.device).eval().requires_grad_(False)
        head.load_state_dict(torch.load(path, map_location=args.device, weights_only=True), strict=True)
        evaluators.append(head)
    summaries = []
    optimized_arm = "raft_selection" if args.method == "raft" else "grpo_selection"
    baseline_external = None
    for name, (seqs, a, h), safety in (("baseline_activity_control", tuple(x[:args.eval_draws] for x in baseline), False),
                                      ("baseline_selection_equal_eval", tuple(x[:args.eval_draws] for x in baseline), True),
                                      ("baseline_selection_matched_total", baseline, True), (optimized_arm, optimized, True)):
        ids = select(seqs, a, h, reference, args.top, safety)
        selected = [seqs[i] for i in ids]
        external = []
        raw = {}
        pool_external = None
        if args.raw_evaluation:
            from generator_improvement import raw_metrics

            # Score baseline once, then slice its prefix for equal-draw controls.
            if name.startswith("baseline") and baseline_external is not None:
                pool_external = baseline_external[:len(seqs)]
            else:
                target = baseline[0] if name.startswith("baseline") else seqs
                values = []
                for start in range(0, len(target), 64):
                    features = torch.as_tensor(encoder.encode(target[start:start+64], 64), device=args.device)
                    with torch.no_grad():
                        logits = torch.stack([head(features) for head in evaluators]).mean(0).squeeze(-1)
                        values.extend(torch.sigmoid(logits / temperature).cpu().tolist())
                if name.startswith("baseline"):
                    baseline_external = values
                pool_external = values[:len(seqs)]
            external = [pool_external[i] for i in ids]
            raw = raw_metrics(seqs, a, h, pool_external, reference, args.activity_floor, args.risk_ceiling)
        elif selected:
            features = torch.as_tensor(encoder.encode(selected, 64), device=args.device)
            with torch.no_grad():
                logits = torch.stack([head(features) for head in evaluators]).mean(0).squeeze(-1)
                external = torch.sigmoid(logits / temperature).cpu().tolist()
        table = [{"sequence": seqs[i], "activity": float(a[i]), "reward_risk": float(h[i]), "evaluation_risk": external[j]} for j, i in enumerate(ids)]
        write_summary(args.out / f"{name}_pool.csv", [{"sequence": s, "activity": float(a[i]), "risk": float(h[i]),
                      **({"evaluation_risk": pool_external[i]} if pool_external is not None else {})} for i, s in enumerate(seqs)])
        if table:
            write_summary(args.out / f"{name}_selected.csv", table)
        from Levenshtein import ratio

        distances = [1 - ratio(s, t) for i, s in enumerate(selected) for t in selected[i+1:]]
        summaries.append({"arm": name, "draws": len(seqs), "selected": len(ids), "selection_complete": len(ids) == args.top,
                          "unique_fraction": len(set(seqs))/len(seqs),
                          "pool_activity_mean": float(a.mean()), "pool_risk_mean": float(h.mean()),
                          "selected_mean_length": float(np.mean([len(s) for s in selected])) if selected else None,
                          "selected_mean_pairwise_distance": float(np.mean(distances)) if distances else None,
                          **raw, **{key: float(np.mean([r[key] for r in table])) if table else None for key in ("activity", "reward_risk", "evaluation_risk")}})
    if any(sha256(Path(p)) != digest for p, digest in inputs.items()):
        raise ValueError("Pinned input changed during pilot")
    write_summary(args.out / "training_samples.csv", training_rows)
    write_summary(args.out / "summary.csv", summaries)
    write_json(args.out / "status.json", {"training_draws": draw_count, "kl_stopped": stopped, "deployment_approved": False, **extra_status})
    mark_files(args.out, "complete.json", [str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file()])
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
