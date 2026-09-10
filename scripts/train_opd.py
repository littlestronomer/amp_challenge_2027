"""Frozen specialist teacher, baseline-initialized student; no reward-head training."""
import argparse
import copy
import json
import math
from pathlib import Path

from experiment_utils import code_identity, mark_files, sha256, write_json, write_summary
from scale_validation import checked, sources
from selection_cache import separate_output


def optimize(student, teacher, baseline, args, encoder=None):
    import numpy as np
    import torch
    from pilot_grpo import assert_strict_runtime, strict_runtime

    from amp_challenge_2027.grpo import distributions, rollout
    from amp_challenge_2027.opd import CoverageFeatures, coverage_cost, coverage_loss, forward_kl

    strict_runtime(args.seed)
    student.train(False).requires_grad_(True)
    teacher.eval().requires_grad_(False)
    baseline.eval().requires_grad_(False)
    student_rng = torch.Generator(device=args.device).manual_seed(50000+args.seed)
    anchor_rng = torch.Generator(device=args.device).manual_seed(51000+args.seed)
    probe_rng = torch.Generator(device=args.device).manual_seed(52000+args.seed)
    probe, probe_sequences = rollout(baseline, 32, probe_rng, args.device)
    with torch.no_grad():
        probe_prior = distributions(baseline, probe).detach()
    features = None
    if args.variant == "coverage":
        if encoder is None:
            raise ValueError("Coverage variant requires its frozen training encoder")
        target_rng = torch.Generator(device=args.device).manual_seed(53000+args.seed)
        target_sequences = []
        while len(target_sequences) < args.coverage_draws:
            _, batch = rollout(baseline, min(32, args.coverage_draws-len(target_sequences)), target_rng, args.device)
            target_sequences.extend(batch)
        features = CoverageFeatures(encoder, target_sequences, 54000+args.seed)
        np.savez(args.out / "coverage_features.npz", projection=features.projection, phase=features.phase,
                 target=features.target, bandwidth=features.bandwidth)
        write_summary(args.out / "coverage_reference.csv", [{"sequence": s} for s in target_sequences])
    write_summary(args.out / "guard_reference.csv", [{"sequence": s} for s in probe_sequences])
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0)
    rows, history, updates, stopped, multiplier = [], [], 0, False, .1
    for step in range(args.steps):
        assert_strict_runtime()
        tokens, seqs = rollout(student, args.batch_size, student_rng, args.device)
        with torch.no_grad():
            target = distributions(teacher, tokens).detach()
        current = distributions(student, tokens)
        specialist_loss = forward_kl(target, current, tokens)
        loss = specialist_loss
        anchor_value, coverage_value, discrepancy = 0., 0., None
        if args.variant != "specialist":
            anchor_tokens, anchor_seqs = rollout(baseline, args.batch_size, anchor_rng, args.device)
            with torch.no_grad():
                anchor_target = distributions(baseline, anchor_tokens).detach()
            preservation = forward_kl(anchor_target, distributions(student, anchor_tokens), anchor_tokens)
            loss = loss + args.anchor_weight*preservation
            anchor_value = float(preservation.detach())
            rows.extend({"step": step, "source": "baseline_anchor", "sequence": s} for s in anchor_seqs)
        used_multiplier = multiplier
        if features is not None:
            costs, discrepancy = coverage_cost(features.transform(seqs), features.target)
            penalty = coverage_loss(current, tokens, torch.as_tensor(costs, device=args.device))
            loss = loss + used_multiplier*penalty
            coverage_value = float(penalty.detach())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite distillation objective")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        before = [p.detach().clone() for p in student.parameters()]
        optimizer.step()
        with torch.no_grad():
            drift = forward_kl(probe_prior, distributions(student, probe), probe)
            if not torch.isfinite(drift) or float(drift) > args.kl_limit:
                for p, old in zip(student.parameters(), before, strict=True):
                    p.copy_(old)
                stopped = True
            else:
                updates += 1
        if discrepancy is not None:
            multiplier = min(10., max(0., multiplier+args.dual_lr*(discrepancy-args.coverage_target)))
        rows.extend({"step": step, "source": "student_on_policy", "sequence": s} for s in seqs)
        history.append({"step": step, "specialist_forward_kl": float(specialist_loss.detach()),
                        "anchor_forward_kl": anchor_value, "coverage_policy_loss": coverage_value,
                        "coverage_discrepancy": discrepancy, "coverage_multiplier_used": used_multiplier if features else 0.,
                        "probe_forward_kl_attempted": float(drift) if torch.isfinite(drift) else None,
                        "update_accepted": not stopped, "kl_stopped": stopped})
        write_summary(args.out / "history.csv", history)
        print(f"[opd] {args.variant} step={step} teacher_kl={float(specialist_loss.detach()):.6f} anchor_kl={anchor_value:.6f} stopped={stopped}", flush=True)
        if stopped:
            break
    write_summary(args.out / "training_samples.csv", rows)
    return {"accepted_updates": updates, "attempted_updates": len(history), "kl_stopped": stopped,
            "student_draws": len(history)*args.batch_size,
            "anchor_draws": len(history)*args.batch_size if args.variant != "specialist" else 0,
            "guard_draws": 32, "coverage_reference_draws": args.coverage_draws if features else 0,
            "total_training_draws": len(history)*args.batch_size*(1 if args.variant == "specialist" else 2)
                                    + 32 + (args.coverage_draws if features else 0),
            "reward_head_calls_during_training": 0, "evaluation_head_calls_during_training": 0,
            "teacher_training_cost_included": False, "deployment_approved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("sweep_results"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", choices=("specialist", "anchored", "coverage"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--anchor-weight", type=float, default=1.)
    parser.add_argument("--kl-limit", type=float, default=.05)
    parser.add_argument("--coverage-draws", type=int, default=1024)
    parser.add_argument("--coverage-target", type=float, default=.01)
    parser.add_argument("--dual-lr", type=float, default=.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.steps <= 2000 and 2 <= args.batch_size <= 64 and 32 <= args.coverage_draws <= 4096 and
            all(math.isfinite(x) for x in (args.lr, args.anchor_weight, args.kl_limit, args.coverage_target, args.dual_lr)) and
            0 < args.lr <= 1e-5 and 0 < args.anchor_weight <= 10 and 0 < args.kl_limit <= .1 and
            0 <= args.coverage_target <= .1 and 0 < args.dual_lr <= 10):
        raise ValueError("Invalid or unbounded OPD parameters")
    pinned = sources(args.pilot_root)
    teacher = pinned["cells"][f"grpo/seed{args.seed}"]
    checkpoint = Path(pinned["cells"][f"baseline/seed{args.seed}"]["checkpoint"])
    separate_output(args.out, [checkpoint, Path(pinned["evaluator"])] + [Path(p) for p in pinned["common"]["inputs"]] +
                    [Path(c["root"]) for c in pinned["cells"].values() if "root" in c])
    if args.out.exists():
        raise ValueError("Use a new output directory; OPD does not resume partial training")
    if json.loads((checkpoint / "config.json").read_text()) != json.loads((Path(teacher["checkpoint"]) / "config.json").read_text()):
        raise ValueError("Teacher and baseline generator configurations differ")
    run = {"kind": "anchored_opd_v1", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "sources": pinned, "code": code_identity(), "teacher_cell": f"grpo/seed{args.seed}",
           "objective": "forward KL specialist->student on detached student rollouts; optional forward KL baseline->student on baseline rollouts; optional leave-one-out moment coverage policy gradient",
           "sampling": "full categorical masked residues8..50; dropout off; one update per fresh batch",
           "coverage_protocol": {"encoder": "pinned35M", "rff_dimensions": 128, "descriptors": "length/50 and20 AA fractions",
                                 "initial_multiplier": .1, "max_multiplier": 10., "target": args.coverage_target},
           "limitations": ["Same-sized specialist transfer, not compression or proven biological improvement",
                           "Coverage regularizer is a baseline raw-distribution proxy, not an official FBD/MMD constraint",
                           "Guard uses32 fixed baseline rollouts, not a global policy-divergence guarantee",
                           "No evaluation predictor used in training; no model selection on evaluation scores",
                           "Anchor/coverage variants have extra draw and compute costs; original teacher training is additional"]}
    if args.list:
        print(json.dumps(run, indent=2))
        return
    import torch
    from pilot_grpo import strict_runtime

    from amp_challenge_2027.model import load_model, save_model

    strict_runtime(args.seed)
    student, cfg = load_model(checkpoint, map_location=args.device)
    student = student.to(args.device).float().eval()
    baseline = copy.deepcopy(student).requires_grad_(False)
    specialist, _ = load_model(Path(teacher["checkpoint"]), map_location=args.device)
    specialist = specialist.to(args.device).float().eval().requires_grad_(False)
    encoder = None
    if args.variant == "coverage":
        from amp_challenge_2027.reward_benchmark import FrozenEncoder

        encoder = FrozenEncoder(pinned["common"]["configs"]["activity"]["esm_model"], pinned["common"]["revision"], device=args.device)
    run["runtime"] = strict_runtime(args.seed)
    run["runtime"]["device"] = args.device
    if str(args.device).startswith("cuda"):
        run["runtime"]["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    args.out.mkdir(parents=True)
    write_json(args.out / "run.json", run)
    status = optimize(student, specialist, baseline, args, encoder=encoder)
    save_model(student, args.out / "policy", config=cfg)
    if sources(args.pilot_root) != pinned:
        raise ValueError("Pinned inputs changed during training")
    checked(Path(teacher["root"]), required=("policy/model.pt", "policy/config.json"))
    status["teacher_marker_sha256"] = sha256(Path(teacher["root"]) / "complete.json")
    write_json(args.out / "status.json", status)
    mark_files(args.out, "complete.json", [str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file()])
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
