"""Diverse on-policy reward-ranked fine-tuning and fresh-draw diagnostics.

This is selected-sample maximum likelihood, not token-distribution distillation.
Evaluation-only predictors must never be supplied to the training functions.
"""
import math
from collections import Counter

import numpy as np


def validate_scores(sequences, *scores):
    if not sequences or any(np.asarray(x).shape != (len(sequences),) or
                            not np.isfinite(x).all() or (np.asarray(x) < 0).any() or
                            (np.asarray(x) > 1).any() for x in scores):
        raise ValueError("Empty pool or invalid probabilities")


def teaching_ids(sequences, activity, risk, reference, retain, floor=.6, ceiling=.5):
    """Greedy .8 similarity exclusion and capped length bins; never relax gates."""
    from Levenshtein import ratio

    validate_scores(sequences, activity, risk)
    bins = Counter(min(len(s) // 10, 5) for s in sequences)
    caps = {b: max(1, math.ceil(retain * n / len(sequences) * 1.5)) for b, n in bins.items()}
    order = sorted(range(len(sequences)), key=lambda i: (-(activity[i] - risk[i]), sequences[i], i))
    selected, counts = [], Counter()
    for i in order:
        s, b = sequences[i], min(len(sequences[i]) // 10, 5)
        if s in reference or activity[i] < floor or risk[i] > ceiling or counts[b] >= caps[b]:
            continue
        if any(ratio(s, sequences[j]) >= .8 for j in selected):
            continue
        selected.append(i)
        counts[b] += 1
        if len(selected) == retain:
            break
    return selected


def raw_metrics(sequences, activity, risk, evaluation_risk, reference, floor=.6, ceiling=.5):
    """Denominator is ALL raw draws, including repeats/reference matches/failures."""
    from Levenshtein import ratio

    validate_scores(sequences, activity, risk, evaluation_risk)
    unique = list(dict.fromkeys(sequences))
    good_reward, good_evaluation, good_both = set(), set(), set()
    for s, a, h, e in zip(sequences, activity, risk, evaluation_risk, strict=True):
        if s in reference or a < floor:
            continue
        if h <= ceiling:
            good_reward.add(s)
        if e <= ceiling:
            good_evaluation.add(s)
        if h <= ceiling and e <= ceiling:
            good_both.add(s)
    # Bounded diagnostic, not a family graph or all-pool diversity estimate.
    sample = sequences[:256]
    distances = [1-ratio(s, t) for i, s in enumerate(sample) for t in sample[i+1:]]
    return {"raw_reward_yield_per_1000": 1000*len(good_reward)/len(sequences),
            "raw_evaluation_yield_per_1000": 1000*len(good_evaluation)/len(sequences),
            "raw_joint_yield_per_1000": 1000*len(good_both)/len(sequences),
            "raw_unique_novel_fraction": sum(s not in reference for s in unique)/len(sequences),
            "raw_evaluation_risk_mean": float(np.mean(evaluation_risk)),
            "raw_evaluation_risk_p75": float(np.quantile(evaluation_risk, .75)),
            "raw_prefix256_pairwise_distance": float(np.mean(distances)) if distances else None,
            "raw_mean_length": float(np.mean([len(s) for s in sequences]))}


def token_batch(sequences, device):
    import torch

    from amp_challenge_2027.config import PAD_ID
    from amp_challenge_2027.tokenizer import encode

    rows = [encode(s) for s in sequences]
    return torch.tensor([r + [PAD_ID]*(max(map(len, rows))-len(r)) for r in rows], device=device)


def likelihood_loss(current, tokens):
    from amp_challenge_2027.config import PAD_ID

    action = tokens[:, 1:]
    mask = action != PAD_ID
    logp = current.gather(-1, action[..., None]).squeeze(-1)
    return (-(logp*mask).sum(1)/mask.sum(1)).mean()


def prefix_kl(current, prior, tokens):
    from amp_challenge_2027.config import PAD_ID

    mask = tokens[:, 1:] != PAD_ID
    return (((current.exp()*(current-prior)).sum(-1))*mask).sum()/mask.sum()


def train_raft(policy, frozen, score, reference, args):
    """Fresh policy draws each round; fixed initial-policy replay; endpoint only."""
    import torch
    from experiment_utils import write_summary
    from pilot_grpo import assert_strict_runtime

    from amp_challenge_2027.grpo import distributions, rollout

    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    replay_rng = torch.Generator(device=args.device).manual_seed(args.seed + 20000)
    replay = []
    while len(replay) < args.raft_replay:
        _, seqs = rollout(frozen, min(32, args.raft_replay-len(replay)), replay_rng, args.device)
        replay.extend(seqs)
    write_summary(args.out / "replay.csv", [{"sequence": s} for s in replay])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0)
    history, rows, draw_count, stopped, shortfall, updates = [], [], 0, False, False, 0
    for step in range(args.steps):
        assert_strict_runtime()
        sequences = []
        while len(sequences) < args.raft_draws:
            _, seqs = rollout(policy, min(32, args.raft_draws-len(sequences)), rng, args.device)
            sequences.extend(seqs)
        activity, risk = score(sequences)
        ids = teaching_ids(sequences, activity, risk, reference, args.raft_retain,
                           args.activity_floor, args.risk_ceiling)
        draw_count += len(sequences)
        chosen = set(ids)
        rows.extend({"step": step, "sequence": s, "activity": float(activity[i]), "risk": float(risk[i]),
                     "teaching_selected": i in chosen} for i, s in enumerate(sequences))
        shortfall = len(ids) < args.raft_min_retain
        max_kl = 0.
        if not shortfall:
            for _ in range(args.raft_epochs):
                order = torch.randperm(len(ids), generator=rng, device=args.device).cpu().tolist()
                for start in range(0, len(order), 16):
                    assert_strict_runtime()
                    selected = [sequences[ids[i]] for i in order[start:start+16]]
                    replay_ids = torch.randint(len(replay), (len(selected),), generator=rng, device=args.device).cpu().tolist()
                    tokens = token_batch(selected, args.device)
                    anchor = token_batch([replay[i] for i in replay_ids], args.device)
                    with torch.no_grad():
                        prior, anchor_prior = distributions(frozen, tokens), distributions(frozen, anchor)
                    current, anchor_current = distributions(policy, tokens), distributions(policy, anchor)
                    kl = (prefix_kl(current, prior, tokens) + prefix_kl(anchor_current, anchor_prior, anchor))/2
                    loss = likelihood_loss(current, tokens) + .25*likelihood_loss(anchor_current, anchor) + .05*kl
                    if not torch.isfinite(loss) or not torch.isfinite(kl):
                        raise ValueError("Nonfinite RAFT objective")
                    if float(kl.detach()) > args.kl_limit:
                        stopped = True
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                    before = [p.detach().clone() for p in policy.parameters()]
                    optimizer.step()
                    with torch.no_grad():
                        post = (prefix_kl(distributions(policy, tokens), prior, tokens) +
                                prefix_kl(distributions(policy, anchor), anchor_prior, anchor))/2
                        if not torch.isfinite(post) or float(post) > args.kl_limit:
                            for p, saved in zip(policy.parameters(), before, strict=True):
                                p.copy_(saved)
                            stopped = True
                        else:
                            max_kl = max(max_kl, float(post))
                            updates += 1
                    if stopped:
                        break
                if stopped:
                    break
        history.append({"step": step, "teaching_count": len(ids), "teaching_shortfall": shortfall,
                        "activity_mean": float(activity.mean()), "risk_mean": float(risk.mean()),
                        "max_accepted_batch_kl": max_kl, "kl_stop": stopped, "updates_total": updates})
        write_summary(args.out / "history.csv", history)
        print(f"[raft] round={step} teaching={len(ids)} updates={updates} kl_stop={stopped} shortfall={shortfall}", flush=True)
        if stopped or shortfall:
            break
    return rows, draw_count, stopped, {"teaching_shortfall": shortfall, "updates": updates,
                                        "unscored_replay_draws": len(replay)}
