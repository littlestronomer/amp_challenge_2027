"""Small unconditional, group-relative clipped policy-gradient pilot."""
import torch

from amp_challenge_2027.config import BOS_ID, EOS_ID, PAD_ID
from amp_challenge_2027.tokenizer import RESIDUE_TO_ID, decode


def policy_logps(logits, positions, minimum=8, maximum=50):
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[..., list(RESIDUE_TO_ID.values())] = True
    allowed[..., EOS_ID] = positions >= minimum
    allowed = allowed & (positions < maximum).unsqueeze(-1)
    allowed[..., EOS_ID] |= positions >= maximum
    return torch.log_softmax(logits.masked_fill(~allowed, -1e9), dim=-1)


@torch.no_grad()
def rollout(model, count, generator, device, maximum=50):
    model.eval()
    tokens = torch.full((count, 1), BOS_ID, device=device, dtype=torch.long)
    finished = torch.zeros(count, device=device, dtype=torch.bool)
    for step in range(maximum + 1):
        logits = model(tokens).logits[:, -1]
        lp = policy_logps(logits, torch.full((count,), step, device=device), maximum=maximum)
        nxt = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
        tokens = torch.cat((tokens, nxt[:, None]), dim=1)
        finished |= nxt == EOS_ID
        if finished.all():
            break
    return tokens, [decode(r.tolist()) for r in tokens]


def distributions(model, tokens, maximum=50):
    logits = model(tokens[:, :-1]).logits
    positions = torch.arange(logits.shape[1], device=tokens.device).expand(tokens.shape[0], -1)
    return policy_logps(logits, positions, maximum=maximum)


def advantages(rewards, group_size):
    grouped = rewards.reshape(-1, group_size)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - grouped.mean(dim=1, keepdim=True)) / std.clamp_min(1e-6)).reshape(-1)


def objective(current, old, reference, tokens, advantage, beta=.05, clip=.2):
    action = tokens[:, 1:]
    mask = action != PAD_ID
    selected = current.gather(-1, action[..., None]).squeeze(-1)
    prior = old.gather(-1, action[..., None]).squeeze(-1)
    ratio = (selected - prior).clamp(-20, 20).exp()
    gain = torch.minimum(ratio * advantage[:, None], ratio.clamp(1-clip, 1+clip) * advantage[:, None])
    kl = (current.exp() * (current - reference)).sum(-1)
    per_sequence = ((-gain + beta * kl) * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_sequence.mean(), (kl * mask).sum() / mask.sum().clamp_min(1)
