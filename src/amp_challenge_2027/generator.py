"""Facade over the custom PeptideDecoder for AMP generation.

The architecture lives in ``model.py`` (config-driven: residual mode + FFN
mode). This module owns the sampling surface used by the inference entry point
(``generate.py``) and the RL trainer, including the reproducibility contract:
a seeded ``torch.Generator`` drives all sampling randomness.

One place owns the decoding strategy (temperature/top-k/top-p, EOS handling),
so the model can be swapped (dense ↔ MoE ↔ AttnRes) without touching selection
or inference code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amp_challenge_2027.model import DecoderConfig

if TYPE_CHECKING:
    import torch  # noqa: F401


def sample_sequences(
    model,
    *,
    n_sequences: int,
    device: str,
    max_length: int,
    generator: torch.Generator | int | None = None,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    batch_size: int = 256,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    vocab_size: int = 24,
    allowed_ids: list[int] | None = None,
    repetition_penalty: float = 1.2,
    min_length: int = 8,
    charge: list[int] | None = None,
) -> list[str]:
    """Sample ``n_sequences`` peptides from the model autoregressively.

    Determinism contract: pass a seeded ``torch.Generator`` on the target
    device; all sampling randomness flows through it.

    Generates in batches to bound peak memory. Only the 20 amino-acid token ids
    (plus EOS) are allowed during generation; specials like PAD/BOS/MASK are
    masked out after the first step. Invalid/short/degenerate samples are left
    in — ``select.py`` filters them downstream.

    ``charge`` (optional, length ``n_sequences``): per-sequence charge bins for
    a charge-conditioned model (``conditioning="charge"``). Each sequence is
    generated conditioned on its bin. ``None`` → unconditional (or the model's
    default bin if it was trained conditioned) — backward compatible.

    ``repetition_penalty`` (>1.0) penalizes residues that already appear in the
    partial sequence, breaking the degenerate "KKKKK" repeats AR models produce
    when undertrained. Standard CTRL/CTRL-paper penalty: divide logit of any
    token that appeared before by this factor.
    """
    import torch

    from amp_challenge_2027 import tokenizer as tok

    device = torch.device(device)
    if allowed_ids is None:
        allowed_ids = list(tok.RESIDUE_TO_ID.values()) + [eos_id]
    if charge is not None and len(charge) != n_sequences:
        raise ValueError(
            f"charge has {len(charge)} bins but n_sequences={n_sequences}; must align"
        )

    sequences: list[str] = []
    remaining = n_sequences
    offset = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        ids = _sample_batch(
            model,
            bs=bs,
            max_length=max_length,
            device=device,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            vocab_size=vocab_size,
            allowed_ids=allowed_ids,
            repetition_penalty=repetition_penalty,
            min_length=min_length,
            charge=charge[offset : offset + bs] if charge is not None else None,
        )
        for row in ids:
            sequences.append(tok.decode(row))
        remaining -= bs
        offset += bs
    return sequences


def _sample_batch(
    model,
    *,
    bs: int,
    max_length: int,
    device,
    generator,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    vocab_size: int,
    allowed_ids: list[int],
    repetition_penalty: float = 1.2,
    min_length: int = 8,
    charge: list[int] | None = None,
) -> list[list[int]]:
    """Sample one batch. Returns raw token-id lists (specials not yet stripped)."""
    import torch

    cur = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(bs, dtype=torch.bool, device=device)
    mask_bias = torch.full((vocab_size,), float("-inf"), device=device)
    mask_bias[allowed_ids] = 0.0
    charge_t = (
        torch.as_tensor(charge, device=device, dtype=torch.long) if charge is not None else None
    )

    for step in range(max_length):
        out = model(cur, charge=charge_t)
        logits = out.logits[:, -1, :] / max(temperature, 1e-8) + mask_bias

        # Repetition penalty: for each sequence, find tokens already generated
        # and divide their logits by the penalty factor (>1 → less likely to repeat).
        if repetition_penalty > 1.0:
            for i in range(bs):
                if finished[i]:
                    continue
                seen = cur[i].unique()
                logits[i, seen] = logits[i, seen] / repetition_penalty

        # Don't emit EOS before min_length — forces non-trivial sequences.
        if step < min_length:
            logits[:, eos_id] = float("-inf")

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            kth = v[:, -1:]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if top_p is not None and 0 < top_p < 1.0:
            logits = _top_p_filter(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        finished = finished | (nxt == eos_id)
        cur = torch.cat([cur, nxt.unsqueeze(-1)], dim=1)
        if finished.all():
            break
    return cur.tolist()


def _top_p_filter(logits, top_p: float):
    """Nucleus filter: keep smallest token set with cumulative prob ≥ top_p."""
    import torch

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = remove.scatter(1, sorted_idx, remove)
    return logits.masked_fill(remove, float("-inf"))


# Re-export model factory + serialization so callers import from one place.
from amp_challenge_2027.model import build_model, load_model, save_model  # noqa: E402

__all__ = [
    "sample_sequences",
    "DecoderConfig",
    "build_model",
    "save_model",
    "load_model",
]
