"""LoRA for the custom PeptideDecoder.

We use our own LoRA implementation rather than ``peft`` because the decoder is
a plain ``nn.Module`` (not an HF model), and ``peft.get_peft_model`` expects an
HF-style model. The math is identical to peft's LoRA:

    For a frozen linear W (in, out), learn ΔW = (α/r) · B @ A
    where A: (r, in), B: (out, r), B zero-init so ΔW = 0 at start.

Only A, B train; the base weight is frozen. ``merge()`` folds ΔW into W for
inference. This is the same approach peft uses internally; we just apply it to
our own named modules.

If/when the decoder is also exposed as an HF model, ``peft`` can replace this
module entirely with no changes to the RL trainer's interface (apply → train →
merge). The function signatures here mirror what a peft-based path would need.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` + trainable low-rank delta. ``forward = base + ΔW x``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)  # uniform, matching peft; B = 0 → ΔW = 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t()
        delta = (delta @ self.lora_B.t()) * self.scaling
        return base_out + delta

    def merge(self) -> nn.Linear:
        """Fold ΔW into the base weight; return a plain Linear (no adapter).

        The merged layer is created on the base weight's device (a bare
        ``nn.Linear`` lands on CPU, which would strand GPU models mid-merge)
        and keeps gradients disabled — the adapter is being folded away, not
        retrained.
        """
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            merged = nn.Linear(
                self.base.in_features,
                self.base.out_features,
                bias=self.base.bias is not None,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
            merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
            merged.weight.requires_grad_(False)
            if merged.bias is not None:
                merged.bias.requires_grad_(False)
        return merged


# Default targets: the decoder's attention projections.
DEFAULT_TARGETS = ("qkv", "proj")


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> nn.Module:
    """Replace target ``nn.Linear`` submodules with ``LoRALinear``, in-place.

    ``targets`` matches by suffix on the dotted module name (e.g. ``"qkv"``
    matches ``blocks.0.attn.qkv``). All non-LoRA parameters are frozen. The
    AttnRes pseudo-queries and MoE gates are intentionally NOT targeted by
    default — leave them trainable only if you want to adapt them too.
    """
    targets = tuple(targets)
    to_replace: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(t) for t in targets):
            to_replace.append((name, mod))
    for name, linear in to_replace:
        _set_submodule(model, name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))

    # Freeze everything that is not a LoRA adapter parameter (collected once —
    # rescanning all modules per parameter is O(params × modules)).
    lora_param_ids = set()
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            lora_param_ids.add(id(mod.lora_A))
            lora_param_ids.add(id(mod.lora_B))
    for p in model.parameters():
        if id(p) not in lora_param_ids:
            p.requires_grad = False
    return model


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def merge_and_strip_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRALinear modules back into plain Linears in-place."""
    to_merge = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, lora in to_merge:
        _set_submodule(model, name, lora.merge())
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA adapter parameters (for lightweight checkpointing).

    Uses ``.clone()`` so the returned tensors are decoupled from the model's
    storage — without it, ``.detach().cpu()`` shares memory when the model is
    already on CPU, and later in-place ops (like ``zero_()``) would corrupt
    the saved values.
    """
    sd: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            sd[f"{name}.lora_A"] = mod.lora_A.detach().clone()
            sd[f"{name}.lora_B"] = mod.lora_B.detach().clone()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """Load LoRA adapter weights produced by ``lora_state_dict``."""
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear) and f"{name}.lora_A" in sd:
            mod.lora_A.data.copy_(sd[f"{name}.lora_A"])
            mod.lora_B.data.copy_(sd[f"{name}.lora_B"])


__all__ = [
    "LoRALinear",
    "DEFAULT_TARGETS",
    "inject_lora",
    "merge_and_strip_lora",
    "lora_state_dict",
    "load_lora_state_dict",
]
