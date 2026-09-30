"""Custom autoregressive transformer decoder for AMP generation.

Full custom ``nn.Module`` (not HF GPT-2) so that residual-mode and FFN-mode
variants — including Attention Residuals (AttnRes, Kimi Team 2026,
arXiv:2603.15031) and Mixture-of-Experts — are first-class, config-driven
options rather than monkeypatches on HF internals.

Config-driven variants (all via ``DecoderConfig``):
    residual: "standard" | "attnres" | "block_attnres"
    ffn:      "dense" | "moe"

Architecture sizing:
    Defaults are tuned for ~100k short peptides (8-50 residues, 24-token vocab):
    6 layers × 384 hidden × 6 heads (~8M params). This is proportionate to the
    data scale per Chinchilla (~20 tokens/param); the prior 12L/768H GPT-2
    defaults were ~10× over-parameterized for this task. For larger datasets or
    when AttnRes shifts the depth optimum (paper §5.4.1), increase depth via
    ``DecoderConfig(num_layers=...)``.

References:
    - AttnRes: Eq. 2-4, Fig. 2 of arXiv:2603.15031. Zero-init pseudo-queries
      mandatory (paper §5).
    - MoE load-balance: Switch Transformer (Fedus et al. 2022), eq. 4.
    - SDPA: ``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from amp_challenge_2027.assay_conditioning import (
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    collate_assay_conditions,
    validate_condition_schema,
)
from amp_challenge_2027.conditioning import (
    HMOMENT_MODE_VALUE,
    HYDRO_MODE_VALUE,
    parse_conditioning_axes,
)
from amp_challenge_2027.config import MAX_LENGTH
from amp_challenge_2027.tokenizer import VOCAB

ResidualMode = Literal["standard", "attnres", "block_attnres"]
FFNMode = Literal["dense", "moe"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecoderConfig:
    """Architecture hyperparameters. Defaults are sized for ~100k short peptides.

    All architecture knobs are fields so the trainer can sweep them via CLI
    without editing source. Flip ``residual`` / ``ffn`` to switch variants.
    """

    vocab_size: int = len(VOCAB)  # 24
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    max_position_embeddings: int = MAX_LENGTH + 2  # 52
    ffn_inner: int | None = None  # None → 4 * hidden_size
    dropout: float = 0.1
    # Architecture variants
    residual: ResidualMode = "standard"
    ffn: FFNMode = "dense"
    # AttnRes / Block AttnRes
    attnres_num_blocks: int = 4  # N; for 6 layers → block_size ≈ 1-2
    # Version 1 is the deployed legacy topology. Never infer a migration when
    # loading a checkpoint without this field.
    residual_impl_version: int = 1
    attnres_block_layers: int = 2
    # MoE
    moe_num_experts: int = 8
    moe_num_active: int = 2  # top-k
    moe_load_balance_weight: float = 0.01
    moe_expert_inner: int | None = None
    # Biological conditioning is independent of physicochemical bin embeddings.
    # A None schema creates no new modules and preserves old state-dict keys.
    assay_schema: dict[str, Any] | None = None
    # Tokenizer ids
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    # Conditioning ("none" = unconditional, backward-compatible). Accepts a
    # comma-separated subset of {charge, hydro, hmoment}; one bin-embedding
    # table per axis is added to every position.
    # Charge bin index = round(modbjellqvist_charge) - charge_min, clamped to
    # [0, num_charge_bins). Charge range −8..+12 covers the reference (mean
    # +2.85, σ 3.28; extremes beyond ±3σ clamp into the edge bins).
    # Hydro/hmoment bins use the seqme/modlamp Eisenberg scale; the default
    # edges are provisional and validated by scripts/parity_modlamp_descriptors.py.
    conditioning: str = "none"
    num_charge_bins: int = 21
    charge_min: int = -8
    num_hydro_bins: int = 20
    hydro_min: float = -1.5
    hydro_step: float = 0.13
    num_hmoment_bins: int = 14
    hmoment_min: float = 0.0
    hmoment_step: float = 0.08

    @property
    def inner_size(self) -> int:
        return self.ffn_inner or self.hidden_size * 4

    @property
    def expert_inner_size(self) -> int:
        return self.moe_expert_inner or self.inner_size

    def __post_init__(self) -> None:
        if self.residual_impl_version not in (1, 2):
            raise ValueError("residual_impl_version must be 1 (legacy) or 2")
        if self.attnres_block_layers < 1 or self.attnres_num_blocks < 1:
            raise ValueError("AttnRes block sizes must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.ffn == "moe" and not 1 <= self.moe_num_active <= self.moe_num_experts:
            raise ValueError("MoE active experts must be between 1 and num_experts")
        if self.assay_schema is not None:
            validate_condition_schema(self.assay_schema)

    @property
    def num_params(self) -> int:
        """Rough parameter estimate (exactly matches build for tied embeddings)."""
        h, v, nl, n = (
            self.hidden_size, self.vocab_size, self.num_layers, self.max_position_embeddings,
        )
        emb = v * h + n * h
        # Per block: qkv (3h²) + proj (h²) + ffn (2·h·inner + inner + h biases)
        # + 2 LN (2h each)
        per_block = 3 * h * h + h * h + 2 * h * self.inner_size + self.inner_size + h + 4 * h
        total = emb + nl * per_block + 2 * h  # ln_f + (lm_head if not tied)
        if self.ffn == "moe":
            # Replace dense FFN with E experts of the same shape.
            total -= nl * 2 * h * self.inner_size
            total += nl * self.moe_num_experts * 2 * h * self.expert_inner_size
            total += nl * h * self.moe_num_experts  # gate
        if self.assay_schema is not None:
            total += nl * (4 * h * h + 6 * h + 1)
            total += sum((len(self.assay_schema["vocabularies"][field]) + 2) * h for field in CATEGORICAL_FIELDS)
            total += 2 * len(NUMERIC_FIELDS) * h + h * h + 5 * h
        for axis in parse_conditioning_axes(self.conditioning):
            bins = {
                "charge": self.num_charge_bins,
                "hydro": self.num_hydro_bins,
                "hmoment": self.num_hmoment_bins,
            }[axis]
            total += bins * h
        return int(total)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecoderConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class DecoderOutput:
    """Forward output: logits, MoE aux loss, optional KV cache."""

    logits: torch.Tensor
    aux_loss: torch.Tensor
    past_keys_values: list | None = None
    routing_metrics: list[dict[str, torch.Tensor]] | None = None


# ---------------------------------------------------------------------------
# Building blocks (module-level classes)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm — used by AttnRes key normalization (paper Eq. 2)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


class CausalSDPA(nn.Module):
    """Multi-head causal self-attention via ``scaled_dot_product_attention``.

    Uses PyTorch's fused SDPA kernel (faster + more memory-efficient than a
    manual attention mask). Determinism requires
    ``torch.use_deterministic_algorithms(True)`` (see ``training.enable_determinism``).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.h = num_heads
        self.d = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.drop_p = dropout

    def forward(
        self, x: torch.Tensor, *, layer_past=None, use_cache: bool = False
    ) -> tuple[torch.Tensor, tuple | None]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if use_cache and layer_past is not None:
            pk, pv = layer_past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out), present


# ---------------------------------------------------------------------------
# Feed-forward variants
# ---------------------------------------------------------------------------


class DenseFFN(nn.Module):
    """GPT-2 style MLP: up → GELU → down. Returns (out, aux_loss=0)."""

    def __init__(self, hidden: int, inner: int, dropout: float):
        super().__init__()
        self.fc_in = nn.Linear(hidden, inner)
        self.fc_out = nn.Linear(inner, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc_out(self.drop(self.act(self.fc_in(x))))
        return out, torch.tensor(0.0, device=x.device, requires_grad=False)


class MoEFFN(nn.Module):
    """Top-k Mixture-of-Experts FFN with load-balancing loss (Switch Transformer eq. 4).

    Token-level routing: each position routes to the top-k of ``num_experts``
    expert MLPs. Returns (out, aux_loss).
    """

    def __init__(self, hidden: int, inner: int, num_experts: int, top_k: int, lb_weight: float, dropout: float = 0.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.lb_weight = lb_weight
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_ExpertMLP(hidden, inner, dropout) for _ in range(num_experts)]
        )
        self.routing_stats: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        xf = x.reshape(-1, H)
        valid = (torch.ones((B * T,), dtype=torch.bool, device=x.device)
                 if valid_mask is None else valid_mask.to(device=x.device, dtype=torch.bool).reshape(-1))
        if valid.numel() != B * T:
            raise ValueError("MoE valid_mask must match the token dimensions")
        # Disable autocast explicitly: routing probabilities and their gradients
        # must stay FP32 even when expert matmuls use mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(xf.float(), self.gate.weight.float())
        vals, idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(vals, dim=-1)

        out = torch.zeros_like(xf)
        for e in range(self.num_experts):
            mask = (idx == e) & valid[:, None]
            tok_ids, slot_ids = mask.nonzero(as_tuple=True)
            if tok_ids.numel() == 0:
                continue
            exp_out = self.experts[e](xf[tok_ids])
            w = weights[tok_ids, slot_ids].to(exp_out.dtype).unsqueeze(-1)
            out.index_add_(0, tok_ids, (exp_out * w).to(out.dtype))
        out = out.reshape(B, T, H)

        count = valid.sum()
        # Normalize over all selected expert assignments, not only the first
        # expert. Counts are discrete; average probabilities must keep autograd.
        assignments = F.one_hot(idx[valid], self.num_experts).float().sum(dim=(0, 1)).detach()
        fraction = assignments / (count.clamp_min(1) * self.top_k)
        probabilities = torch.softmax(logits, dim=-1)
        mean_probability = (probabilities * valid[:, None]).sum(0) / count.clamp_min(1)
        aux = self.lb_weight * self.num_experts * (fraction * mean_probability).sum()
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
        self.routing_stats = {
            "assignment_fraction": fraction.detach(),
            "router_probability": mean_probability.detach(),
            "entropy": ((entropy * valid).sum() / count.clamp_min(1)).detach(),
            "valid_token_count": count.detach(),
        }
        return out, aux


class _ExpertMLP(nn.Module):
    def __init__(self, hidden: int, inner: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(hidden, inner)
        self.c_proj = nn.Linear(inner, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.drop(self.act(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Attention Residuals (AttnRes / Block AttnRes) — module-level
# ---------------------------------------------------------------------------


class AttentionResidual(nn.Module):
    """Attention Residuals: softmax attention over depth, per arXiv:2603.15031.

    Holds per-layer pseudo-queries (zero-initialized, paper §5) and RMSNorm key
    normalization (paper Eq. 2). Two aggregation ops per layer: pre-attn and
    pre-MLP, matching the paper's Fig. 2 where AttnRes is applied before each
    sub-layer.

    The ``aggregate`` method computes h_l = Σ α_{i→l} v_i (paper Eq. 4) where
    the source set V is either all prior layer outputs (full AttnRes) or block
    sums + current intra-block partial (Block AttnRes).
    """

    def __init__(self, hidden_size: int, mode: ResidualMode, num_layers: int, *, num_aggregations: int | None = None):
        super().__init__()
        self.mode = mode
        n = num_layers * 2 if num_aggregations is None else num_aggregations
        self.queries = nn.Parameter(torch.zeros(n, hidden_size))  # zero-init (§5)
        self.key_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(n)])

    def aggregate(
        self, sources: list[torch.Tensor], partial: torch.Tensor | None, layer_idx: int
    ) -> torch.Tensor:
        """h_l = Σ α_{i→l} v_i (paper Eq. 4)."""
        V = torch.stack(list(sources) + ([partial] if partial is not None else []))
        K = self.key_norms[layer_idx](V)
        w = self.queries[layer_idx]
        logits = torch.einsum("d,sbtd->sbt", w, K)  # (S, B, T)
        alpha = torch.softmax(logits, dim=0)
        return torch.einsum("sbt,sbtd->btd", alpha, V)


class AssayConditionEncoder(nn.Module):
    """Encode bound/context tokens; never infer an absent measured property."""

    def __init__(self, schema: dict[str, Any], hidden: int):
        super().__init__()
        validate_condition_schema(schema)
        self.schema = schema
        self.categories = nn.ModuleDict({
            field: nn.Embedding(len(schema["vocabularies"][field]) + 2, hidden, padding_idx=0)
            for field in CATEGORICAL_FIELDS
        })
        self.numerical = nn.Linear(2 * len(NUMERIC_FIELDS), hidden)
        self.projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.null = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.null, std=0.02)

    def forward(self, conditions: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.null.device
        categorical = conditions["categorical"].to(device=device, dtype=torch.long)
        values = conditions["values"].to(device=device, dtype=self.null.dtype)
        observed = conditions["observed"].to(device=device, dtype=torch.bool)
        valid = conditions["valid_mask"].to(device=device, dtype=torch.bool)
        if categorical.ndim != 3 or values.ndim != 3 or categorical.shape[:2] != values.shape[:2]:
            raise ValueError("Condition features must have matching (batch, memory, features) dimensions")
        if categorical.shape[-1] != len(CATEGORICAL_FIELDS) or values.shape[-1] != len(NUMERIC_FIELDS):
            raise ValueError("Condition feature widths do not match the checkpoint schema")
        if observed.shape != values.shape or valid.shape != values.shape[:2] or valid.shape[1] < 1:
            raise ValueError("Invalid assay observation or token mask")
        if not bool(valid[:, 0].all()):
            raise ValueError("The first condition token must be an always-visible NULL token")
        observed = observed & valid[..., None]
        # Never allow NaNs in masked slots to leak through a projection or SDPA.
        values = torch.where(observed, values, 0.0)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Observed numerical conditions must be finite")
        categorical = categorical.masked_fill(~valid[..., None], 0)
        features = self.numerical(torch.cat([values, observed.to(values.dtype)], dim=-1))
        for column, field in enumerate(CATEGORICAL_FIELDS):
            features = features + self.categories[field](categorical[..., column])
        memory = self.projection(features)
        memory = memory.masked_fill(~valid[..., None], 0.0)
        # The first memory entry is a learned NULL, not a pseudo-assay.
        memory = torch.cat([self.null.expand(memory.shape[0], 1, -1), memory[:, 1:]], dim=1)
        return memory, valid


class GatedAssayAttention(nn.Module):
    """Condition cross-attention branch whose initial residual is exactly zero."""

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dim = hidden // heads
        self.norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.gate = nn.Parameter(torch.zeros(()))
        self.drop_p = dropout

    def forward(self, h: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = h.shape
        def split(x):
            return x.reshape(batch, -1, self.heads, self.dim).transpose(1, 2)
        q = split(self.query(self.norm(h)))
        k, v = split(self.key(memory)), split(self.value(memory))
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :],
            dropout_p=self.drop_p if self.training else 0.0,
        ).transpose(1, 2).reshape(batch, length, hidden)
        return torch.tanh(self.gate) * self.proj(result)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    """Pre-norm transformer block (residual-mode-agnostic).

    The block itself just does f(ln(h)) and returns the delta; residual-mode
    handling (standard accumulation vs AttnRes aggregation) lives in
    ``PeptideDecoder.forward``, so one block class serves all residual modes.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = CausalSDPA(cfg.hidden_size, cfg.num_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.ffn: nn.Module = self._build_ffn(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.cross_attn = (
            GatedAssayAttention(cfg.hidden_size, cfg.num_heads, cfg.dropout)
            if cfg.assay_schema is not None else None
        )

    @staticmethod
    def _build_ffn(cfg: DecoderConfig) -> nn.Module:
        if cfg.ffn == "dense":
            return DenseFFN(cfg.hidden_size, cfg.inner_size, cfg.dropout)
        if cfg.ffn == "moe":
            return MoEFFN(
                cfg.hidden_size, cfg.expert_inner_size, cfg.moe_num_experts,
                cfg.moe_num_active, cfg.moe_load_balance_weight, cfg.dropout,
            )
        raise ValueError(f"Unknown ffn mode: {cfg.ffn!r}")

    def forward(
        self, h: torch.Tensor, *, layer_past=None, use_cache: bool = False,
        memory: torch.Tensor | None = None, memory_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple | None, torch.Tensor]:
        attn_out, present = self.attn(self.ln1(h), layer_past=layer_past, use_cache=use_cache)
        h1 = h + self.drop(attn_out)
        if self.cross_attn is not None:
            h1 = h1 + self.drop(self.cross_attn(h1, memory, memory_mask))
        ffn_out, aux = self.ffn(self.ln2(h1), valid_mask=valid_mask)
        h2 = h1 + self.drop(ffn_out)
        return h2, present, aux


# ---------------------------------------------------------------------------
# Full decoder (module-level)
# ---------------------------------------------------------------------------


def _normalize_cond_bins(
    *,
    charge: torch.Tensor | list[int] | None,
    cond_bins: dict[str, torch.Tensor | list[int]] | None,
    axes: tuple[str, ...],
    batch: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize conditioning-bin inputs to ``{axis: (B,) long tensor}``.

    Accepts the legacy ``charge=`` keyword (translated to the "charge" axis)
    or the multi-axis ``cond_bins`` dict — never both. Legacy semantics: a
    ``charge=`` value on a model that was not charge-trained is ignored, as
    the original forward always did. The new ``cond_bins`` API is strict:
    bins for axes the model was NOT trained with fail closed rather than
    being silently dropped; a scalar is broadcast to the batch.
    """
    if charge is not None and cond_bins is not None:
        raise ValueError("pass either charge= or cond_bins=, not both")
    source: dict[str, torch.Tensor | list[int]] = {}
    if charge is not None:
        if "charge" in axes:
            source = {"charge": charge}
    elif cond_bins is not None:
        source = dict(cond_bins)
        unknown = sorted(axis for axis in source if axis not in axes)
        if unknown:
            raise ValueError(
                f"conditioning bins given for inactive axes {unknown}; model axes = {list(axes)}"
            )
    normalized: dict[str, torch.Tensor] = {}
    for axis, value in source.items():
        tensor = torch.as_tensor(value, device=device, dtype=torch.long)
        if tensor.dim() == 0:
            tensor = tensor.expand(batch)
        if tensor.shape[0] != batch:
            raise ValueError(
                f"cond_bins[{axis!r}] has {tensor.shape[0]} entries but batch is {batch}"
            )
        normalized[axis] = tensor
    return normalized


class PeptideDecoder(nn.Module):
    """Custom autoregressive transformer for AMP generation.

    Combines config-driven residual modes (standard / AttnRes / Block AttnRes)
    and FFN modes (dense / MoE). The forward returns a ``DecoderOutput`` with
    logits + MoE aux loss, so the trainer can add the aux loss to the LM loss
    without caring which FFN is active.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head: nn.Linear | None = (
            None if cfg.tie_word_embeddings
            else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        )
        self.attnres: AttentionResidual | None = (
            AttentionResidual(
                cfg.hidden_size, cfg.residual, cfg.num_layers,
                num_aggregations=(cfg.num_layers * (3 if cfg.assay_schema is not None else 2) + 1
                                  if cfg.residual_impl_version == 2 else None),
            )
            if cfg.residual in ("attnres", "block_attnres")
            else None
        )
        self.block_size: int = (
            (cfg.attnres_block_layers if cfg.residual_impl_version == 2 else
             max(1, cfg.num_layers // cfg.attnres_num_blocks))
            if cfg.residual == "block_attnres"
            else 0
        )
        self.assay_encoder = (
            AssayConditionEncoder(cfg.assay_schema, cfg.hidden_size)
            if cfg.assay_schema is not None else None
        )
        # Multi-axis conditioning: one embedding table per active axis
        # ("charge", "hydro", "hmoment"), added to every position alongside
        # the positional embedding (class-conditional style). Attributes are
        # named "<axis>_emb" so existing charge-only checkpoints keep their
        # exact state-dict keys (charge_emb.weight). ``default_<axis>_bin``
        # holds the reference-mode bin so an unconditioned forward() call on
        # a conditioned model still works (generation without bins).
        self.conditioning_axes: tuple[str, ...] = parse_conditioning_axes(cfg.conditioning)
        self.charge_emb: nn.Embedding | None = None
        for axis in self.conditioning_axes:
            num_bins = {
                "charge": cfg.num_charge_bins,
                "hydro": cfg.num_hydro_bins,
                "hmoment": cfg.num_hmoment_bins,
            }[axis]
            setattr(self, f"{axis}_emb", nn.Embedding(num_bins, cfg.hidden_size))
            if axis == "charge":
                default_bin = 3 - cfg.charge_min  # reference-mode charge +3 (legacy)
            elif axis == "hydro":
                default_bin = int(round((HYDRO_MODE_VALUE - cfg.hydro_min) / cfg.hydro_step))
            else:
                default_bin = int(round((HMOMENT_MODE_VALUE - cfg.hmoment_min) / cfg.hmoment_step))
            self.register_buffer(
                f"default_{axis}_bin",
                torch.tensor(max(0, min(num_bins - 1, default_bin)), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        charge: torch.Tensor | list[int] | None = None,
        cond_bins: dict[str, torch.Tensor | list[int]] | None = None,
        assay_conditions: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
        past_keys_values=None,
    ) -> DecoderOutput:
        B, T = input_ids.shape
        if self.cfg.residual_impl_version == 2 and (use_cache or past_keys_values is not None):
            raise ValueError("Research residual version 2 uses full-prefix sampling; KV caching is unsupported")
        if assay_conditions is not None and self.assay_encoder is None:
            raise ValueError("Assay conditions supplied to a checkpoint without an assay schema")
        memory, memory_mask = None, None
        if self.assay_encoder is not None:
            if assay_conditions is None:
                assay_conditions = collate_assay_conditions([[] for _ in range(B)], self.cfg.assay_schema, input_ids.device)
            memory, memory_mask = self.encode_conditions(assay_conditions)
            if memory.shape[0] != B:
                raise ValueError("Assay condition batch does not match input_ids")
        valid_mask = input_ids != self.cfg.pad_token_id
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        bins = _normalize_cond_bins(
            charge=charge,
            cond_bins=cond_bins,
            axes=self.conditioning_axes,
            batch=B,
            device=input_ids.device,
        )
        for axis in self.conditioning_axes:
            bins_t = bins.get(axis)
            if bins_t is None:
                bins_t = getattr(self, f"default_{axis}_bin").to(input_ids.device).expand(B)
            h = h + getattr(self, f"{axis}_emb")(bins_t)[:, None, :]

        if self.cfg.residual_impl_version == 2 and self.attnres is not None:
            h, aux_total = self._forward_attnres_v2(h, memory, memory_mask, valid_mask)
            h = self.ln_f(h)
            logits = h @ self.tok_emb.weight.t() if self.lm_head is None else self.lm_head(h)
            return DecoderOutput(logits, aux_total, [], self._routing_metrics())

        sources: list[torch.Tensor] = []
        partial: torch.Tensor | None = None
        if self.cfg.residual == "block_attnres":
            sources.append(h)  # b0 = embedding

        aux_total = torch.tensor(0.0, device=input_ids.device)
        presents: list = []
        for i, block in enumerate(self.blocks):
            if self.cfg.residual == "attnres":
                if i > 0:
                    h = self.attnres.aggregate(sources, None, layer_idx=2 * i)
                sources.append(h)
            elif self.cfg.residual == "block_attnres":
                if partial is None:
                    h = self.attnres.aggregate(sources, None, layer_idx=2 * i)

            layer_past = past_keys_values[i] if past_keys_values else None
            h, present, aux = block(
                h, layer_past=layer_past, use_cache=use_cache,
                memory=memory, memory_mask=memory_mask, valid_mask=valid_mask,
            )
            aux_total = aux_total + aux
            if use_cache and present is not None:
                presents.append(present)

            if self.cfg.residual == "block_attnres":
                partial = h if partial is None else partial + h
                if self.block_size and (i + 1) % self.block_size == 0:
                    sources.append(h)
                    partial = None

        h = self.ln_f(h)
        logits = h @ self.tok_emb.weight.t() if self.lm_head is None else self.lm_head(h)
        return DecoderOutput(logits=logits, aux_loss=aux_total, past_keys_values=presents, routing_metrics=self._routing_metrics())

    def encode_conditions(self, conditions: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.assay_encoder is None:
            raise ValueError("This checkpoint has no assay condition encoder")
        return self.assay_encoder(conditions)

    def _routing_metrics(self) -> list[dict[str, torch.Tensor]]:
        return [block.ffn.routing_stats for block in self.blocks if isinstance(block.ffn, MoEFFN)]

    def _forward_attnres_v2(self, embedding, memory, memory_mask, valid_mask):
        """Aggregate branch *deltas*, including the final readout aggregation.

        Full: embedding and every preceding sublayer delta are separate sources.
        Block: completed block sums and the current partial sum are sources.
        The conditional extension has three sublayers per decoder layer.
        """
        sources = [embedding]
        partial = None
        query = 0
        aux_total = embedding.new_zeros(())

        def aggregate():
            return self.attnres.aggregate(sources, partial, query)

        def record(delta):
            nonlocal partial, query
            if self.cfg.residual == "attnres":
                sources.append(delta)
            else:
                partial = delta if partial is None else partial + delta
            query += 1

        for index, block in enumerate(self.blocks):
            attn_delta, _ = block.attn(block.ln1(aggregate()))
            record(block.drop(attn_delta))
            if block.cross_attn is not None:
                record(block.drop(block.cross_attn(aggregate(), memory, memory_mask)))
            ffn_delta, aux = block.ffn(block.ln2(aggregate()), valid_mask=valid_mask)
            record(block.drop(ffn_delta))
            aux_total = aux_total + aux
            if self.cfg.residual == "block_attnres" and (index + 1) % self.block_size == 0:
                sources.append(partial)
                partial = None
        return aggregate(), aux_total


# ---------------------------------------------------------------------------
# Public factory + serialization
# ---------------------------------------------------------------------------


def build_model(config: DecoderConfig | None = None) -> tuple[PeptideDecoder, DecoderConfig]:
    """Construct the decoder. Returns (model, config)."""
    cfg = config or DecoderConfig()
    return PeptideDecoder(cfg), cfg


def save_model(model: PeptideDecoder, out_dir: Path | str, *, config: DecoderConfig | None = None) -> None:
    """Save model state + config to ``out_dir`` (clean inference checkpoint)."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    cfg = config or model.cfg
    (p / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.save(model.state_dict(), p / "model.pt")


def load_model(model_dir: Path | str, *, map_location: str = "cpu") -> tuple[PeptideDecoder, DecoderConfig]:
    """Load a decoder saved by ``save_model``. Returns (model, config)."""
    p = Path(model_dir)
    cfg = DecoderConfig.from_dict(json.loads((p / "config.json").read_text()))
    model = PeptideDecoder(cfg)
    model.load_state_dict(torch.load(p / "model.pt", map_location=map_location, weights_only=True))
    model.eval()
    return model, cfg


def is_assay_parameter(name: str) -> bool:
    """Identify only new biological modules for the first-epoch backbone freeze."""
    return name.startswith("assay_encoder.") or (
        name.startswith("blocks.") and ".cross_attn." in name
    )


def warm_start_model(
    model_dir: Path | str, assay_schema: dict[str, Any], *, map_location: str = "cpu",
) -> tuple[PeptideDecoder, DecoderConfig]:
    """Add zero-gated assay conditioning without changing the source topology.

    Missing keys must equal the complete set of newly introduced assay keys.
    A partial, mismatched or previously conditional checkpoint fails closed.
    """
    path = Path(model_dir)
    source_config = DecoderConfig.from_dict(json.loads((path / "config.json").read_text()))
    if source_config.assay_schema is not None:
        raise ValueError("Warm start expects an unconditional biological source; use load_model for conditional checkpoints")
    config = DecoderConfig.from_dict({**source_config.to_dict(), "assay_schema": assay_schema})
    # A v2 AttnRes source changes its query count when adding a third branch and
    # cannot be claimed to preserve logits. The deployed checkpoints are v1.
    if config.residual_impl_version == 2 and config.residual != "standard":
        raise ValueError("Adding biological branches to v2 AttnRes changes its topology; train that architecture from scratch")
    model = PeptideDecoder(config)
    state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
    expected_new = {key for key in model.state_dict() if is_assay_parameter(key)}
    unexpected = set(state) - set(model.state_dict())
    missing = set(model.state_dict()) - set(state)
    if unexpected or missing != expected_new:
        raise ValueError(
            f"Source checkpoint differs beyond new assay modules: "
            f"unexpected={sorted(unexpected)}, missing_backbone={sorted(missing - expected_new)}"
        )
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise ValueError("Warm-start key validation failed")
    model.eval()
    return model, config


__all__ = [
    "DecoderConfig",
    "DecoderOutput",
    "PeptideDecoder",
    "DecoderBlock",
    "CausalSDPA",
    "DenseFFN",
    "MoEFFN",
    "AttentionResidual",
    "RMSNorm",
    "build_model",
    "save_model",
    "load_model",
    "warm_start_model",
    "is_assay_parameter",
    "AssayConditionEncoder",
    "GatedAssayAttention",
]
