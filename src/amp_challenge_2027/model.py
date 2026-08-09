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
    # MoE
    moe_num_experts: int = 8
    moe_num_active: int = 2  # top-k
    moe_load_balance_weight: float = 0.01
    # Tokenizer ids
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True

    @property
    def inner_size(self) -> int:
        return self.ffn_inner or self.hidden_size * 4

    @property
    def num_params(self) -> int:
        """Rough parameter estimate (exactly matches build for tied embeddings)."""
        h, v, nl, n = (
            self.hidden_size, self.vocab_size, self.num_layers, self.max_position_embeddings,
        )
        emb = v * h + n * h
        # Per block: qkv (3h²) + proj (h²) + ffn (2·h·inner) + 2 LN (2h)
        per_block = 3 * h * h + h * h + 2 * h * self.inner_size + 4 * h
        total = emb + nl * per_block + 2 * h  # ln_f + (lm_head if not tied)
        if self.ffn == "moe":
            # Replace dense FFN with E experts of the same shape.
            total -= nl * 2 * h * self.inner_size
            total += nl * self.moe_num_experts * 2 * h * self.inner_size
            total += nl * h * self.moe_num_experts  # gate
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc_out(self.drop(self.act(self.fc_in(x))))
        return out, torch.tensor(0.0, device=x.device, requires_grad=False)


class MoEFFN(nn.Module):
    """Top-k Mixture-of-Experts FFN with load-balancing loss (Switch Transformer eq. 4).

    Token-level routing: each position routes to the top-k of ``num_experts``
    expert MLPs. Returns (out, aux_loss).
    """

    def __init__(self, hidden: int, inner: int, num_experts: int, top_k: int, lb_weight: float):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lb_weight = lb_weight
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_ExpertMLP(hidden, inner) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        xf = x.reshape(-1, H)
        N = xf.shape[0]
        logits = self.gate(xf)
        vals, idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(vals, dim=-1)

        out = torch.zeros_like(xf)
        for e in range(self.num_experts):
            mask = idx == e
            tok_ids, slot_ids = mask.nonzero(as_tuple=True)
            if tok_ids.numel() == 0:
                continue
            exp_out = self.experts[e](xf[tok_ids])
            w = weights[tok_ids, slot_ids].unsqueeze(-1)
            out.index_add_(0, tok_ids, exp_out * w)
        out = out.reshape(B, T, H)

        with torch.no_grad():
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, idx[:, :1], 1.0)
            tpe = one_hot.sum(0)
            rp = torch.softmax(logits, dim=-1).mean(0)
        aux = self.lb_weight * self.num_experts * (tpe / N * rp).sum()
        return out, aux


class _ExpertMLP(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.c_fc = nn.Linear(hidden, inner)
        self.c_proj = nn.Linear(inner, hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


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

    def __init__(self, hidden_size: int, mode: ResidualMode, num_layers: int):
        super().__init__()
        self.mode = mode
        n = num_layers * 2  # one query per sub-layer
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

    @staticmethod
    def _build_ffn(cfg: DecoderConfig) -> nn.Module:
        if cfg.ffn == "dense":
            return DenseFFN(cfg.hidden_size, cfg.inner_size, cfg.dropout)
        if cfg.ffn == "moe":
            return MoEFFN(
                cfg.hidden_size, cfg.inner_size, cfg.moe_num_experts,
                cfg.moe_num_active, cfg.moe_load_balance_weight,
            )
        raise ValueError(f"Unknown ffn mode: {cfg.ffn!r}")

    def forward(
        self, h: torch.Tensor, *, layer_past=None, use_cache: bool = False
    ) -> tuple[torch.Tensor, tuple | None, torch.Tensor]:
        attn_out, present = self.attn(self.ln1(h), layer_past=layer_past, use_cache=use_cache)
        h1 = h + self.drop(attn_out)
        ffn_out, aux = self.ffn(self.ln2(h1))
        h2 = h1 + self.drop(ffn_out)
        return h2, present, aux


# ---------------------------------------------------------------------------
# Full decoder (module-level)
# ---------------------------------------------------------------------------


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
            AttentionResidual(cfg.hidden_size, cfg.residual, cfg.num_layers)
            if cfg.residual in ("attnres", "block_attnres")
            else None
        )
        self.block_size: int = (
            max(1, cfg.num_layers // cfg.attnres_num_blocks)
            if cfg.residual == "block_attnres"
            else 0
        )

    def forward(
        self, input_ids: torch.Tensor, *, use_cache: bool = False, past_keys_values=None
    ) -> DecoderOutput:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

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
            h, present, aux = block(h, layer_past=layer_past, use_cache=use_cache)
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
        return DecoderOutput(logits=logits, aux_loss=aux_total, past_keys_values=presents)


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
    model.load_state_dict(torch.load(p / "model.pt", map_location=map_location))
    model.eval()
    return model, cfg


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
]
