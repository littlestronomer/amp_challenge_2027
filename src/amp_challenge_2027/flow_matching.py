"""Latent flow matching (OT-CFM) for AMP generation in ESM-2 embedding space.

The second generator architecture, complementary to the autoregressive
``PeptideDecoder``. The two are independent so you can compare them head-to-head.

Pipeline:
    peptide ──ESM-2 (frozen)──▶ x_1 (L, d) embeddings   ← data distribution
    noise ε ~ N(0, I) ─────────▶ x_0                    ← source distribution
    flow: x_t = (1-t)·x_0 + t·x_1   (straight-line OT path)
    train a velocity field v_θ(x_t, t) to predict x_1 - x_0
    at inference: ODE-integrate ε → x_1 in ~10-50 Euler steps
    decode: x_1 → sequence via the ESM-2 LM head (argmax, for now)

Why flow matching over DDPM (the AMP-Diffusion baseline):
    - straight OT paths → fewer steps (10-50 vs 1000)
    - more stable training
    - conditioning is native (CFG, via time/class embedding)

Known risk (same as AMP-Diffusion): the embedding→sequence decode is lossy.
``decode_sequences`` uses argmax over the LM head now; a round-trip consistency
loss and constrained decoding are the planned upgrade.

Defaults sized for ~100k peptides: 4 layers × 256 hidden × 4 heads (~3M params
for the denoiser, on top of frozen ESM-2 8M embeddings). Increase if using a
larger ESM-2 backbone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FlowMatchingConfig:
    """Hyperparameters for the latent flow-matching denoiser.

    ``esm_model`` selects the frozen encoder whose embeddings we flow-match in.
    ``hidden_size`` is the denoiser's internal width (independent of the ESM
    embedding dim; a projection adapts between them).
    """

    esm_model: str = "facebook/esm2_t6_8M_UR50D"
    esm_embed_dim: int = 320  # 8M=320, 35M=480, 150M=640, 650M=1280
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_length: int = 52
    num_flow_steps: int = 50
    dropout: float = 0.1
    cfg_cond_drop: float = 0.1
    num_classes: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> FlowMatchingConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Denoiser (module-level)
# ---------------------------------------------------------------------------


class FlowDenoiser(nn.Module):
    """Predicts the velocity v(x_t, t) ∈ R^{L×d}.

    A small transformer over the (L, esm_dim) embedding sequence, with time
    ``t`` injected via a sinusoidal MLP and optional class conditioning for
    classifier-free guidance.

    Inputs:
        x_t   — (B, L, esm_dim) noisy embeddings
        t     — (B,) flow time in [0, 1]
        cond  — (B,) optional class label, or None for unconditional
    Output:
        v     — (B, L, esm_dim) predicted velocity (= x_1 - x_0 under OT-CFM)
    """

    def __init__(self, cfg: FlowMatchingConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.esm_embed_dim, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.esm_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.class_emb = (
            nn.Embedding(cfg.num_classes, cfg.hidden_size)
            if cfg.num_classes is not None
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_size * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.ln_f = nn.LayerNorm(cfg.hidden_size)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding of shape (B, hidden), before the MLP."""
        half = self.cfg.hidden_size // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.in_proj(x_t)  # (B, L, hidden)
        t_emb = self.time_mlp(self._time_embedding(t)).unsqueeze(1)  # (B, 1, hidden)
        h = h + t_emb
        if self.class_emb is not None and cond is not None:
            h = h + self.class_emb(cond).unsqueeze(1)
        h = self.transformer(h)
        h = self.ln_f(h)
        return self.out_proj(h)  # (B, L, esm_dim)


def build_denoiser(cfg: FlowMatchingConfig) -> FlowDenoiser:
    """Construct a FlowDenoiser from config."""
    return FlowDenoiser(cfg)


# ---------------------------------------------------------------------------
# Flow matching core: training objective and sampling
# ---------------------------------------------------------------------------


def ot_cfm_loss(
    denoiser: FlowDenoiser,
    x_1: torch.Tensor,
    *,
    cond: torch.Tensor | None = None,
    cfg_cond_drop: float = 0.1,
) -> torch.Tensor:
    """Optimal-transport conditional flow matching loss.

    Given data x_1 and noise x_0 ~ N(0,I), the OT path is
        x_t = (1-t) x_0 + t x_1
    with target velocity u_t = x_1 - x_0. We train v_θ to predict u_t.
    """
    B = x_1.shape[0]
    t = torch.rand(B, device=x_1.device)
    x_0 = torch.randn_like(x_1)
    x_t = (1 - t).view(B, 1, 1) * x_0 + t.view(B, 1, 1) * x_1
    target_v = x_1 - x_0

    if cond is not None and cfg_cond_drop > 0:
        drop = torch.rand(B, device=x_1.device) < cfg_cond_drop
        cond = cond.masked_fill(drop, 0)

    pred_v = denoiser(x_t, t, cond)
    return nn.functional.mse_loss(pred_v, target_v)


def integrate_flow(
    denoiser: FlowDenoiser,
    shape: tuple[int, int, int],
    *,
    device: str | torch.device,
    num_steps: int,
    cond: torch.Tensor | None = None,
    cfg_scale: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-integrate the flow from noise to data. Returns x_1 approximations."""
    device = torch.device(device)
    x = torch.randn(shape, device=device, generator=generator)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((shape[0],), i * dt, device=device)
        if cfg_scale > 0 and cond is not None:
            v_cond = denoiser(x, t, cond)
            v_uncond = denoiser(x, t, None)
            v = (1 + cfg_scale) * v_cond - cfg_scale * v_uncond
        else:
            v = denoiser(x, t, cond)
        x = x + v * dt
    return x


# ---------------------------------------------------------------------------
# ESM-2 encode / decode bridges
# ---------------------------------------------------------------------------

_ESM_CACHE: dict = {}


def load_esm2(model_id: str, *, device: str = "cpu"):
    """Load (model, tokenizer) for the frozen ESM-2 encoder. Cached."""
    from transformers import AutoModel, AutoTokenizer

    key = (model_id, device)
    if key not in _ESM_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        _ESM_CACHE[key] = (model, tokenizer)
    return _ESM_CACHE[key]


def embed_sequences(
    sequences: list[str], model_id: str, *, device: str = "cpu", max_length: int = 52
) -> torch.Tensor:
    """Encode peptides → (B, L, esm_dim) last-hidden-state embeddings (x_1)."""
    model, tokenizer = load_esm2(model_id, device=device)
    enc = tokenizer(
        sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state


# ESM-2 amino-acid id → residue mapping (id 4..23 in vocab order).
_ESM_AA_ORDER = "LAGVSETSPIFRKDQNWYMHCG"
_ESM_ID_TO_AA = {4 + i: aa for i, aa in enumerate(_ESM_AA_ORDER)}


def decode_sequences(
    embeddings: torch.Tensor, model_id: str, *, device: str = "cpu"
) -> list[str]:
    """Decode embeddings → sequences via the ESM-2 LM head (argmax).

    This is the lossy step. Argmax is the AMP-Diffusion baseline's approach;
    a round-trip consistency loss + constrained decoding is the planned upgrade.
    """
    model, _ = load_esm2(model_id, device=device)
    lm_head = _get_esm_lm_head(model)
    logits = embeddings @ lm_head.t()  # (B, L, vocab)
    ids = logits.argmax(dim=-1)  # (B, L)
    sequences: list[str] = []
    for row in ids.cpu().tolist():
        seq = "".join(_ESM_ID_TO_AA.get(i, "") for i in row if i >= 4)
        sequences.append(seq)
    return sequences


def _get_esm_lm_head(model) -> torch.Tensor:
    """Extract the LM-head weight matrix from an ESM-2 model (tied embeddings)."""
    if hasattr(model, "lm_head"):
        return model.lm_head.decoder.weight
    return model.embeddings.word_embeddings.weight


# ---------------------------------------------------------------------------
# LoRA (peft-compatible) for the denoiser
# ---------------------------------------------------------------------------

FLOW_LORA_TARGETS = ["in_proj", "out_proj", "linear1", "linear2"]


def apply_lora_to_denoiser(
    denoiser: FlowDenoiser, *, rank: int = 16, alpha: int = 32, dropout: float = 0.05
):
    """Wrap the denoiser with peft LoRA. Returns the PeftModel."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=FLOW_LORA_TARGETS,
        bias="none",
    )
    return get_peft_model(denoiser, cfg)


# ---------------------------------------------------------------------------
# Generation wrapper (mirrors AR sample_sequences for pipeline compatibility)
# ---------------------------------------------------------------------------


def flow_generate(
    denoiser: FlowDenoiser,
    *,
    batch_size: int,
    device: str,
    esm_model: str,
    esm_embed_dim: int,
    length: int,
    num_steps: int,
    cond=None,
    cfg_scale: float = 0.0,
    generator=None,
) -> list[str]:
    """End-to-end: sample noise → integrate flow → decode to sequences."""
    x_1 = integrate_flow(
        denoiser,
        shape=(batch_size, length, esm_embed_dim),
        device=device,
        num_steps=num_steps,
        cond=cond,
        cfg_scale=cfg_scale,
        generator=generator,
    )
    return decode_sequences(x_1, esm_model, device=device)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_flow_model(denoiser: FlowDenoiser, cfg: FlowMatchingConfig, dir_path: Path | str) -> None:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    torch.save(denoiser.state_dict(), p / "denoiser.pt")


def load_flow_model(dir_path: Path | str, *, map_location: str = "cpu") -> tuple[FlowDenoiser, FlowMatchingConfig]:
    p = Path(dir_path)
    cfg = FlowMatchingConfig.from_dict(json.loads((p / "config.json").read_text()))
    denoiser = FlowDenoiser(cfg)
    denoiser.load_state_dict(torch.load(p / "denoiser.pt", map_location=map_location))
    denoiser.eval()
    return denoiser, cfg


__all__ = [
    "FlowMatchingConfig",
    "FlowDenoiser",
    "build_denoiser",
    "ot_cfm_loss",
    "integrate_flow",
    "embed_sequences",
    "decode_sequences",
    "flow_generate",
    "save_flow_model",
    "load_flow_model",
    "load_esm2",
    "apply_lora_to_denoiser",
    "FLOW_LORA_TARGETS",
]
