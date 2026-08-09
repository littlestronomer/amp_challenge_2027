"""Tests for the flow-matching (OT-CFM) components.

Exercises shape contracts and the core OT-CFM math without requiring ESM-2
weights or a GPU: we instantiate a tiny denoiser and verify the loss, the
flow integration, and the CFG guidance path all behave correctly.

Skipped if torch is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.flow_matching import (  # noqa: E402
    FlowMatchingConfig,
    _build_denoiser,
    integrate_flow,
    ot_cfm_loss,
)


def _tiny_cfg(**kw) -> FlowMatchingConfig:
    base = dict(
        esm_embed_dim=16, hidden_size=32, num_layers=1, num_heads=2,
        max_length=8, num_flow_steps=5, dropout=0.0, cfg_cond_drop=0.0,
    )
    base.update(kw)
    return FlowMatchingConfig(**base)


def test_denoiser_forward_shape():
    cfg = _tiny_cfg()
    denoiser = _build_denoiser(cfg)
    x_t = torch.randn(4, cfg.max_length, cfg.esm_embed_dim)
    t = torch.rand(4)
    v = denoiser(x_t, t)
    assert v.shape == (4, cfg.max_length, cfg.esm_embed_dim)


def test_ot_cfm_loss_is_finite_and_scalar():
    cfg = _tiny_cfg()
    denoiser = _build_denoiser(cfg)
    x_1 = torch.randn(8, cfg.max_length, cfg.esm_embed_dim)
    loss = ot_cfm_loss(denoiser, x_1, cfg_cond_drop=0.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_ot_cfm_loss_decreases_with_optimization():
    """A few SGD steps should reduce the loss on a fixed batch."""
    cfg = _tiny_cfg()
    denoiser = _build_denoiser(cfg)
    opt = torch.optim.SGD(denoiser.parameters(), lr=1e-2)
    x_1 = torch.randn(16, cfg.max_length, cfg.esm_embed_dim)
    torch.manual_seed(0)
    loss0 = ot_cfm_loss(denoiser, x_1, cfg_cond_drop=0.0).item()
    for _ in range(20):
        opt.zero_grad()
        ot_cfm_loss(denoiser, x_1, cfg_cond_drop=0.0).backward()
        opt.step()
    loss1 = ot_cfm_loss(denoiser, x_1, cfg_cond_drop=0.0).item()
    assert loss1 < loss0


def test_integrate_flow_output_shape():
    cfg = _tiny_cfg()
    denoiser = _build_denoiser(cfg)
    x = integrate_flow(
        denoiser, shape=(6, cfg.max_length, cfg.esm_embed_dim),
        device="cpu", num_steps=cfg.num_flow_steps,
    )
    assert x.shape == (6, cfg.max_length, cfg.esm_embed_dim)


def test_integrate_flow_is_deterministic_with_generator():
    cfg = _tiny_cfg()
    denoiser = _build_denoiser(cfg).eval()
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    x1 = integrate_flow(
        denoiser, shape=(4, cfg.max_length, cfg.esm_embed_dim),
        device="cpu", num_steps=5, generator=g1,
    )
    x2 = integrate_flow(
        denoiser, shape=(4, cfg.max_length, cfg.esm_embed_dim),
        device="cpu", num_steps=5, generator=g2,
    )
    assert torch.allclose(x1, x2)


def test_cfg_guidance_path_runs():
    """Classifier-free guidance with a class condition should not error."""
    cfg = _tiny_cfg(num_classes=3)
    denoiser = _build_denoiser(cfg)
    cond = torch.tensor([0, 1, 2, 0])
    x = integrate_flow(
        denoiser, shape=(4, cfg.max_length, cfg.esm_embed_dim),
        device="cpu", num_steps=4, cond=cond, cfg_scale=2.0,
    )
    assert x.shape == (4, cfg.max_length, cfg.esm_embed_dim)


def test_conditioned_denoiser_forward():
    """Denoiser with class conditioning accepts a cond vector."""
    cfg = _tiny_cfg(num_classes=5)
    denoiser = _build_denoiser(cfg)
    x_t = torch.randn(3, cfg.max_length, cfg.esm_embed_dim)
    t = torch.rand(3)
    cond = torch.tensor([0, 2, 4])
    v = denoiser(x_t, t, cond)
    assert v.shape == (3, cfg.max_length, cfg.esm_embed_dim)
