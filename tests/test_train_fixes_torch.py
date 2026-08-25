"""Regression tests for the trainer fixes (Track A) — torch-dependent part.

Skips in the minimal dev environment; runs with ``--extra ml``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.flow_matching import (  # noqa: E402
    FlowDenoiser,
    FlowMatchingConfig,
    _uncond_cond,
    decode_sequences,
    integrate_flow,
    ot_cfm_loss,
)
from amp_challenge_2027.lora import inject_lora, merge_and_strip_lora  # noqa: E402

# ---------------------------------------------------------------------------
# LoRA merge
# ---------------------------------------------------------------------------


def test_lora_merge_preserves_device_and_freezes():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 8)
    model = torch.nn.Sequential(base)
    model = inject_lora(model, rank=2, alpha=4)
    model.eval()
    x = torch.randn(3, 8)
    before = model(x)
    merged = merge_and_strip_lora(model)
    assert isinstance(merged[0], torch.nn.Linear)
    assert merged[0].weight.device == base.weight.device
    assert not merged[0].weight.requires_grad
    # Zero-init adapters → merging right after injection is an identity.
    assert torch.allclose(merged(x), before, atol=1e-6)


# ---------------------------------------------------------------------------
# Flow matching: positional encoding, masking, CFG null class, decode cut
# ---------------------------------------------------------------------------


def _tiny_denoiser(num_classes: int | None = None) -> FlowDenoiser:
    cfg = FlowMatchingConfig(
        esm_embed_dim=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_length=12,
        num_classes=num_classes,
    )
    denoiser = FlowDenoiser(cfg)
    denoiser.eval()  # deterministic forward (no dropout)
    return denoiser


def test_positional_encoding_breaks_permutation_equivariance():
    denoiser = _tiny_denoiser()
    x = torch.randn(1, 10, 16)
    t = torch.tensor([0.5])
    v1 = denoiser(x, t)
    v2 = denoiser(x.flip(dims=(1,)), t)
    assert not torch.allclose(v1, v2.flip(dims=(1,))), (
        "a position-aware denoiser must distinguish residue order"
    )


def test_cfm_loss_respects_valid_mask():
    torch.manual_seed(0)
    denoiser = _tiny_denoiser()
    x1 = torch.randn(4, 12, 16)
    mask_full = torch.ones(4, 12)
    mask_half = torch.cat([torch.ones(4, 6), torch.zeros(4, 6)], dim=1)
    loss_full = ot_cfm_loss(denoiser, x1, mask=mask_full, cfg_cond_drop=0.0)
    loss_half = ot_cfm_loss(denoiser, x1, mask=mask_half, cfg_cond_drop=0.0)
    assert torch.isfinite(loss_full) and torch.isfinite(loss_half)
    assert not torch.allclose(loss_full, loss_half)


def test_cfg_null_class_is_extra_embedding_row():
    denoiser = _tiny_denoiser(num_classes=4)
    assert denoiser.class_emb is not None
    assert denoiser.class_emb.num_embeddings == 5  # 4 classes + learned null
    cond = torch.tensor([0, 1, 2])
    null = _uncond_cond(denoiser, cond)
    assert null is not None and (null == 4).all()


def test_cfg_scale_one_equals_pure_conditional():
    """Standard convention: v = uncond + s·(cond − uncond) ⇒ s=1 is pure cond."""
    torch.manual_seed(1)
    denoiser = _tiny_denoiser(num_classes=3)
    shape = (2, 8, 16)
    gen = torch.Generator().manual_seed(5)

    x_guided = integrate_flow(
        denoiser,
        shape,
        device="cpu",
        num_steps=4,
        cond=torch.tensor([0, 1]),
        cfg_scale=1.0,
        generator=gen,
    )
    gen2 = torch.Generator().manual_seed(5)
    x_plain = integrate_flow(
        denoiser,
        shape,
        device="cpu",
        num_steps=4,
        cond=torch.tensor([0, 1]),
        cfg_scale=0.0,
        generator=gen2,
    )
    assert torch.allclose(x_guided, x_plain, atol=1e-6)


def _fake_esm(monkeypatch):
    """Patch load_esm2 with a tied-embedding 'LM head' over a 24-token vocab."""
    import amp_challenge_2027.flow_matching as fm

    vocab, dim = 24, 16
    weight = torch.randn(vocab, dim)

    class FakeHead:
        class decoder:  # noqa: N999 - mirrors ESM structure
            weight = weight

    class FakeModel:
        lm_head = FakeHead()

    monkeypatch.setattr(fm, "load_esm2", lambda *a, **k: (FakeModel(), None))
    return weight


def test_decode_stops_at_first_special_and_honors_lengths(monkeypatch):
    weight = _fake_esm(monkeypatch)
    # Construct embeddings whose argmax ids are [residue, pad(0), residue...].
    b = 1
    ids = torch.tensor([[5, 0, 9, 3, 11]])
    emb = weight[ids] + 1e-3 * torch.randn(b, 5, 16, generator=torch.Generator().manual_seed(0))
    seqs = decode_sequences(emb, "fake/model", device="cpu")
    assert len(seqs[0]) == 1, f"decode must stop at the first special token; got {seqs!r}"

    seqs_cut = decode_sequences(emb, "fake/model", device="cpu", lengths=[5])
    assert len(seqs_cut[0]) == 1  # still stops at the special id in position 1
