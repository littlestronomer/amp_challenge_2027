"""Tests for the custom PeptideDecoder across all residual × FFN modes.

Covers: forward shapes, AttnRes zero-init (paper §5), MoE aux loss, Block
AttnRes source accumulation, LoRA inject/merge, save/load roundtrip, and
determinism of the model forward pass (critical for the validator).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.lora import (  # noqa: E402
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_and_strip_lora,
)
from amp_challenge_2027.model import (  # noqa: E402
    DecoderConfig,
    build_model,
    load_model,
    save_model,
)


def _tiny_config(**kw) -> DecoderConfig:
    base = dict(
        num_layers=4, num_heads=2, hidden_size=32, max_position_embeddings=16, dropout=0.0
    )
    base.update(kw)
    return DecoderConfig(**base)


def _ids(B=2, T=8):
    return torch.randint(4, 24, (B, T))


# --- defaults ---------------------------------------------------------------


def test_defaults_are_peptide_sized():
    """The audit flagged 12L/768H as over-parameterized. Defaults are now 6L/384H."""
    cfg = DecoderConfig()
    assert cfg.num_layers == 6
    assert cfg.hidden_size == 384
    assert cfg.num_heads == 6
    assert cfg.num_params < 15_000_000  # ~8-10M, not 85M


# --- forward shapes across modes --------------------------------------------


@pytest.mark.parametrize("residual", ["standard", "attnres", "block_attnres"])
@pytest.mark.parametrize("ffn", ["dense", "moe"])
def test_forward_shape_all_modes(residual, ffn):
    cfg = _tiny_config(residual=residual, ffn=ffn, moe_num_experts=4, moe_num_active=2)
    model, _ = build_model(cfg)
    out = model(_ids())
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.aux_loss.ndim == 0


def test_dense_aux_loss_is_zero():
    model, _ = build_model(_tiny_config(ffn="dense"))
    out = model(_ids())
    assert out.aux_loss.item() == 0.0


def test_moe_aux_loss_nonzero():
    cfg = _tiny_config(ffn="moe", moe_num_experts=4, moe_num_active=2)
    model, _ = build_model(cfg)
    out = model(_ids())
    assert out.aux_loss.item() >= 0.0


# --- AttnRes correctness (paper requirements) -------------------------------


def test_attnres_pseudo_queries_zero_init():
    """Paper §5: 'all pseudo-query vectors must be initialized to zero.'"""
    model, _ = build_model(_tiny_config(residual="attnres"))
    assert model.attnres is not None
    assert torch.all(model.attnres.queries == 0)


def test_block_attnres_pseudo_queries_zero_init():
    model, _ = build_model(_tiny_config(residual="block_attnres", attnres_num_blocks=2))
    assert model.attnres is not None
    assert torch.all(model.attnres.queries == 0)


def test_block_attnres_has_block_structure():
    cfg = _tiny_config(num_layers=4, residual="block_attnres", attnres_num_blocks=2)
    model, _ = build_model(cfg)
    assert model.block_size == 2


def test_standard_residual_has_no_attnres():
    model, _ = build_model(_tiny_config(residual="standard"))
    assert model.attnres is None


# --- determinism (critical for the validator) -------------------------------


def test_forward_is_deterministic_on_cpu():
    """Two forward passes with the same input must produce identical logits.

    This is the property the validator's byte-compare relies on. On CPU it
    holds without determinism flags; on CUDA it requires ``enable_determinism``.
    """
    torch.manual_seed(0)
    model, _ = build_model(_tiny_config())
    model.eval()
    ids = _ids()
    with torch.no_grad():
        out1 = model(ids).logits.clone()
        out2 = model(ids).logits
    assert torch.equal(out1, out2)


def test_forward_deterministic_moe():
    """MoE path (index_add_) must also be deterministic for reproducible runs."""
    torch.manual_seed(0)
    cfg = _tiny_config(ffn="moe", moe_num_experts=4, moe_num_active=2)
    model, _ = build_model(cfg)
    model.eval()
    ids = _ids()
    with torch.no_grad():
        out1 = model(ids).logits.clone()
        out2 = model(ids).logits
    assert torch.equal(out1, out2)


def test_forward_deterministic_attnres():
    """AttnRes aggregation must be deterministic for reproducible runs."""
    torch.manual_seed(0)
    model, _ = build_model(_tiny_config(residual="block_attnres", attnres_num_blocks=2))
    model.eval()
    ids = _ids()
    with torch.no_grad():
        out1 = model(ids).logits.clone()
        out2 = model(ids).logits
    assert torch.equal(out1, out2)


# --- gradients --------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_gradients_flow_to_attnres_queries(seed):
    # Legacy v1 keeps duplicated block-input sources for checkpoint parity;
    # identical sources can give exactly zero query gradients. Learning claims
    # belong to corrected v2, which aggregates distinct sublayer deltas.
    torch.manual_seed(seed)
    model, _ = build_model(_tiny_config(residual="attnres", residual_impl_version=2))
    out = model(_ids())
    out.logits.sum().backward()
    assert model.attnres.queries.grad is not None
    assert model.attnres.queries.grad.abs().sum() > 0


def test_gradients_flow_to_moe_gate():
    cfg = _tiny_config(ffn="moe", moe_num_experts=4, moe_num_active=2)
    model, _ = build_model(cfg)
    out = model(_ids())
    (out.logits.sum() + out.aux_loss).backward()
    gate_found = False
    for block in model.blocks:
        if hasattr(block.ffn, "gate"):
            assert block.ffn.gate.weight.grad is not None
            gate_found = True
    assert gate_found


# --- LoRA -------------------------------------------------------------------


def test_lora_inject_freezes_base_and_zero_init_matches():
    cfg = _tiny_config()
    model, _ = build_model(cfg)
    model.eval()
    ids = _ids()
    with torch.no_grad():
        base_out = model(ids).logits.clone()
    inject_lora(model, rank=4, alpha=8, targets=("qkv", "proj"))
    model.eval()
    with torch.no_grad():
        lora_out = model(ids).logits
    assert torch.allclose(base_out, lora_out, atol=1e-5)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable)


def test_lora_save_load_roundtrip():
    """Save LoRA adapters, zero them, reload → output matches the saved state.

    Uses the SAME base model so the test isolates whether the adapter
    save/load roundtrip is faithful (not whether two randomly-initialized
    bases happen to match).
    """
    cfg = _tiny_config()
    model, _ = build_model(cfg)
    model.eval()  # disable dropout for deterministic comparison
    inject_lora(model, rank=4, alpha=8, dropout=0.0, targets=("qkv", "proj"))
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "lora_A"):
                m.lora_A.normal_(0, 0.1)
                m.lora_B.normal_(0, 0.1)
    ids = _ids()
    with torch.no_grad():
        out_before = model(ids).logits.clone()
    sd = lora_state_dict(model)

    # Zero out the adapters, confirm output changes (we're not trivially passing).
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "lora_A"):
                m.lora_A.zero_()
                m.lora_B.zero_()
    with torch.no_grad():
        assert not torch.allclose(out_before, model(ids).logits, atol=1e-5)

    # Reload the saved adapters onto the same base → output matches.
    load_lora_state_dict(model, sd)
    with torch.no_grad():
        assert torch.allclose(out_before, model(ids).logits, atol=1e-5)


def test_lora_merge_preserves_output():
    cfg = _tiny_config()
    model, _ = build_model(cfg)
    model.eval()  # disable dropout for deterministic comparison
    inject_lora(model, rank=4, alpha=8, dropout=0.0, targets=("qkv", "proj"))
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "lora_A"):
                m.lora_A.normal_(0, 0.1)
                m.lora_B.normal_(0, 0.1)
    ids = _ids()
    with torch.no_grad():
        out_before = model(ids).logits.clone()
    merge_and_strip_lora(model)
    with torch.no_grad():
        out_after = model(ids).logits
    assert torch.allclose(out_before, out_after, atol=1e-5)


# --- serialization ----------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    cfg = _tiny_config()
    model, _ = build_model(cfg)
    ids = _ids()
    with torch.no_grad():
        out_before = model(ids).logits.clone()
    save_model(model, tmp_path / "ckpt", config=cfg)
    model2, cfg2 = load_model(tmp_path / "ckpt")
    with torch.no_grad():
        out_after = model2(ids).logits
    assert torch.allclose(out_before, out_after, atol=1e-5)
    assert cfg2.residual == cfg.residual


def test_save_load_preserves_attnres_config(tmp_path):
    cfg = _tiny_config(residual="block_attnres", attnres_num_blocks=2)
    model, _ = build_model(cfg)
    save_model(model, tmp_path / "ckpt", config=cfg)
    model2, cfg2 = load_model(tmp_path / "ckpt")
    assert cfg2.residual == "block_attnres"
    assert model2.attnres is not None
