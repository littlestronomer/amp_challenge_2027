"""Behavioral checks for biological conditioning and versioned research models."""
from __future__ import annotations

# ruff: noqa: E402
import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from amp_challenge_2027.assay_conditioning import (
    CATEGORICAL_FIELDS,
    collate_assay_conditions,
    condition_support,
    fit_condition_schema,
)
from amp_challenge_2027.model import (
    DecoderConfig,
    MoEFFN,
    PeptideDecoder,
    is_assay_parameter,
    load_model,
    save_model,
    warm_start_model,
)


def condition(value=8.0):
    return {"endpoint": "MIC", "target": "E. coli", "value_lower": value,
            "value_upper": value, "unit": "uM", "operator": "=",
            "chemistry": "canonical_free_linear_L"}


def schema():
    return fit_condition_schema([
        {"split": "train", "conditions": [condition(2.0), condition(32.0)]},
        {"split": "train", "conditions": [{"endpoint": "%lysis", "target": "human",
         "value_upper": 10.0, "dose_lower": 64.0, "rbc_species": "human", "unit": "%"}]},
    ])


def tiny(**kwargs):
    values = dict(hidden_size=16, num_heads=2, num_layers=3,
                  max_position_embeddings=12, dropout=0.0, attnres_num_blocks=2)
    values.update(kwargs)
    return DecoderConfig(**values)


def test_schema_only_fits_training_and_preserves_missingness():
    with pytest.raises(ValueError, match="training split"):
        fit_condition_schema([{"split": "test", "conditions": [condition()]}])
    fitted = schema()
    assert "publication_id" not in fitted["categorical_fields"]
    assert json.loads(json.dumps(fitted)) == fitted
    encoded = collate_assay_conditions([[condition()], []], fitted, "cpu")
    assert encoded["valid_mask"].tolist() == [[True, True], [True, False]]
    assert not encoded["observed"][1].any()
    warnings = condition_support([condition(1e6)], fitted)
    assert any(item["reason"] == "outside_training_range" for item in warnings)
    unknown = collate_assay_conditions([[{"target": "unseen organism"}]], fitted)
    index = CATEGORICAL_FIELDS.index("target")
    assert unknown["categorical"][0, 1, index] == 1
    assert condition_support([{"target": "unseen organism"}], fitted)[0]["reason"] == "unseen_category"


def test_condition_dropout_cannot_drop_null_or_leave_values():
    fitted = schema()
    encoded = collate_assay_conditions([[condition(), condition(32)]], fitted, dropout=1.0)
    assert encoded["valid_mask"].tolist() == [[True, False, False]]
    assert not encoded["observed"].any()
    assert not encoded["values"].any()
    assert not encoded["categorical"].any()


def legacy_reference(model, ids):
    """Frozen v1 branch topology, independent of the new decoder dispatcher."""
    cfg = model.cfg
    h = model.drop(model.tok_emb(ids) + model.pos_emb(torch.arange(ids.shape[1])[None]))
    sources, partial = [], None
    if cfg.residual == "block_attnres":
        sources.append(h)
    for index, block in enumerate(model.blocks):
        if cfg.residual == "attnres":
            if index:
                h = model.attnres.aggregate(sources, None, 2 * index)
            sources.append(h)
        elif cfg.residual == "block_attnres" and partial is None:
            h = model.attnres.aggregate(sources, None, 2 * index)
        delta, _ = block.attn(block.ln1(h))
        h = h + block.drop(delta)
        delta, _ = block.ffn(block.ln2(h))
        h = h + block.drop(delta)
        if cfg.residual == "block_attnres":
            partial = h if partial is None else partial + h
            if (index + 1) % model.block_size == 0:
                sources.append(h)
                partial = None
    return model.ln_f(h) @ model.tok_emb.weight.t()


@pytest.mark.parametrize("residual", ["standard", "attnres", "block_attnres"])
def test_old_checkpoint_and_zero_gate_warm_start_preserve_logits_and_samples(tmp_path, residual):
    torch.manual_seed(11)
    source = PeptideDecoder(tiny(residual=residual)).eval()
    save_model(source, tmp_path)
    # Represent actual old configs: no implementation-version or assay field.
    config_file = tmp_path / "config.json"
    config = json.loads(config_file.read_text())
    for key in ("residual_impl_version", "assay_schema", "attnres_block_layers", "moe_expert_inner"):
        config.pop(key)
    config_file.write_text(json.dumps(config))
    old, cfg = load_model(tmp_path)
    assert cfg.residual_impl_version == 1
    ids = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    assert torch.equal(old(ids).logits, legacy_reference(old, ids))
    model, _ = warm_start_model(tmp_path, schema())
    conditions = collate_assay_conditions([[condition()], [condition(32)]], schema())
    assert torch.equal(source(ids).logits, model(ids, assay_conditions=conditions).logits)
    for decoder, kwargs in ((old, {}), (model, {"assay_conditions": conditions})):
        generator = torch.Generator().manual_seed(2027)
        sampled = torch.ones((2, 1), dtype=torch.long)
        for _ in range(5):
            p = decoder(sampled, **kwargs).logits[:, -1].softmax(-1)
            sampled = torch.cat((sampled, torch.multinomial(p, 1, generator=generator)), dim=1)
        if decoder is old:
            expected = sampled
        else:
            assert torch.equal(sampled, expected)


def test_warm_start_rejects_missing_backbone_weight(tmp_path):
    source = PeptideDecoder(tiny())
    save_model(source, tmp_path)
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    state.pop("blocks.0.attn.qkv.weight")
    torch.save(state, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="missing_backbone"):
        warm_start_model(tmp_path, schema())


def test_gate_then_condition_encoder_receive_gradients_and_conditions_have_effect():
    torch.manual_seed(23)
    model = PeptideDecoder(tiny(assay_schema=schema()))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_assay_parameter(name))
    ids = torch.tensor([[1, 4, 5, 6]])
    conditions = collate_assay_conditions([[condition(2)]], schema())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(ids, assay_conditions=conditions).logits.square().mean().backward()
    assert any(block.cross_attn.gate.grad.abs() > 0 for block in model.blocks)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.assay_encoder.parameters())
    optimizer.step()
    optimizer.zero_grad()
    model(ids, assay_conditions=conditions).logits.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.assay_encoder.parameters())
    changed = collate_assay_conditions([[condition(32)]], schema())
    assert not torch.equal(model(ids, assay_conditions=conditions).logits,
                           model(ids, assay_conditions=changed).logits)


def test_masked_nans_and_arbitrary_padded_features_cannot_leak_into_attention():
    torch.manual_seed(6)
    model = PeptideDecoder(tiny(assay_schema=schema())).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.cross_attn.gate.fill_(0.5)
    batch = collate_assay_conditions([[condition()], []], schema())
    ids = torch.tensor([[1, 4, 5], [1, 6, 7]])
    before = model(ids, assay_conditions=batch).logits
    batch["values"].masked_fill_(~batch["observed"], float("nan"))
    batch["categorical"].masked_fill_(~batch["valid_mask"][..., None], 999999)
    after = model(ids, assay_conditions=batch).logits
    assert torch.isfinite(after).all()
    assert torch.equal(before, after)
    batch["values"][0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(ids, assay_conditions=batch)


@pytest.mark.parametrize("residual", ["standard", "attnres", "block_attnres"])
@pytest.mark.parametrize("ffn", ["dense", "moe"])
def test_conditional_v2_causality_and_save_load(tmp_path, residual, ffn):
    torch.manual_seed(1)
    model = PeptideDecoder(tiny(residual=residual, residual_impl_version=2, ffn=ffn,
        moe_num_experts=4, moe_num_active=2, moe_expert_inner=32, assay_schema=schema())).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.cross_attn.gate.fill_(0.2)
    ids = torch.tensor([[1, 4, 5, 6, 7], [1, 8, 9, 10, 11]])
    conditions = collate_assay_conditions([[condition(2)], [condition(32)]], schema())
    before = model(ids, assay_conditions=conditions).logits
    changed = ids.clone()
    changed[:, 3:] = 15
    after = model(changed, assay_conditions=conditions).logits
    # Expert dispatch batches change when later tokens route differently; GEMM
    # roundoff can differ by a few ulps even though dependency is causal.
    assert torch.allclose(before[:, :3], after[:, :3], atol=5e-6)
    captured = []
    def retain_embeddings(module, inputs, output):
        output.retain_grad()
        captured.append(output)
    handle = model.tok_emb.register_forward_hook(retain_embeddings)
    model(ids, assay_conditions=conditions).logits[:, :3].sum().backward()
    handle.remove()
    assert torch.count_nonzero(captured[0].grad[:, 3:]) == 0
    save_model(model, tmp_path)
    loaded, cfg = load_model(tmp_path)
    assert cfg.assay_schema == schema()
    assert cfg.residual_impl_version == 2
    assert torch.equal(before, loaded(ids, assay_conditions=conditions).logits)
    # Row permutation checks condition/sequence batch alignment.
    permuted = {key: value.flip(0) for key, value in conditions.items()}
    assert torch.allclose(before.flip(0), model(ids.flip(0), assay_conditions=permuted).logits, atol=5e-6)


class ConstantAttention(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, h):
        return self.value.expand_as(h), None


class ConstantFFN(ConstantAttention):
    def forward(self, h, **kwargs):
        return self.value.expand_as(h), h.new_zeros(())


@pytest.mark.parametrize("residual, denominator", [("attnres", 7), ("block_attnres", 3)])
def test_v2_aggregates_branch_deltas_and_partial_at_final_readout(residual, denominator):
    model = PeptideDecoder(tiny(residual=residual, residual_impl_version=2,
                               attnres_block_layers=2))
    for index, block in enumerate(model.blocks):
        block.attn = ConstantAttention(2 * index + 1)
        block.ffn = ConstantFFN(2 * index + 2)
    embedding = torch.arange(16, dtype=torch.float32)[None, None].expand(1, 2, -1)
    h, _ = model._forward_attnres_v2(embedding, None, None, torch.ones((1, 2), dtype=torch.bool))
    # Full: embedding + six deltas; Block: embedding + completed four-delta
    # sum + unfinished two-delta partial. A skipped final aggregate fails this.
    expected = (embedding + 21) / denominator
    assert torch.allclose(h, expected)
    h.square().sum().backward()
    assert model.blocks[0].attn.value.grad.abs() > 0
    assert model.attnres.queries.shape[0] == 7
    assert model.attnres.queries.grad[-1].abs().sum() > 0


def test_moe_auxiliary_loss_has_router_gradient_and_excludes_padding():
    torch.manual_seed(7)
    moe = MoEFFN(4, 8, 4, 2, 0.1)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.tensor([[3.] * 4, [1.] * 4, [-1.] * 4, [-2.] * 4]))
    values = torch.ones((1, 4, 4))
    valid = torch.tensor([[True, True, False, False]])
    _, auxiliary = moe(values, valid_mask=valid)
    auxiliary.backward()
    assert moe.gate.weight.grad.abs().sum() > 0
    assert all(p.grad is None for expert in moe.experts for p in expert.parameters())
    assert torch.allclose(moe.routing_stats["assignment_fraction"], torch.tensor([0.5, 0.5, 0., 0.]))
    assert moe.routing_stats["valid_token_count"] == 2
    _, reference = moe(values[:, :2])
    values[:, 2:] = 1e4
    output, padded = moe(values, valid_mask=valid)
    assert torch.equal(padded, reference)
    assert not output[:, 2:].any()
    output, auxiliary = moe(values, valid_mask=torch.zeros_like(valid))
    assert auxiliary == 0 and torch.isfinite(auxiliary)
    assert not output.any()


def test_top_two_dispatch_normalizes_weights_and_drops_no_valid_tokens():
    torch.manual_seed(8)
    moe = MoEFFN(4, 8, 4, 2, 0.1)
    for expert in moe.experts:
        with torch.no_grad():
            expert.c_fc.weight.zero_()
            expert.c_fc.bias.zero_()
            expert.c_proj.weight.zero_()
            expert.c_proj.bias.fill_(3.0)
    values = torch.randn((2, 6, 4))
    result, _ = moe(values)
    # Identical constant experts must produce the same constant for every token,
    # regardless of routing decisions or top-two probability split.
    assert torch.allclose(result, torch.full_like(result, 3.0))
    assert torch.allclose(moe.routing_stats["assignment_fraction"].sum(), torch.tensor(1.0))
    assert moe.routing_stats["valid_token_count"] == 12
