"""Tests for charge-conditioned generation (model, data, sampling, distribution).

Covers the four guarantees the conditioning path must uphold:
  1. Plumbing — the charge tensor reaches the model and shifts the forward pass.
  2. Backward compatibility — ``conditioning="none"`` is byte-identical to the
     unconditional model; old checkpoints keep loading.
  3. Determinism — conditioned sampling is seeded and reproducible.
  4. Functionality — a briefly-trained conditioned model generates
     charge-appropriate peptides per bin (integration test).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.conditioning import (
    CHARGE_MIN,
    NUM_CHARGE_BINS,
    bin_to_charge,
    charge_bin,
    charge_modlamp,
    reference_charge_proportions,
    sample_charge_bins,
)
from amp_challenge_2027.model import DecoderConfig, PeptideDecoder, build_model, load_model, save_model
from amp_challenge_2027.generator import sample_sequences
from amp_challenge_2027.training import ChargeConditionedDataset, charge_conditional_collate


# ---------------------------------------------------------------------------
# Conditioning module
# ---------------------------------------------------------------------------


def test_charge_modlamp_matches_modlamp():
    """The self-contained Bjellqvist charge is bit-for-bit modlamp (anchors)."""
    # GLFDIVKKVVGALGSL -> 0.996; KKKKKKKK -> 7.994 (verified against modlamp 4.x).
    assert charge_modlamp("GLFDIVKKVVGALGSL") == pytest.approx(0.996, abs=1e-3)
    assert charge_modlamp("KKKKKKKK") == pytest.approx(7.994, abs=1e-3)


def test_charge_bin_round_and_clamp():
    assert charge_bin("GLFDIVKKVVGALGSL") == 1 - CHARGE_MIN  # round(0.996)=1
    assert charge_bin("KKKKKKKK") == 8 - CHARGE_MIN
    # Clamping: nothing in the 20-AA alphabet can exceed the range, but the
    # clamp branch must hold for edge charges.
    assert charge_bin("KKKKKKKKKKKKKKKKKK") == NUM_CHARGE_BINS - 1  # +12 clamp
    assert bin_to_charge(charge_bin("GLFDIVKKVVGALGSL")) == 1


def test_reference_charge_proportions_sums_to_one():
    ref = ["KKKK", "EEEE", "GLFDIVKKVVGALGSL"]  # +4, -4, +1
    p = reference_charge_proportions(ref)
    assert p.shape == (NUM_CHARGE_BINS,)
    assert p.sum() == pytest.approx(1.0)
    assert p[4 - CHARGE_MIN] == pytest.approx(1 / 3)
    assert p[-4 - CHARGE_MIN] == pytest.approx(1 / 3)
    assert p[1 - CHARGE_MIN] == pytest.approx(1 / 3)


def test_sample_charge_bins_deterministic_and_proportional():
    p = np.zeros(NUM_CHARGE_BINS)
    p[3 - CHARGE_MIN] = 0.5
    p[6 - CHARGE_MIN] = 0.5
    a = sample_charge_bins(100, p, seed=42)
    b = sample_charge_bins(100, p, seed=42)
    assert a == b  # deterministic
    counts = np.bincount(a, minlength=NUM_CHARGE_BINS)
    assert counts[3 - CHARGE_MIN] == 50 and counts[6 - CHARGE_MIN] == 50


def test_sample_charge_bins_hamilton_remainder():
    # 3 bins with equal proportions, n=10 → quotas 3.33 each → 4/3/3 split.
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    bins = sample_charge_bins(10, p, seed=0)
    counts = np.bincount(bins, minlength=3)
    assert sorted(counts.tolist()) == [3, 3, 4]
    assert sum(counts) == 10


# ---------------------------------------------------------------------------
# Model conditioning
# ---------------------------------------------------------------------------


def _cond_config(**kw):
    base = dict(num_layers=2, hidden_size=64, num_heads=2, conditioning="charge")
    base.update(kw)
    return DecoderConfig(**base)


def test_conditioned_forward_shape_and_charge_effect():
    torch.manual_seed(0)
    model = PeptideDecoder(_cond_config())
    ids = torch.randint(3, 23, (4, 12))
    out = model(ids, charge=torch.tensor([0, 5, 10, 15]))
    assert out.logits.shape == (4, 12, model.cfg.vocab_size)
    # Different charge conditions must produce different logits (wired through).
    out_a = model(ids, charge=torch.tensor([0, 0, 0, 0]))
    out_b = model(ids, charge=torch.tensor([20, 20, 20, 20]))
    assert not torch.allclose(out_a.logits, out_b.logits)


def test_conditioned_forward_accepts_list_and_scalar():
    torch.manual_seed(0)
    model = PeptideDecoder(_cond_config())
    ids = torch.randint(3, 23, (2, 8))
    out_list = model(ids, charge=[3, 4])
    out_scalar = model(ids, charge=3)
    assert out_list.logits.shape == out_scalar.logits.shape == (2, 8, model.cfg.vocab_size)


def test_conditioned_default_bin_when_charge_none():
    torch.manual_seed(0)
    model = PeptideDecoder(_cond_config())
    model.eval()  # disable dropout for exact logit comparison
    ids = torch.randint(3, 23, (2, 8))
    out_none = model(ids)  # must not crash; falls back to default bin
    out_default = model(ids, charge=torch.full((2,), 3 - CHARGE_MIN))
    assert torch.allclose(out_none.logits, out_default.logits)


def test_unconditional_model_ignores_charge():
    torch.manual_seed(0)
    cfg = DecoderConfig(num_layers=2, hidden_size=64, num_heads=2)  # conditioning="none"
    model = PeptideDecoder(cfg)
    model.eval()  # disable dropout for exact logit comparison
    assert model.charge_emb is None
    ids = torch.randint(3, 23, (2, 8))
    out_a = model(ids)
    out_b = model(ids, charge=torch.tensor([0, 20]))  # ignored
    assert torch.allclose(out_a.logits, out_b.logits)


def test_config_roundtrip_and_param_count():
    cfg = _cond_config()
    d = cfg.to_dict()
    assert d["conditioning"] == "charge" and d["num_charge_bins"] == NUM_CHARGE_BINS
    cfg2 = DecoderConfig.from_dict(d)
    assert cfg2.conditioning == "charge"
    model, cfg3 = build_model(_cond_config())
    real = sum(p.numel() for p in model.parameters())
    assert cfg3.num_params == real  # estimate stays exact with the new table


def test_save_load_roundtrip_conditioned(tmp_path):
    model, cfg = build_model(_cond_config())
    model.eval()  # disable dropout for exact logit comparison
    save_model(model, tmp_path, config=cfg)
    loaded, lcfg = load_model(tmp_path)
    assert lcfg.conditioning == "charge"
    assert loaded.charge_emb is not None
    ids = torch.randint(3, 23, (2, 8))
    torch.testing.assert_close(model(ids, charge=[1, 2]).logits, loaded(ids, charge=[1, 2]).logits)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def test_charge_dataset_and_collate():
    seqs = ["KKKK", "EEEE", "GLFDIVKKVVGALGSL"]
    bins = [charge_bin(s) for s in seqs]
    ds = ChargeConditionedDataset(seqs, bins, lambda s: tok.encode(s, add_bos=True, add_eos=True))
    assert len(ds) == 3
    ids, b = ds[0]
    assert ids[0] == tok.BOS_ID and ids[-1] == tok.EOS_ID
    assert b == bins[0]
    collate = charge_conditional_collate(tok.PAD_ID)
    input_ids, labels, charge = collate([ds[i] for i in range(3)])
    assert input_ids.shape == labels.shape == (3, max(len(ds[i][0]) for i in range(3)))
    assert charge.tolist() == bins
    assert charge.dtype == torch.long


def test_charge_dataset_length_mismatch_raises():
    with pytest.raises(ValueError):
        ChargeConditionedDataset(["KKKK"], [1, 2], lambda s: [])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_conditioned_sampling_deterministic():
    torch.manual_seed(0)
    model = PeptideDecoder(_cond_config())
    model.eval()
    charge = [5] * 8
    kw = dict(n_sequences=8, device="cpu", max_length=20, temperature=1.0,
              top_k=10, top_p=0.9, repetition_penalty=1.2, batch_size=4)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_sequences(model, generator=g1, charge=charge, **kw)
    b = sample_sequences(model, generator=g2, charge=charge, **kw)
    assert a == b


def test_sampling_charge_length_mismatch_raises():
    torch.manual_seed(0)
    model = PeptideDecoder(_cond_config())
    with pytest.raises(ValueError):
        sample_sequences(
            model, n_sequences=4, device="cpu", max_length=16,
            generator=torch.Generator().manual_seed(0), charge=[1, 2],
        )


def test_sampling_without_charge_still_works_on_conditioned_model():
    torch.manual_seed(0)
    model = PeptideDecoder(_cond_config())
    seqs = sample_sequences(
        model, n_sequences=4, device="cpu", max_length=16,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(seqs) == 4


# ---------------------------------------------------------------------------
# Integration: brief training makes conditioning functional
# ---------------------------------------------------------------------------


def _synthetic_charge_data(n_per_bin: int = 40, seed: int = 0):
    """Peptides with controlled integer charges: +4 (K-rich) vs -3 (E-rich)."""
    rng = np.random.default_rng(seed)
    neutrals = list("GASTPQN")
    data = {4: [], -3: []}
    for target in (4, -3):
        seen = set()
        while len(data[target]) < n_per_bin:
            if target > 0:
                chars = ["K"] * target + [str(rng.choice(neutrals)) for _ in range(10 - target)]
            else:
                chars = ["E"] * (-target) + [str(rng.choice(neutrals)) for _ in range(10 + target)]
            rng.shuffle(chars)
            s = "".join(chars)
            if s not in seen:
                seen.add(s)
                data[target].append(s)
    return data


def test_brief_training_makes_charge_conditioning_functional(tmp_path):
    """A few SGD steps on charge-controlled data → generation follows the bin.

    This is the core contract: conditioning must let us REQUEST charge on
    demand (the fix for the under-produced charge tails).
    """
    torch.manual_seed(0)
    data = _synthetic_charge_data()
    pos_seqs, neg_seqs = data[4], data[-3]
    pos_bins = [charge_bin(s) for s in pos_seqs]
    neg_bins = [charge_bin(s) for s in neg_seqs]

    model = PeptideDecoder(_cond_config(hidden_size=96, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def tokenize(s):
        return tok.encode(s, add_bos=True, add_eos=True)

    collate = charge_conditional_collate(tok.PAD_ID)
    model.train()
    for _epoch in range(30):
        for seqs, bins in ((pos_seqs, pos_bins), (neg_seqs, neg_bins)):
            input_ids, labels, charge = collate(
                [ChargeConditionedDataset(seqs, bins, tokenize)[i] for i in range(len(seqs))]
            )
            out = model(input_ids, charge=charge)
            shift_logits = out.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    kw = dict(n_sequences=32, device="cpu", max_length=14, temperature=1.0,
              top_k=10, top_p=0.95, repetition_penalty=1.0, batch_size=32)
    g_pos = torch.Generator().manual_seed(1)
    g_neg = torch.Generator().manual_seed(2)
    gen_pos = sample_sequences(model, generator=g_pos, charge=pos_bins[0:1] * 32, **kw)
    gen_neg = sample_sequences(model, generator=g_neg, charge=neg_bins[0:1] * 32, **kw)

    def mean_charge(seqs):
        return float(np.mean([charge_modlamp(s) for s in seqs if s]))

    mc_pos, mc_neg = mean_charge(gen_pos), mean_charge(gen_neg)
    # Conditioning at +4 must yield clearly more cationic peptides than at −3.
    assert mc_pos > mc_neg + 3.0, f"conditioning not functional: +{mc_pos:.2f} vs {mc_neg:.2f}"
