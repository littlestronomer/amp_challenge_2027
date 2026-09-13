"""Multi-axis conditioned SFT → generation: model plumbing + e2e integration.

Pins the guarantees the multi-axis conditioning path must uphold:
  1. Config/state compatibility — legacy "charge" checkpoints keep their exact
     state-dict keys and load unchanged; new axes add ``<axis>_emb`` weights.
  2. Plumbing — each axis's bins shift the forward pass; unknown axes and
     double conditioning inputs fail closed.
  3. End-to-end — ``train_sft(--conditioning charge,hydro,hmoment)`` on a toy
     corpus produces a checkpoint whose generation auto-draws joint bins from
     the reference.

Skips in the minimal dev environment; runs with ``--extra ml``.
"""

from __future__ import annotations

import csv
import json

import pytest

torch = pytest.importorskip("torch")

from train_generator import train_sft  # flat import via tests/conftest.py sys.path

from amp_challenge_2027.conditioning import (  # noqa: E402
    NUM_HMOMENT_BINS,
    NUM_HYDRO_BINS,
    parse_conditioning_axes,
    sample_condition_bins,
)
from amp_challenge_2027.generate import generate_with_model  # noqa: E402
from amp_challenge_2027.model import DecoderConfig, PeptideDecoder  # noqa: E402
from amp_challenge_2027.tokenizer import RESIDUE_TO_ID  # noqa: E402

MOTIFS = ["KLLAKLLAKLLA", "GVKLAGVKLAGV", "RRWFKLLAKLLA", "KLAGVKLAGVKK"]


def _synthetic_csv(path, n_per_motif: int = 20) -> list[str]:
    seqs = []
    for m in MOTIFS:
        for cut in range(n_per_motif):
            ln = 8 + cut % 5
            seqs.append((m * 2)[:ln])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence"])
        for s in seqs:
            w.writerow([s])
    return seqs


def _tiny_config(conditioning: str) -> DecoderConfig:
    return DecoderConfig(
        vocab_size=len(RESIDUE_TO_ID) + 4,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=16,
        conditioning=conditioning,
    )


IDS = torch.tensor([[RESIDUE_TO_ID["K"], RESIDUE_TO_ID["L"], RESIDUE_TO_ID["A"]]])


# ---------------------------------------------------------------------------
# Config + state compatibility
# ---------------------------------------------------------------------------


def test_config_roundtrip_multi_axis():
    cfg = _tiny_config("hmoment,charge")
    assert parse_conditioning_axes(cfg.conditioning) == ("charge", "hmoment")
    restored = DecoderConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_state_dict_keys_legacy_charge_unchanged():
    torch.manual_seed(0)
    legacy = PeptideDecoder(_tiny_config("charge"))
    keys = set(legacy.state_dict())
    assert "charge_emb.weight" in keys
    assert not any(k.startswith("hydro_emb") or k.startswith("hmoment_emb") for k in keys)
    multi = PeptideDecoder(_tiny_config("charge,hydro,hmoment"))
    multi_keys = set(multi.state_dict())
    assert {"charge_emb.weight", "hydro_emb.weight", "hmoment_emb.weight"} <= multi_keys
    # the legacy keys are a subset — old checkpoints load into same-config models
    assert keys <= multi_keys


def test_num_params_tracks_axis_bins():
    base = PeptideDecoder(_tiny_config("none"))
    multi = PeptideDecoder(_tiny_config("charge,hydro,hmoment"))
    expected_extra = (21 + NUM_HYDRO_BINS + NUM_HMOMENT_BINS) * 32
    assert sum(p.numel() for p in multi.parameters()) - sum(p.numel() for p in base.parameters()) == expected_extra


def test_invalid_conditioning_fails_at_build():
    with pytest.raises(ValueError):
        PeptideDecoder(_tiny_config("charge,gravvy"))


# ---------------------------------------------------------------------------
# Forward plumbing
# ---------------------------------------------------------------------------


def test_each_axis_influences_logits():
    torch.manual_seed(0)
    model = PeptideDecoder(_tiny_config("charge,hydro,hmoment")).eval()
    with torch.no_grad():
        low = model(IDS, cond_bins={"hydro": torch.tensor([0])}).logits[0, -1]
        high = model(IDS, cond_bins={"hydro": torch.tensor([NUM_HYDRO_BINS - 1])}).logits[0, -1]
    assert not torch.allclose(low, high), "hydro bins must influence the output head"
    with torch.no_grad():
        low = model(IDS, cond_bins={"hmoment": torch.tensor([0])}).logits[0, -1]
        high = model(IDS, cond_bins={"hmoment": torch.tensor([NUM_HMOMENT_BINS - 1])}).logits[0, -1]
    assert not torch.allclose(low, high), "hmoment bins must influence the output head"


def test_cond_bins_equivalent_to_legacy_charge_kwarg():
    torch.manual_seed(0)
    model = PeptideDecoder(_tiny_config("charge")).eval()
    bins = torch.tensor([4, 7])
    ids = IDS.expand(2, 3).contiguous()
    with torch.no_grad():
        legacy = model(ids, charge=bins).logits
        via_dict = model(ids, cond_bins={"charge": bins}).logits
    assert torch.allclose(legacy, via_dict)


def test_default_bins_used_when_none_passed():
    torch.manual_seed(0)
    model = PeptideDecoder(_tiny_config("charge,hydro")).eval()
    with torch.no_grad():
        unconditioned = model(IDS).logits
        with_defaults = model(
            IDS,
            cond_bins={
                "charge": torch.tensor([model.default_charge_bin]),
                "hydro": torch.tensor([model.default_hydro_bin]),
            },
        ).logits
    assert torch.allclose(unconditioned, with_defaults)


def test_fail_closed_on_bad_conditioning_inputs():
    model = PeptideDecoder(_tiny_config("charge,hydro")).eval()
    with pytest.raises(ValueError, match="not both"):
        model(IDS, charge=torch.tensor([0]), cond_bins={"charge": torch.tensor([0])})
    with pytest.raises(ValueError, match="inactive axes"):
        model(IDS, cond_bins={"hmoment": torch.tensor([0])})
    with pytest.raises(ValueError, match="batch"):
        model(IDS, cond_bins={"hydro": torch.tensor([0, 1])})


def test_legacy_charge_kwarg_ignored_when_charge_not_trained():
    """charge= keeps its legacy ignore semantics on models without the axis."""
    torch.manual_seed(0)
    model = PeptideDecoder(_tiny_config("hydro")).eval()
    with torch.no_grad():
        plain = model(IDS).logits
        with_charge = model(IDS, charge=torch.tensor([7])).logits
    assert torch.allclose(plain, with_charge)


# ---------------------------------------------------------------------------
# End-to-end: multi-axis SFT → checkpoint → joint-bin generation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_multiaxis_checkpoint(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("multiaxis")
    data_csv = tmp_path / "gen.csv"
    _synthetic_csv(data_csv)
    out_dir = tmp_path / "ckpt"
    train_sft(
        data_path=data_csv,
        epochs=3,
        batch_size=8,
        lr=3e-3,
        seed=42,
        device="cpu",
        out_dir=out_dir,
        residual="standard",
        ffn="dense",
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        precision="fp32",
        save_every=0,
        eval_every=0,
        patience=None,
        log_dir=str(tmp_path / "logs"),
        wandb_project=None,
        num_workers=0,
        resume=True,
        conditioning="charge,hydro,hmoment",
    )
    return out_dir, data_csv


def test_multiaxis_sft_writes_canonical_config(trained_multiaxis_checkpoint):
    out_dir, _ = trained_multiaxis_checkpoint
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["conditioning"] == "charge,hydro,hmoment"
    assert cfg["num_hydro_bins"] == NUM_HYDRO_BINS
    assert (out_dir / "model.pt").exists()


def test_multiaxis_generation_autodraws_joint_bins(trained_multiaxis_checkpoint):
    out_dir, data_csv = trained_multiaxis_checkpoint
    ref = set(_synthetic_csv(data_csv))
    generated = generate_with_model(
        24, seed=123, length=12, device="cpu", checkpoint_dir=out_dir, reference_set=ref
    )
    assert len(generated) == 24
    for s in generated:
        assert s and all(c in RESIDUE_TO_ID for c in s), f"invalid residue in {s!r}"


def test_multiaxis_generation_requires_reference(trained_multiaxis_checkpoint):
    out_dir, _ = trained_multiaxis_checkpoint
    with pytest.raises(ValueError, match="reference"):
        generate_with_model(4, seed=1, length=12, device="cpu", checkpoint_dir=out_dir, reference_set=None)


def test_sample_sequences_cond_bins_alignment(trained_multiaxis_checkpoint):
    from amp_challenge_2027.generator import load_model, sample_sequences

    out_dir, _ = trained_multiaxis_checkpoint
    model, _config = load_model(out_dir, map_location="cpu")
    draws = sample_condition_bins(8, sorted(MOTIFS), axes=("charge", "hydro", "hmoment"), seed=2)
    bins = {axis: [draw[axis] for draw in draws] for axis in ("charge", "hydro", "hmoment")}
    g = torch.Generator().manual_seed(0)
    seqs = sample_sequences(
        model, n_sequences=8, device="cpu", max_length=12, generator=g, cond_bins=bins, batch_size=4
    )
    assert len(seqs) == 8
    with pytest.raises(ValueError, match="must align"):
        sample_sequences(model, n_sequences=8, device="cpu", max_length=12, generator=g,
                         cond_bins={"hydro": [0, 1]})
    with pytest.raises(ValueError, match="not both"):
        sample_sequences(model, n_sequences=8, device="cpu", max_length=12, generator=g,
                         charge=[0] * 8, cond_bins=bins)
