"""End-to-end integration test for charge-conditioned SFT → generation.

Exercises the real pipeline on a toy problem (CPU, fp32, seconds):

    synthetic CSV → train_sft(--conditioning charge) → checkpoint
                  → generate_with_model (auto-detects conditioning)
                  → reference-drawn charge bins threaded to sampling

plus the mechanics the conditioned path depends on: the charge embedding must
actually change the model's predictions, and resume-from-checkpoint must work
with the epoch_start_step bookkeeping introduced in the trainer fixes.

Skips in the minimal dev environment; runs with ``--extra ml``.
"""

from __future__ import annotations

import csv
import json

import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.conditioning import (  # noqa: E402
    reference_charge_proportions,
    sample_charge_bins,
)
from amp_challenge_2027.generate import generate_with_model  # noqa: E402
from amp_challenge_2027.tokenizer import RESIDUE_TO_ID  # noqa: E402
from scripts.train_generator import train_sft  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOTIFS = ["KLLAKLLAKLLA", "GVKLAGVKLAGV", "RRWFKLLAKLLA", "KLAGVKLAGVKK"]


def _synthetic_csv(path, n_per_motif: int = 20) -> list[str]:
    """Highly structured peptides so a tiny model shows clear loss progress."""
    seqs = []
    for m in MOTIFS:
        for cut in range(n_per_motif):
            ln = 8 + cut % 5  # lengths 8..12
            seqs.append((m * 2)[:ln])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence"])
        for s in seqs:
            w.writerow([s])
    return seqs


@pytest.fixture()
def trained_checkpoint(tmp_path):
    data_csv = tmp_path / "gen.csv"
    _synthetic_csv(data_csv)
    out_dir = tmp_path / "ckpt"
    common = dict(
        data_csv,
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
        precision="fp32",  # cuda autocast would fail on CPU
        save_every=0,
        eval_every=0,
        patience=None,
        log_dir=str(tmp_path / "logs"),
        wandb_project=None,
        num_workers=0,
        resume=True,
        conditioning="charge",
    )
    train_sft(**common)
    return out_dir, tmp_path, common


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_conditioned_sft_writes_conditioned_checkpoint(trained_checkpoint):
    out_dir, _tmp, _common = trained_checkpoint
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["conditioning"] == "charge"
    assert (out_dir / "model.pt").exists()


def test_resume_from_checkpoint_continues(trained_checkpoint):
    """Second train_sft call resumes past finished epochs via ckpt_epoch*."""
    out_dir, tmp_path, common = trained_checkpoint
    ckpts = sorted((out_dir / "checkpoints").glob("ckpt_epoch*.pt"))
    assert ckpts, "per-epoch checkpoints expected"
    resumed = dict(common)
    resumed["epochs"] = common["epochs"] + 1  # one more epoch on top of resume
    train_sft(**resumed)
    newer = sorted((out_dir / "checkpoints").glob("ckpt_epoch*.pt"))
    assert len(newer) > len(ckpts), "resume should produce additional checkpoints"


def test_generation_autodetects_and_uses_reference_bins(trained_checkpoint):
    out_dir, tmp_path, _common = trained_checkpoint
    seqs = _synthetic_csv(tmp_path / "gen.csv")
    ref = set(seqs)

    generated = generate_with_model(
        24, seed=123, length=12, device="cpu", checkpoint_dir=out_dir, reference_set=ref
    )
    assert len(generated) == 24
    for s in generated:
        assert s, "conditioned sampling must produce non-empty sequences"
        assert all(c in RESIDUE_TO_ID for c in s), f"invalid residue in {s!r}"


def test_charge_embedding_changes_predictions():
    """Same input, different conditioning bins ⇒ different next-token logits."""
    from amp_challenge_2027.model import DecoderConfig, PeptideDecoder

    cfg = DecoderConfig(
        vocab_size=len(RESIDUE_TO_ID) + 4,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=16,
        conditioning="charge",
    )
    torch.manual_seed(0)
    model = PeptideDecoder(cfg).eval()

    ids = torch.tensor([[RESIDUE_TO_ID["K"], RESIDUE_TO_ID["L"], RESIDUE_TO_ID["A"]]])
    with torch.no_grad():
        low = model(ids, charge=torch.tensor([0])).logits[0, -1]
        high = model(ids, charge=torch.tensor([15])).logits[0, -1]
    assert not torch.allclose(low, high), "charge bins must influence the output head"


def test_reference_bin_sampling_matches_distribution():
    """sample_charge_bins reproduces reference proportions (Hamilton appt)."""
    ref = [m[:10] for m in MOTIFS for _ in range(10)]  # skewed toward K-rich motifs
    props = reference_charge_proportions(ref)
    drawn = sample_charge_bins(2000, props, seed=7)
    hist = torch.bincount(torch.tensor(drawn), minlength=len(props)).float()
    got = hist / hist.sum()
    assert torch.allclose(got, torch.tensor(props, dtype=torch.float32), atol=0.01)
    assert all(0 <= b < len(props) for b in drawn)
