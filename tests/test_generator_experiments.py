"""Regression checks for checkpoint bookkeeping and controlled library sweeps."""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import pytest
from experiment_utils import mark_files, prepare_run, verify_files
from sweep_checkpoints import choose_snapshots
from sweep_library_selection import main as selection_main
from sweep_library_selection import panel_artifacts, read_scores

from amp_challenge_2027.library_selection import select_stratified_library, stratum
from amp_challenge_2027.training import EarlyStoppingState


def test_patience_resets_and_nonfinite_loss_is_rejected():
    state = EarlyStoppingState(patience=2)
    assert state.observe(4, 1)
    assert not state.observe(5, 2)
    assert state.observe(3, 3)
    assert state.bad_evaluations == 0
    assert not state.observe(4, 4)
    assert not state.should_stop
    assert not state.observe(4, 5)
    assert state.should_stop and state.best_step == 3
    with pytest.raises(ValueError, match="non-finite"):
        state.observe(float("nan"), 6)


def test_snapshot_selection_is_numeric_and_missing_epochs_fail(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for epoch in (1, 2, 10, 20, 30, 40):
        (checkpoints / f"ckpt_epoch{epoch}.pt").touch()
    (checkpoints / "ckpt_500.pt").touch()
    selected = choose_snapshots(tmp_path, epochs=None, max_snapshots=3)
    assert [p.name for p in selected] == ["ckpt_epoch2.pt", "ckpt_epoch20.pt", "ckpt_epoch40.pt"]
    assert choose_snapshots(tmp_path, epochs=None, max_snapshots=0) == []
    with pytest.raises(ValueError, match="not saved"):
        choose_snapshots(tmp_path, epochs=[19], max_snapshots=3)


BASELINE = ["KALGLALA", "KLAGLALA", "KAGLLALA", "KGALLALA", "KKALGLALA", "KKLAGLALA"]
ALTERNATIVES = ["KLGALALA", "KLLAGALA", "KKGALLALA"]


def test_replacements_preserve_joint_bins_and_budget_and_improve_score():
    pool = BASELINE + ALTERNATIVES
    scores = np.asarray([1, 2, 3, 4, 5, 6, 10, 11, 12], dtype=float)
    output = select_stratified_library(BASELINE, pool, scores, strength=0.5)
    assert len(output) == len(set(output)) == len(BASELINE)
    assert Counter(map(stratum, output)) == Counter(map(stratum, BASELINE))
    assert len(set(output) - set(BASELINE)) == 3
    by_sequence = dict(zip(pool, scores))
    assert sum(by_sequence[s] for s in output) > sum(by_sequence[s] for s in BASELINE)
    assert output == select_stratified_library(BASELINE, pool, scores, strength=0.5)
    assert select_stratified_library(BASELINE, pool, scores, strength=0) == BASELINE


def test_tied_or_worse_alternatives_retain_baseline_members():
    pool = BASELINE + ALTERNATIVES
    output = select_stratified_library(BASELINE, pool, np.ones(len(pool)), strength=1)
    assert set(output) == set(BASELINE)


@pytest.mark.parametrize("scores", [np.array([1.0]), np.full(len(BASELINE), np.nan)])
def test_bad_scores_rejected(scores):
    with pytest.raises(ValueError, match="scores"):
        select_stratified_library(BASELINE, BASELINE, scores, strength=0.5)


def test_reused_directory_and_changed_cached_artifacts_rejected(tmp_path):
    out = tmp_path / "experiment"
    prepare_run(out, {"weights": "abc"})
    prepare_run(out, {"weights": "abc"})
    with pytest.raises(ValueError, match="changed"):
        prepare_run(out, {"weights": "def"})
    (out / "library.fasta").write_text("original")
    mark_files(out, "done.json", ["library.fasta"])
    assert verify_files(out, "done.json")
    (out / "library.fasta").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_files(out, "done.json")


def test_missing_panel_config_never_uses_shared_stale_metadata(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"unfreeze_layers": 4}))
    with pytest.raises(ValueError, match="actual frozen member"):
        panel_artifacts(tmp_path)


def test_stale_panel_config_is_rejected_even_when_per_artifact(tmp_path):
    (tmp_path / "classifier_panel_config.json").write_text(json.dumps({
        "task": "panel", "unfreeze_layers": 4,
        "checkpoint_format": "head-only", "temperature": 4.6,
    }))
    with pytest.raises(ValueError, match="frozen head"):
        panel_artifacts(tmp_path)


def test_scores_are_aligned_by_sequence_and_missing_rows_fail(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("sequence,score\nKALGLALA,1\nKLAGLALA,2\n")
    assert read_scores(path, ["KLAGLALA", "KALGLALA"]).tolist() == [2, 1]
    with pytest.raises(ValueError, match="missing"):
        read_scores(path, BASELINE)


def test_library_sweep_cli_preserves_anchor_bytes_and_resumes(tmp_path):
    from amp_challenge_2027.data import write_fasta

    baseline = tmp_path / "baseline.fasta"
    baseline.write_text("".join(f">custom-{i}\n{s}\n" for i, s in enumerate(BASELINE)))
    pool = tmp_path / "pool.fasta"
    write_fasta(BASELINE + ALTERNATIVES, pool)
    reference = tmp_path / "reference.fasta"
    reference.write_text(">ref\nRRRRRRRR\n")
    scores = tmp_path / "scores.csv"
    scores.write_text("sequence,score\n" + "".join(f"{s},{i}\n" for i, s in enumerate(BASELINE + ALTERNATIVES)))
    out = tmp_path / "out"
    args = ["--baseline", str(baseline), "--pools", str(pool), "--reference", str(reference),
            "--scores", str(scores), "--library-size", "6", "--strengths", "0", "0.5",
            "--out", str(out), "--generate-only", "--device", "cpu"]
    selection_main(args)
    assert (out / "strength_0/library.fasta").read_bytes() == baseline.read_bytes()
    before = (out / "strength_0.5/library.fasta").read_bytes()
    selection_main(args)
    assert (out / "strength_0.5/library.fasta").read_bytes() == before
    with (out / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and int(rows[1]["replaced"]) == 3


def test_sft_keeps_best_when_periodic_and_final_saves_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import train_generator as trainer

    from amp_challenge_2027.generator import load_model

    data = tmp_path / "training.csv"
    data.write_text("sequence\n" + "KALGLALA\n" * 48)
    expected = {}
    losses = iter([4.0, 5.0, 3.0, 4.0, 5.0])

    def controlled_validation(model, *_args):
        loss = next(losses)
        if loss == 3:
            expected.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        return loss

    monkeypatch.setattr(trainer, "_evaluate", controlled_validation)
    out = tmp_path / "model"
    trainer.train_sft(data, epochs=2, batch_size=8, device="cpu", precision="fp32",
                      num_layers=1, hidden_size=16, num_heads=4, num_workers=0,
                      out_dir=out, save_every=1, eval_every=1, patience=2, resume=False,
                      split_seed=123)
    selected, _ = load_model(out)
    last, _ = load_model(out / "last")
    assert all(torch.equal(selected.state_dict()[k], v) for k, v in expected.items())
    assert any(not torch.equal(last.state_dict()[k], v) for k, v in expected.items())
    assert json.loads((out / "selection.json").read_text())["step"] == 3
    stopped = torch.load(out / "checkpoints/ckpt_stop_5.pt", weights_only=True)
    assert stopped["extra"]["bad_evaluations"] == 2
    assert stopped["extra"]["best_step"] == 3


def test_fixed_split_is_independent_of_model_seed(tmp_path):
    pytest.importorskip("torch")
    from train_generator import train_sft

    data = tmp_path / "training.csv"
    data.write_text("sequence\n" + "".join(s + "\n" for s in (BASELINE + ALTERNATIVES) * 5))
    manifests = []
    for seed in (42, 43):
        out = tmp_path / f"seed{seed}"
        train_sft(data, epochs=0, device="cpu", precision="fp32", num_layers=1,
                  hidden_size=16, num_heads=4, num_workers=0, out_dir=out,
                  eval_every=0, resume=False, seed=seed, split_seed=123)
        manifests.append(json.loads((out / "training_manifest.json").read_text()))
    assert manifests[0]["val_sha256"] == manifests[1]["val_sha256"]
    assert manifests[0]["train_sha256"] == manifests[1]["train_sha256"]


@pytest.mark.parametrize("resume", [False, True])
def test_sft_rejects_reusing_legacy_checkpoints_without_touching_them(tmp_path, resume):
    pytest.importorskip("torch")
    from train_generator import train_sft

    data = tmp_path / "training.csv"
    data.write_text("sequence\n" + "KALGLALA\n" * 24)
    out = tmp_path / "legacy"
    checkpoint = out / "checkpoints/ckpt_epoch1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing training artifact")
    with pytest.raises(ValueError, match="new --out-dir"):
        train_sft(data, epochs=0, device="cpu", num_layers=1, hidden_size=16,
                  num_heads=4, num_workers=0, out_dir=out, resume=resume)
    assert checkpoint.read_bytes() == b"existing training artifact"
    assert not (out / "training_manifest.json").exists()


def test_checkpoint_sweep_exports_and_generates_without_a_fallback(tmp_path):
    torch = pytest.importorskip("torch")
    from sweep_checkpoints import main as checkpoint_main

    from amp_challenge_2027.data import iter_fasta
    from amp_challenge_2027.generator import build_model, save_model
    from amp_challenge_2027.model import DecoderConfig
    from amp_challenge_2027.training import save_checkpoint

    run = tmp_path / "training"
    torch.manual_seed(123)
    model, cfg = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=4))
    # A uniform-logit decoder exercises real sampling without depending on an
    # untrained random model producing enough non-degenerate peptides.
    with torch.no_grad():
        model.tok_emb.weight.zero_()
    save_model(model, run, config=cfg)
    save_checkpoint(run / "checkpoints/ckpt_epoch1.pt", model=model,
                    optimizer=torch.optim.AdamW(model.parameters()), step=2, epoch=1)
    reference = tmp_path / "reference.fasta"
    reference.write_text(">ref\nRRRRRRRR\n")
    out = tmp_path / "sweep"
    args = ["--checkpoint-dir", str(run), "--epochs", "1", "--library-size", "20",
            "--raw-count", "80", "--reference", str(reference), "--device", "cpu",
            "--out", str(out), "--generate-only"]
    checkpoint_main(args)
    anchor = out / "inference/seed42/library.fasta"
    archive = out / "ckpt_epoch1/seed42/library.fasta"
    assert anchor.read_bytes() == archive.read_bytes()
    assert len(list(iter_fasta(archive))) == 20
    checkpoint_main(args)  # Complete stages resume from verified output hashes.


def test_strict_metric_builder_rejects_partial_protocol(monkeypatch):
    import sys
    from types import SimpleNamespace

    from amp_challenge_2027.metrics_official import build_metric_list

    names = ["Uniqueness", "Novelty", "Diversity", "Length", "NGramJaccardSimilarity",
             "FBD", "MMD", "FKEA", "Precision", "Recall", "ConformityScore", "AuthPct"]
    metrics = SimpleNamespace(**{
        name: (lambda name=name, **_kwargs: SimpleNamespace(name=name)) for name in names
    })
    models = SimpleNamespace(**{name: lambda: object() for name in [
        "Charge", "Hydrophobicity", "HydrophobicMoment", "Amphiphilicity",
    ]})
    monkeypatch.setitem(sys.modules, "seqme", SimpleNamespace(metrics=metrics, models=models))
    built = build_metric_list(["KALGLALA"], object(), strict=True)
    assert len(built) == 13
    assert len({metric.name for _, metric in built}) == 13

    def unavailable():
        raise ImportError("missing descriptor dependency")

    models.Hydrophobicity = unavailable
    with pytest.raises(RuntimeError, match="property predictors 2/3"):
        build_metric_list(["KALGLALA"], object(), strict=True)
