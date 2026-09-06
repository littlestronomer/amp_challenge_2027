"""Exact blend quotas, immutable source reuse and one conditioned draw per seed."""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from collections import Counter

import pytest
import sweep_blend_ratios as sweep
from experiment_utils import code_identity, mark_files, sha256, verify_files, write_json

from amp_challenge_2027.data import iter_fasta, write_fasta
from amp_challenge_2027.library_blending import fixed_ratio_blend, parse_ratio, source_quotas
from amp_challenge_2027.pipeline import interleave_blend


@pytest.mark.parametrize("ratio,counts", [((7, 1), (43750, 6250)), ((3, 1), (37500, 12500)), ((1, 1), (25000, 25000))])
def test_exact_50k_source_quotas(ratio, counts):
    assert source_quotas(50000, ratio) == counts


@pytest.mark.parametrize("value", ["0:1", "1:0", "-1:1", "0.875", "1:2:3", "x:1"])
def test_invalid_ratio_rejected(value):
    with pytest.raises(ValueError):
        parse_ratio(value)


def test_ratio_normalization_and_nonintegral_quota_rejected():
    assert parse_ratio("75:25") == (3, 1)
    with pytest.raises(ValueError, match="exact integer counts"):
        source_quotas(50000, (2, 1))


def test_changed_control_evaluator_is_rejected(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "scripts/eval_official.py\n")
    with pytest.raises(ValueError, match="Control evaluation code differs"):
        sweep.check_control_protocol("a" * 40)


@pytest.mark.parametrize("ratio", [(7, 1), (3, 1), (1, 1)])
def test_disjoint_blend_matches_block_interleave(ratio):
    primary = [f"P{i}" for i in range(32)]
    secondary = [f"S{i}" for i in range(32)]
    result = fixed_ratio_blend(primary, secondary, size=16, ratio=ratio)
    assert result.sequences == interleave_blend(primary, secondary, per_primary=ratio[0], per_secondary=ratio[1], n=16)
    assert Counter(result.sources) == dict(zip(("primary", "secondary"), source_quotas(16, ratio)))


def test_overlap_does_not_drift_ratio_or_starve_secondary():
    result = fixed_ratio_blend(["SHARED", "P"], ["SHARED"], size=2, ratio=(1, 1))
    assert result.sequences == ["P", "SHARED"]
    assert result.sources == ["primary", "secondary"]
    assert result.selected_shared_count == result.shared_pool_count == 1
    # Identical component pools can still fill disjoint assignment quotas.
    result = fixed_ratio_blend(list("ABCD"), list("ABCD"), size=4, ratio=(3, 1))
    assert len(set(result.sequences)) == 4
    assert Counter(result.sources) == {"primary": 3, "secondary": 1}


def test_feasible_small_pools_always_fill_exact_quotas():
    universe = list("ABCDE")
    subsets = [list(items) for n in range(1, 6) for items in itertools.combinations(universe, n)]
    for primary, secondary in itertools.product(subsets, repeat=2):
        for ratio in ((1, 1), (3, 1)):
            n_primary, n_secondary = source_quotas(4, ratio)
            feasible = len(primary) >= n_primary and len(secondary) >= n_secondary and len(set(primary + secondary)) >= 4
            if feasible:
                blend = fixed_ratio_blend(primary, secondary, size=4, ratio=ratio)
                assert len(blend.sequences) == len(set(blend.sequences)) == 4
                assert Counter(blend.sources) == {"primary": n_primary, "secondary": n_secondary}
                assert all(seq in (primary if origin == "primary" else secondary)
                           for seq, origin in zip(blend.sequences, blend.sources))
            else:
                with pytest.raises(ValueError, match="Insufficient"):
                    fixed_ratio_blend(primary, secondary, size=4, ratio=ratio)


def test_duplicates_in_one_pool_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        fixed_ratio_blend(["A", "A"], ["B", "C"], size=2, ratio=(1, 1))


def source_fixture(tmp_path, monkeypatch, seeds=(42,)):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    primary = [f"KL{aa[i % 20]}{aa[i // 20]}ALGLAA" for i in range(40)]
    secondary = primary[:4] + [f"RR{aa[i % 20]}{aa[i // 20]}LAGLAA" for i in range(36)]
    reference = tmp_path / "reference.fasta"
    write_fasta(["RRRRRRRR"], reference)
    checkpoint = tmp_path / "conditioned"
    write_json(checkpoint / "config.json", {"conditioning": "charge"})
    (checkpoint / "model.pt").write_bytes(b"test checkpoint, sampled only by fixture")
    source = tmp_path / "source"
    manifest = {
        "kind": "checkpoint_sweep_v1", "code": code_identity(),
        "cases": [{"name": "hybrid", "secondary": str(checkpoint)}, {"name": "ckpt_epoch58"}],
        "seeds": list(seeds), "reference_sha256": sha256(reference), "library_size": 16,
        "raw_count": 40, "temperature": 1.0, "top_p": 0.9,
        "repetition_penalty": 1.3, "device": "cpu", "esm_model": "test-embedder",
    }
    write_json(source / "run.json", manifest)
    for seed in seeds:
        for case in ("hybrid", "ckpt_epoch58"):
            pool = primary if case == "ckpt_epoch58" else interleave_blend(
                primary, secondary, per_primary=3, per_secondary=1, n=60,
            )
            cell = source / case / f"seed{seed}"
            write_fasta(pool, cell / "pool.fasta")
            write_fasta(pool[:16], cell / "library.fasta")
            write_json(cell / "generation_stats.json", {"pool_size": len(pool), "library_size": 16})
            mark_files(cell, "generation.json", ["pool.fasta", "library.fasta", "generation_stats.json"])
            write_json(cell / "metrics.json", {**dict.fromkeys(sweep.CONTROL_METRICS, 0.5), "FBD.deviation": None})
            (cell / "metrics.csv").write_text("FBD\n0.5\n")
            mark_files(cell, "evaluation.json", ["library.fasta", "metrics.json", "metrics.csv"])
    calls = []

    def sample(n, **kwargs):
        assert n == 40
        assert kwargs["checkpoint_dir"] == checkpoint
        assert kwargs["charge_conditioned"] is True
        assert kwargs["reference_set"] == {"RRRRRRRR"}
        assert kwargs["top_k"] == 50 and kwargs["top_p"] == 0.9
        assert kwargs["temperature"] == 1.0 and kwargs["repetition_penalty"] == 1.3
        calls.append(kwargs["seed"])
        return list(secondary)

    monkeypatch.setattr(sweep, "generate_with_model", sample)
    monkeypatch.setattr("amp_challenge_2027.training.enable_determinism", lambda _seed: None)
    out = tmp_path / "blends"
    args = ["--source-sweep", str(source), "--secondary-checkpoint", str(checkpoint),
            "--reference", str(reference), "--out", str(out), "--seeds", *map(str, seeds)]
    return args, source, checkpoint, out, calls


def test_cli_reuses_primary_and_samples_conditioned_once_per_seed(tmp_path, monkeypatch):
    args, source, checkpoint, out, calls = source_fixture(tmp_path, monkeypatch, seeds=(42, 43))
    originals = {path: sha256(path) for root in (source, checkpoint) for path in root.rglob("*") if path.is_file()}
    sweep.main([*args, "--generate-only"])
    assert calls == [42, 43]
    for seed in (42, 43):
        for ratio in ((7, 1), (3, 1), (1, 1)):
            directory = out / f"p{ratio[0]}_s{ratio[1]}" / f"seed{seed}"
            sequences = [seq for _, seq in iter_fasta(directory / "library.fasta")]
            assert len(sequences) == len(set(sequences)) == 16
            with (directory / "membership.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert [row["sequence"] for row in rows] == sequences
            assert Counter(row["source"] for row in rows) == dict(zip(("primary", "secondary"), source_quotas(16, ratio)))
    sweep.main([*args, "--generate-only"])
    assert calls == [42, 43]
    with (out / "controls.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["evaluation_origin"] == "recorded_source_control" for row in rows)
    assert all(sha256(path) == digest for path, digest in originals.items())
    assert not list(out.rglob("top.fasta"))


def test_list_is_read_only_and_modified_inputs_or_outputs_fail(tmp_path, monkeypatch):
    args, source, _, out, calls = source_fixture(tmp_path, monkeypatch)
    sweep.main([*args, "--list"])
    assert not out.exists() and not calls
    sweep.main([*args, "--generate-only"])
    with pytest.raises(ValueError, match="Experiment inputs changed"):
        sweep.main([*args, "--ratios", "3:1", "--generate-only"])
    (out / "p7_s1/seed42/library.fasta").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        sweep.main([*args, "--generate-only"])
    (source / "ckpt_epoch58/seed42/pool.fasta").write_text("changed source")
    with pytest.raises(ValueError, match="artifact changed"):
        sweep.main([*args, "--list"])


@pytest.mark.parametrize("change", ["reference", "runtime", "conditioning", "seed", "hybrid_primary"])
def test_mismatched_source_or_checkpoint_is_rejected_before_sampling(tmp_path, monkeypatch, change):
    args, source, checkpoint, out, calls = source_fixture(tmp_path, monkeypatch)
    run = json.loads((source / "run.json").read_text())
    if change == "reference":
        run["reference_sha256"] = "wrong"
    elif change == "runtime":
        run["code"]["python"] = "different"
    elif change == "conditioning":
        write_json(checkpoint / "config.json", {"conditioning": "none"})
    elif change == "seed":
        args += ["--seeds", "44"]
    else:
        args += ["--primary-case", "hybrid"]
    write_json(source / "run.json", run)
    with pytest.raises(ValueError):
        sweep.main([*args, "--generate-only"])
    assert not out.exists() and not calls


def test_shared_conditioned_cache_reused_across_ratio_runs(tmp_path, monkeypatch):
    args, _, _, out, calls = source_fixture(tmp_path, monkeypatch)
    shared = tmp_path / "conditioned-cache"
    sweep.main([*args, "--secondary-cache", str(shared), "--generate-only"])
    new_args = [str(tmp_path / "second") if item == str(out) else item for item in args]
    sweep.main([*new_args, "--secondary-cache", str(shared), "--ratios", "75:25", "--generate-only"])
    assert calls == [42]
    (shared / "seed42/pool.fasta").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        sweep.main([*args, "--secondary-cache", str(shared), "--generate-only"])


def test_evaluation_only_rerun_keeps_libraries_and_does_not_evaluate_controls(tmp_path, monkeypatch):
    args, _, _, out, calls = source_fixture(tmp_path, monkeypatch)
    sweep.main([*args, "--generate-only"])
    libraries = {path: sha256(path) for path in out.rglob("library.fasta")}
    evaluated = []

    def evaluate(directory, **kwargs):
        assert kwargs["esm_model"] == "test-embedder" and kwargs["device"] == "cpu"
        if not verify_files(directory, "evaluation.json"):
            evaluated.append(directory)
            write_json(directory / "metrics.json", {"FBD": 0.2})
            mark_files(directory, "evaluation.json", ["library.fasta", "metrics.json"])
        return {"FBD": 0.2}

    monkeypatch.setattr(sweep, "evaluate_library", evaluate)
    sweep.main(args)
    sweep.main(args)
    assert len(evaluated) == 3 and calls == [42]
    assert all(sha256(path) == digest for path, digest in libraries.items())


def test_real_conditioned_sampler_cpu_smoke(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from amp_challenge_2027.generate import generate_with_model
    from amp_challenge_2027.generator import build_model, save_model
    from amp_challenge_2027.model import DecoderConfig
    from amp_challenge_2027.training import enable_determinism

    args, _, checkpoint, out, _ = source_fixture(tmp_path, monkeypatch)
    torch.manual_seed(123)
    model, cfg = build_model(DecoderConfig(hidden_size=16, num_layers=1, num_heads=4, conditioning="charge"))
    with torch.no_grad():
        model.tok_emb.weight.zero_()
    save_model(model, checkpoint, config=cfg)
    monkeypatch.setattr(sweep, "generate_with_model", generate_with_model)
    monkeypatch.setattr("amp_challenge_2027.training.enable_determinism", enable_determinism)
    sweep.main([*args, "--generate-only"])
    stats = json.loads((out / "secondary/seed42/generation_stats.json").read_text())
    assert stats["raw_count"] == 40 and stats["pool_size"] >= 16
    assert len(list(iter_fasta(out / "p3_s1/seed42/library.fasta"))) == 16
    sweep.main([*args, "--generate-only"])
