import json

import numpy as np
import pytest
import run_label_ablation as ab
from experiment_utils import mark_files, sha256, write_json, write_summary


def records():
    result = []
    for split in (*ab.DEV, "test"):
        for i in (0, 1):
            result.append({"sequence": f"{split}{i}", "family": f"{split}{i}", "split": split,
                           "targets": [i], "mask": [1]})
    return result


def test_arms_have_identical_evaluation_and_distinct_training_labels():
    old = records()
    candidate = [dict(r, targets=[1-r["targets"][0]] if r["split"] == "train" else r["targets"].copy())
                 for r in old if r["split"] in ab.DEV]
    arms, changes = ab.make_arms(old, candidate, ["risky"])
    evaluations = [[r for r in d["records"] if r["split"] != "train"] for d in arms.values()]
    assert all(r == evaluations[0] for r in evaluations)
    assert all(r["split"] != "test" for d in arms.values() for r in d["records"])
    assert sum(r["old"] != r["candidate"] for r in changes) == 2
    assert arms["subset_original"]["records"] != arms["subset_candidate"]["records"]


def test_candidates_reject_unmapped_and_duplicates():
    row = {"sequence": "x", "organism": "", "label": "active"}
    with pytest.raises(ValueError, match="Unmapped"):
        ab.candidate_records([row], ["risky"], {}, "hemolysis")
    with pytest.raises(ValueError, match="Duplicate"):
        ab.candidate_records([row, row], ["risky"], {"x": {"split": "train", "family": "f"}}, "hemolysis")


def test_common_population_intersects_output_masks_not_only_sequences():
    old = [{"sequence": "x", "family": "f", "split": "validation", "targets": [1, 0], "mask": [1, 0]}]
    new = [{"sequence": "x", "family": "f", "split": "validation", "targets": [0, 1], "mask": [1, 1]}]
    arms, transitions = ab.make_arms(old, new, ["a", "b"])
    for data in arms.values():
        assert data["records"][0]["mask"] == [1, 0]
        assert data["records"][0]["targets"] == [0, 1]
    assert len(transitions) == 1 and transitions[0]["output"] == "a"


def fixture(tmp_path, monkeypatch, single_class=False):
    original, candidates, prepared = [tmp_path / s for s in ("original", "candidates", "prepared")]
    original.mkdir()
    candidates.mkdir()
    recs = records()
    write_summary(original / "families.csv", [{k: r[k] for k in ("sequence", "family", "split")} for r in recs])
    mark_files(original, "complete.json", ["families.csv"])
    family_marker = ab.checked_stage(original, "complete.json", {"families.csv"})
    write_json(candidates / "report.json", {"kind": "molar_candidates_v1", "family_marker": family_marker})
    write_summary(candidates / "hemolysis_candidates.csv", [{"sequence": r["sequence"], "organism": "",
                  "label": "inactive" if single_class or r["targets"][0] == 0 else "active"} for r in recs])
    mark_files(candidates, "complete.json", ["report.json", "hemolysis_candidates.csv"])
    recipe = {"backbone": "fake", "revision": "a"*40, "protocol": {"training_seeds": [42, 43, 44],
              "architectures": ["linear", "mlp"], "epochs": 1, "patience": 1, "learning_rate": .001,
              "weight_decay": .01, "head_batch_size": 4}}
    monkeypatch.setattr(ab, "load_prepared", lambda p: (recipe, {"hemolysis": {"outputs": ["risky"], "records": recs}}))
    ab.main(["prepare", "--prepared", str(original), "--candidates", str(candidates), "--tasks", "hemolysis", "--out", str(prepared)])
    return prepared


def test_preparation_support_and_no_test_records(tmp_path, monkeypatch):
    prepared = fixture(tmp_path, monkeypatch)
    assert json.loads((prepared / "preflight.json").read_text())["eligible"]
    data = json.loads((prepared / "dataset.json").read_text())
    assert all(r["split"] in ab.DEV for d in data["hemolysis"].values() for r in d["records"])
    ab.main(["train", "--prepared", str(prepared), "--out", str(tmp_path / "fit"), "--list"])
    assert not (tmp_path / "fit").exists()


def test_failed_support_blocks_training(tmp_path, monkeypatch):
    prepared = fixture(tmp_path, monkeypatch, single_class=True)
    assert not json.loads((prepared / "preflight.json").read_text())["eligible"]
    with pytest.raises(ValueError, match="support"):
        ab.main(["train", "--prepared", str(prepared), "--out", str(tmp_path / "fit"), "--list"])


def test_end_to_end_no_test_embeddings_and_resume(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    prepared = fixture(tmp_path, monkeypatch)
    requested, fitted = [], []

    class Encoder:
        def __init__(self, model, revision, device):
            self.info = {"model": model, "revision": revision}

        def encode(self, sequences, batch_size):
            requested.extend(sequences)
            assert all(not s.startswith("test") for s in sequences)
            return np.zeros((len(sequences), 480), dtype=np.float32)

    import benchmark_reward_generalization as benchmark
    monkeypatch.setattr(benchmark, "FrozenEncoder", Encoder)

    def fit(tx, ty, tm, vx, vy, vm, **kwargs):
        fitted.append(kwargs["seed"])
        return {}, [{"epoch": 1}], np.zeros_like(vy)

    monkeypatch.setattr(ab, "fit_head", fit)
    monkeypatch.setattr(ab, "predict_head", lambda state, x, outputs, arch, device: np.zeros((len(x), outputs)))
    dest = tmp_path / "fit"
    command = ["train", "--prepared", str(prepared), "--out", str(dest), "--device", "cpu"]
    ab.main(command)
    assert len(fitted) == 12
    assert not (dest / "embeddings" / "test").exists()
    digest = sha256(dest / "validation_summary.csv")
    ab.main(command)
    assert len(fitted) == 12
    assert digest == sha256(dest / "validation_summary.csv")
