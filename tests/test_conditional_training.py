"""Measured-condition training checks; tiny real models run offline on CPU."""
# ruff: noqa: E402

import json

import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.assay_conditioning import collate_assay_conditions, fit_condition_schema
from amp_challenge_2027.conditional_training import (
    _read_dataset,
    _sample_epoch,
    _tokens,
    causal_loss,
    configure_optimizer,
    evaluate_model,
    qualifies_for_selective,
    train_experiment,
)
from amp_challenge_2027.model import DecoderConfig, build_model, is_assay_parameter, save_model

TINY = dict(hidden_size=16, num_heads=2, num_layers=2, ffn_inner=32,
            moe_expert_inner=16, dropout=0.0)


def condition(endpoint="mic", upper=8.0, **kwargs):
    return dict(endpoint=endpoint, value_lower=upper, value_upper=upper, unit="uM",
                operator="=", publication_id="paper1", **kwargs)


def joint_conditions():
    lysis = condition("lysis_percent", 5.0, rbc_species="human", dose_lower=64,
                      dose_upper=64, dose_unit="uM")
    lysis["unit"] = "%"
    return [condition(target="E. coli"), lysis]


def make_dataset(tmp_path):
    rows = []
    for index, sequence in enumerate(("ACDEFGHI", "KLMNPQRS", "TVWYACDE", "FGHIKLMN")):
        rows.append(dict(sequence=sequence, split="train", family=f"train{index}",
                         conditions=joint_conditions(), supervision_eligible=True))
    rows.append(dict(sequence="PQRSTVWYAC", split="validation", family="val1",
                     conditions=[condition(target="VALIDATION_ONLY")], supervision_eligible=True))
    rows.append(dict(sequence="WACDEFGHI", split="test", family="test1",
                     conditions=[condition(target="TEST_ONLY")], supervision_eligible=True))
    rows.append(dict(sequence="LMNPQRSTVW", split="calibration", family="calibration1",
                     conditions=[condition(target="CALIBRATION_ONLY")], supervision_eligible=True))
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dict(schema_version=1, examples=rows)))
    return path


def test_selective_requires_measured_compatible_joint_evidence():
    assert qualifies_for_selective(joint_conditions())
    assert not qualifies_for_selective([joint_conditions()[0]])
    for change in ({"value_upper": None, "operator": ">"}, {"rbc_species": "rabbit"},
                   {"dose_upper": 128}, {"publication_id": "another_paper"},
                   {"endpoint": "hc50"}, {"unit": "uM"}):
        conditions = joint_conditions()
        conditions[1].update(change)
        assert not qualifies_for_selective(conditions)
    conditions = joint_conditions()
    conditions[1].update(value_lower=None, value_upper=10, operator="<=")
    assert qualifies_for_selective(conditions)
    conditions.append(dict(conditions[1], value_lower=80, value_upper=80, operator="exact"))
    assert not qualifies_for_selective(conditions)


def test_family_partition_overlap_rejected(tmp_path):
    path = make_dataset(tmp_path)
    data = json.loads(path.read_text())
    data["examples"][-1]["family"] = "train1"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leakage"):
        _read_dataset(path)


def test_sampling_chooses_unique_sequence_before_observation():
    import random

    row = {"conditions": [condition()], "supervision_eligible": True}
    groups = {"ACDEFGHI": [row] * 1000, "KLMNPQRS": [row]}
    draws = _sample_epoch(groups, sorted(groups), 4000, random.Random(42))
    first = sum(sequence == "ACDEFGHI" for sequence, _ in draws)
    assert 1800 < first < 2200
    assert sum(not conditions for _, conditions in draws) == 1000


def test_validation_token_weighting_independent_of_batch_size():
    torch.manual_seed(42)
    model, _ = build_model(DecoderConfig(**TINY))
    rows = {sequence: [dict(conditions=[], supervision_eligible=False)]
            for sequence in ("ACDEFGHI", "KLMNPQRSTVWYACDEFGHI")}
    one = evaluate_model(model, rows, None, batch_size=1)
    two = evaluate_model(model, rows, None, batch_size=2)
    assert one["token_nll"] == pytest.approx(two["token_nll"], rel=1e-6)
    assert one["valid_tokens"] == two["valid_tokens"] == sum(len(seq) + 1 for seq in rows)


def test_condition_warmup_changes_new_gate_only_not_backbone():
    schema = fit_condition_schema([dict(split="train", conditions=joint_conditions())])
    model, _ = build_model(DecoderConfig(**TINY, assay_schema=schema))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = configure_optimizer(model, freeze_backbone=True, new_lr=1e-3, base_lr=1e-5)
    assert all(is_assay_parameter(name) == p.requires_grad for name, p in model.named_parameters())
    ids = _tokens(["ACDEFGHI", "KLMNPQRS"], "cpu")
    conditions = collate_assay_conditions([joint_conditions()] * 2, schema)
    output = model(ids, assay_conditions=conditions)
    sums, counts = causal_loss(output.logits, ids)
    (sums.sum() / counts.sum()).backward()
    optimizer.step()
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters()
               if not is_assay_parameter(name))
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters()
               if is_assay_parameter(name))
    configure_optimizer(model, freeze_backbone=False, new_lr=1e-4, base_lr=1e-5)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("preset", ["A", "B", "C", "D"])
def test_tiny_fresh_training_writes_checkpoint_and_train_only_schema(tmp_path, preset):
    path = make_dataset(tmp_path)
    out = tmp_path / "result"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train_experiment(path, out, "conditional", preset=preset, seed=12,
                                  device="cpu", epochs=1, batch_size=4,
                                  examples_per_epoch=8, model_overrides=TINY)
    finally:
        torch.set_num_threads(old_threads)
    assert result["status"] == "complete"
    assert result["epochs_completed"] == 1
    config = json.loads((out / "checkpoint/config.json").read_text())
    targets = config["assay_schema"]["vocabularies"]["target"]
    assert "E. coli" in targets
    assert "VALIDATION_ONLY" not in targets
    assert "TEST_ONLY" not in targets
    assert "CALIBRATION_ONLY" not in targets
    history = json.loads((out / "history.json").read_text())
    assert history[0]["train_examples"] == 8
    assert history[0]["valid_train_tokens"] == 72
    assert history[0]["tokens_per_second"] > 0
    marker = json.loads((out / "complete.json").read_text())
    assert marker["status"] == "complete"
    assert "checkpoint/model.pt" in marker["files"]
    assert history[0]["routing_last_batch"] if preset in {"C", "D"} else True
    with pytest.raises(ValueError, match="new output directory"):
        train_experiment(path, out, "conditional", device="cpu", epochs=1)


def test_warm_training_and_insufficient_selective_labels(tmp_path):
    path = make_dataset(tmp_path)
    model, cfg = build_model(DecoderConfig(**TINY))
    checkpoint = tmp_path / "base"
    save_model(model, checkpoint, config=cfg)
    before = (checkpoint / "model.pt").read_bytes()
    result = train_experiment(path, tmp_path / "insufficient", "selective", checkpoint=checkpoint,
                              device="cpu", epochs=1, min_selective=32)
    assert result["status"] == "insufficient_joint_labels"
    assert not (tmp_path / "insufficient/checkpoint").exists()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train_experiment(path, tmp_path / "warm", "conditional", checkpoint=checkpoint,
                                  device="cpu", epochs=1, batch_size=4, examples_per_epoch=8)
    finally:
        torch.set_num_threads(old_threads)
    history = json.loads((tmp_path / "warm/history.json").read_text())
    assert result["epochs_completed"] == 2
    assert [row["stage"] for row in history] == ["conditions_only", "joint"]
    assert history[0]["backbone_frozen"] and not history[1]["backbone_frozen"]
    assert (checkpoint / "model.pt").read_bytes() == before
