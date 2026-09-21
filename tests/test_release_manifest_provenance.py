from types import SimpleNamespace

from amp_challenge_2027.release_manifest import _config_provenance, _revision_record


def test_non_neural_scorer_has_no_applicable_backbone():
    record = _revision_record(None, applicable=False)
    assert record == {"applicable": False, "resolved_immutable_revision": None,
                      "resolved": False, "explicitly_pinned_at_load": False}


def test_runtime_resolved_revision_is_not_mislabeled_as_explicitly_pinned():
    scorer = SimpleNamespace(_model=SimpleNamespace(esm=SimpleNamespace(
        config=SimpleNamespace(_commit_hash="immutable-revision"))))
    record = _revision_record(scorer, applicable=True, explicitly_pinned=False)
    assert record["resolved_immutable_revision"] == "immutable-revision"
    assert record["resolved"] is True
    assert record["explicitly_pinned_at_load"] is False


def test_config_provenance_tracks_the_actual_explicit_file(tmp_path):
    project = tmp_path / "project"
    root = project / "checkpoint" / "reward"
    root.mkdir(parents=True)
    head = root / "classifier.pt"
    config = root / "classifier_config.json"
    head.write_bytes(b"head bytes")
    config.write_text('{"temperature":1.0}\n')
    result = _config_provenance(root, "classifier", head, {"temperature": 1.0})
    assert result["origin"] == "explicit_file"
    assert result["source_path"] == "checkpoint/reward/classifier_config.json"
    original_hash = result["source_sha256"]
    config.write_text('{"temperature":2.0}\n')
    changed = _config_provenance(root, "classifier", head, {"temperature": 2.0})
    assert changed["source_sha256"] != original_hash
