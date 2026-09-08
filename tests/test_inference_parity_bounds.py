import json
import shutil

import pytest
from audit_measurement_bounds import audit_file

from amp_challenge_2027.config import REWARD_DIR
from amp_challenge_2027.inference_metadata import load_config
from amp_challenge_2027.measurement_bounds import activity_label, parse_bound


def test_fresh_clone_metadata_and_explicit_mismatch(tmp_path):
    shutil.copy(REWARD_DIR / "classifier.pt", tmp_path / "classifier.pt")
    (tmp_path / "config.json").write_text('{"temperature": 4.6}')
    assert load_config(tmp_path, "classifier")["temperature"] == .98
    explicit = tmp_path / "classifier_config.json"
    explicit.write_text(json.dumps(load_config(tmp_path, "classifier")))
    assert load_config(tmp_path, "classifier")["temperature"] == .98
    explicit.write_text('{"temperature": 4.6}')
    with pytest.raises(ValueError, match="mismatch"):
        load_config(tmp_path, "classifier")


@pytest.mark.parametrize("raw,label", [("16", "active"), (">16", "ambiguous"),
    ("≤16", "active"), (">32", "inactive"), (">=32", "ambiguous"), ("<64", "ambiguous")])
def test_bounds(raw, label):
    assert activity_label(parse_bound(raw)) == label


@pytest.mark.parametrize("raw", ["16-32", "16 trailing", "0", "nan", "1e999"])
def test_reject_unsupported(raw):
    with pytest.raises(ValueError):
        parse_bound(raw)


def test_hemo_bound_not_exact():
    assert not parse_bound(">128").definitely_le(128)
    assert parse_bound(">128").definitely_ge(128)


def test_read_only_audit(tmp_path):
    path = tmp_path / "raw.csv"
    text = "concentration\n>16 µM\n16 µM\n16-32 µM\n"
    path.write_text(text)
    result = audit_file(path, "concentration")
    assert result["counts"]["mic_threshold_label_changes"] == 1
    assert result["counts"]["unsupported"] == 1
    assert path.read_text() == text
