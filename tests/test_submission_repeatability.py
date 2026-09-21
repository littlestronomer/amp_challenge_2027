from pathlib import Path

import pytest
from verify_submission import _verify_repeatability


def _pair(tmp_path: Path, first: dict[str, bytes], second: dict[str, bytes]):
    first_paths = {}
    second_paths = {}
    for key in ("library", "top", "scores"):
        first_paths[key] = tmp_path / f"first-{key}"
        second_paths[key] = tmp_path / f"second-{key}"
        first_paths[key].write_bytes(first[key])
        second_paths[key].write_bytes(second[key])
    return first_paths, second_paths


def test_ordinary_repeatability_compares_saved_first_run_bytes(tmp_path):
    original = {"library": b">a\nAAAA\n", "top": b">a\nAAAA\n", "scores": b"rank,score\n"}
    first, second = _pair(tmp_path, original, original)
    _verify_repeatability(first["library"], first["top"], second["library"], second["top"],
                          first_run_bytes={"library": original["library"], "top": original["top"]})


@pytest.mark.parametrize("changed", ["library", "top"])
def test_ordinary_repeatability_rejects_changed_second_output(tmp_path, changed):
    original = {"library": b">a\nAAAA\n", "top": b">a\nAAAA\n", "scores": b"rank,score\n"}
    second_data = dict(original)
    second_data[changed] += b"changed\n"
    first, second = _pair(tmp_path, original, second_data)
    with pytest.raises(ValueError, match="differs between runs"):
        _verify_repeatability(first["library"], first["top"], second["library"], second["top"],
                              first_run_bytes={"library": original["library"], "top": original["top"]})


def test_strict_repeatability_checks_score_bytes(tmp_path):
    original = {"library": b"lib", "top": b"top", "scores": b"score1"}
    second_data = {**original, "scores": b"score2"}
    first, second = _pair(tmp_path, original, second_data)
    with pytest.raises(ValueError, match="score rows differ"):
        _verify_repeatability(first["library"], first["top"], second["library"], second["top"],
                              first_scores=first["scores"], second_scores=second["scores"])


def test_repeatability_rejects_missing_second_output(tmp_path):
    original = {"library": b"lib", "top": b"top", "scores": b"score"}
    first, second = _pair(tmp_path, original, original)
    second["top"].unlink()
    with pytest.raises(FileNotFoundError):
        _verify_repeatability(first["library"], first["top"], second["library"], second["top"],
                              first_run_bytes={"library": b"lib", "top": b"top"})
