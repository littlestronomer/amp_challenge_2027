import json

import numpy as np
import pytest
from cache_selectivity_risk import _build_union, _load_chunks, _write_chunk


def _cell(seed, sequences):
    return {"case": "hybrid", "seed": seed, "sequences": sequences,
            "record": {"library_sha256": f"seed-{seed}"}}


def test_union_deduplicates_in_seed_then_original_order():
    union, maps = _build_union([_cell(43, ["BBB", "AAA"]),
                                _cell(42, ["AAA", "CCC"]),
                                _cell(44, ["DDD", "BBB"])])
    assert union == ["AAA", "CCC", "BBB", "DDD"]
    np.testing.assert_array_equal(maps[42]["indices"], [0, 1])
    np.testing.assert_array_equal(maps[43]["indices"], [2, 0])
    np.testing.assert_array_equal(maps[44]["indices"], [3, 2])


def test_risk_chunk_resume_validates_hashes_and_keeps_missing_unset(tmp_path):
    identity = {"union_unique_sequences": 5}
    values, done = _load_chunks(tmp_path, identity)
    assert np.isnan(values).all() and not done.any()
    _write_chunk(tmp_path, 0, 2, np.array([0.1, 0.2], dtype=np.float32))
    values, done = _load_chunks(tmp_path, identity)
    np.testing.assert_array_equal(done, [True, True, False, False, False])
    np.testing.assert_allclose(values[:2], [0.1, 0.2])
    assert np.isnan(values[2:]).all()

    array_path = tmp_path / "000000000-000000002.npy"
    array_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Corrupt marked risk chunk"):
        _load_chunks(tmp_path, identity)


def test_chunk_markers_reject_overlapping_ranges(tmp_path):
    _write_chunk(tmp_path, 0, 2, np.array([0.1, 0.2], dtype=np.float32))
    _write_chunk(tmp_path, 1, 3, np.array([0.2, 0.3], dtype=np.float32))
    with pytest.raises(ValueError, match="Overlapping"):
        _load_chunks(tmp_path, {"union_unique_sequences": 4})


def test_chunk_marker_cannot_escape_cache_directory(tmp_path):
    marker = tmp_path / "chunks/000000000-000000001.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"start": 0, "end": 1, "array": "../../outside.npy", "sha256": "0" * 64}))
    with pytest.raises(ValueError):
        _load_chunks(tmp_path, {"union_unique_sequences": 1})
