"""Regression tests for the trainer fixes (Track A) — torch-free part."""

from __future__ import annotations

import numpy as np
import pytest

from amp_challenge_2027.config import BACTERIAL_PANEL, NUM_STRAINS

# ---------------------------------------------------------------------------
# Reward targets (train_reward.build_mic_targets — pure numpy)
# ---------------------------------------------------------------------------


def _mic_rows(mic_by_genus: dict[str, float], seq: str = "KLLAKLLAKL") -> list[dict]:
    return [
        {"sequence": seq, "target_organism": g, "mic_value_um": str(v)}
        for g, v in mic_by_genus.items()
    ]


def _genus_idxs(genus: str) -> list[int]:
    return [i for i, (g, _s, _m) in enumerate(BACTERIAL_PANEL) if g == genus]


def test_nonpositive_mic_dropped_not_crashed():
    from train_reward import build_mic_targets

    rows = _mic_rows({"E. coli": 0.0})
    rows.append({"sequence": "KLLAKLLAKL", "target_organism": "E. coli", "mic_value_um": "8"})
    out = dict(build_mic_targets(rows))
    vec = out["KLLAKLLAKL"]
    # The valid measurement survives on every E. coli head; others stay NaN.
    ecoli = _genus_idxs("E. coli")
    assert np.isfinite(vec[ecoli]).all()
    other = [i for i in range(NUM_STRAINS) if i not in ecoli]
    assert np.isnan(vec[other]).all()


def test_all_heads_receive_data_no_dead_strain_heads():
    from train_reward import build_mic_targets

    genera = {g for g, _s, _m in BACTERIAL_PANEL}
    out = build_mic_targets(_mic_rows({g: 4.0 for g in genera}))
    covered = np.zeros(NUM_STRAINS, dtype=bool)
    for _seq, v in out:
        covered |= ~np.isnan(v)
    assert covered.all(), "every panel head must be supervised by its genus data"


def test_duplicate_measurements_averaged_correctly():
    from train_reward import build_mic_targets

    rows = _mic_rows({"E. coli": 2.0})
    rows.append({"sequence": "KLLAKLLAKL", "target_organism": "E. coli", "mic_value_um": "8"})
    out = dict(build_mic_targets(rows))
    expected = np.log10(4.0)  # mean(log10 2, log10 8) == log10 4
    for i in _genus_idxs("E. coli"):
        assert out["KLLAKLLAKL"][i] == pytest.approx(expected, abs=1e-4)


def test_ceiling_clamped_and_unknown_genus_skipped():
    from train_reward import build_mic_targets

    from amp_challenge_2027.config import MIC_CEILING_UM

    rows = _mic_rows({"E. coli": 9999.0})
    rows.append({"sequence": "KLLAKLLAKL", "target_organism": "Clostridium", "mic_value_um": "4"})
    out = dict(build_mic_targets(rows))
    for i in _genus_idxs("E. coli"):
        assert out["KLLAKLLAKL"][i] == pytest.approx(np.log10(MIC_CEILING_UM), abs=1e-6)
