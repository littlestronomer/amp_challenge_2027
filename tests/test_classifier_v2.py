"""Tests for classifier-v2 helpers: MIC binarization + temperature scaling.

Numpy-only; runs in the minimal dev environment.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# build_activity_labels.binarize — min-across-panel-genera policy
# ---------------------------------------------------------------------------


def _row(seq, organism, mic):
    return {"sequence": seq, "target_organism": organism, "mic_value_um": str(mic)}


SEQ = "KLLAKLLAKLLA"


class TestBinarize:
    def test_active_on_any_potent_genus(self):
        from build_activity_labels import binarize

        rows = [
            _row(SEQ, "Escherichia coli", 2.0),  # potent
            _row(SEQ, "Pseudomonas aeruginosa", 64),  # weak
        ]
        out = binarize(rows)
        assert len(out) == 1 and out[0]["label"] == "active"
        assert out[0]["organism"] == "E. coli"  # most potent genus reported

    def test_inactive_requires_all_weak(self):
        from build_activity_labels import binarize

        rows = [
            _row(SEQ, "Escherichia coli", 64),
            _row(SEQ, "Pseudomonas aeruginosa", 128),
        ]
        out = binarize(rows)
        assert out[0]["label"] == "inactive"

    def test_ambiguous_band_dropped(self):
        from build_activity_labels import binarize

        rows = [_row(SEQ, "Escherichia coli", 8.0)]  # between 4 and 32
        assert binarize(rows) == []

    def test_nonpanel_and_invalid_and_nonpositive_skipped(self):
        from build_activity_labels import binarize

        rows = [
            _row("BAD!", "Escherichia coli", 2.0),
            _row("KLLAKLLAKL", "Clostridium difficile", 1.0),
            _row(SEQ, "Escherichia coli", -3.0),
        ]
        assert binarize(rows) == []

    def test_min_beats_mean_across_genera(self):
        """Potent-on-one-genus must survive averaging dilution (min rule)."""
        from build_activity_labels import binarize

        rows = [
            _row(SEQ, "Escherichia coli", 1.0),
            _row(SEQ, "Pseudomonas aeruginosa", 100.0),
            _row(SEQ, "Klebsiella pneumoniae", 100.0),
        ]
        out = binarize(rows)
        assert out[0]["label"] == "active"


# ---------------------------------------------------------------------------
# train_reward_classifier.fit_temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_recovers_overconfidence():
    from train_reward_classifier import fit_temperature

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=4000).astype(float)
    true_logits = rng.normal(0, 1, size=4000)
    # Overconfident model: logits scaled up by 5 → optimal T ≈ 5.
    overconfident = true_logits * 5
    t = fit_temperature(overconfident, labels)
    assert 3.5 < t < 7.0

    def nll(logits, temp):
        p = np.clip(1 / (1 + np.exp(-logits / temp)), 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    assert nll(overconfident, t) < nll(overconfident, 1.0)


def test_fit_temperature_identity_when_well_calibrated():
    """Labels sampled from sigmoid(logits) ⇒ the logits are already optimal (T≈1)."""
    from train_reward_classifier import fit_temperature

    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, size=4000)
    labels = rng.binomial(1, 1 / (1 + np.exp(-logits))).astype(float)
    t = fit_temperature(logits, labels)
    assert 0.6 < t < 1.8  # near 1 when nothing to fix
