"""Tests for the tokenizer and property computations (no torch needed)."""

from __future__ import annotations

import pytest

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.props import (
    compute_properties,
    hydrophobic_moment,
    is_plausible,
    net_charge,
)

# --- tokenizer ---


def test_encode_decode_roundtrip():
    seq = "KLLKLLKLLK"
    ids = tok.encode(seq)
    assert ids[0] == tok.BOS_ID
    assert ids[-1] == tok.EOS_ID
    assert tok.decode(ids) == seq


def test_encode_rejects_nonstandard():
    with pytest.raises(ValueError):
        tok.encode("KBXZ")  # B, X, Z are non-standard


def test_is_valid_sequence():
    assert tok.is_valid_sequence("ACDEFGHIK")
    assert not tok.is_valid_sequence("ACD-B")
    assert not tok.is_valid_sequence("")


# --- properties ---


def test_net_charge_polyK():
    # Poly-K: 10 Lys → +10, + termini cancel
    assert net_charge("KKKKKKKKKK") == pytest.approx(10.0)


def test_net_charge_neutral_peptide():
    # AGFDVIKKVASVIGGL: 2 Lys (+2), 1 Asp (-1), termini cancel → net +1
    assert net_charge("AGFDVIKKVASVIGGL") == pytest.approx(1.0, abs=0.5)


def test_hydrophobic_moment_homopolymer_is_small():
    # A homopolymer has no amphipathic separation; its helical moment is small
    # (not exactly zero due to the i=0 term in the Eisenberg sum).
    assert hydrophobic_moment("LLLLLLLL") == pytest.approx(0.0, abs=0.1)


def test_compute_properties_keys():
    p = compute_properties("KLLKLLKLLK")
    assert p.length == 10
    assert p.charge > 0
    assert p.cysteine_count == 0
    assert 0.0 <= p.fraction_positive <= 1.0


def test_plausibility_accepts_typical_amp():
    assert is_plausible("KLLKLLKLLK")


def test_plausibility_rejects_extreme_hydrophobic():
    # Long Leu run → aggregation-prone / hemolytic.
    assert not is_plausible("LLLLLLLLLLLLLLLLLLLL")
