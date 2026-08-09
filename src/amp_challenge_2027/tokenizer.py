"""Character-level amino-acid tokenizer.

Uses the same 33-style layout as ESM-2 (special tokens first, then the 20
standard amino acids) so the generator's vocabulary stays compatible with the
ESM-2 reward surrogates downstream. One residue == one token.
"""

from __future__ import annotations

from collections.abc import Iterable

from amp_challenge_2027.config import (
    AMINO_ACIDS,
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    MASK_ID,
    PAD_ID,
    TOKEN_TO_ID,
    VOCAB,
)

# Residues map to their id in VOCAB (offset by the 4 special tokens).
RESIDUE_TO_ID = {aa: TOKEN_TO_ID[aa] for aa in AMINO_ACIDS}
ID_TO_RESIDUE = {i: aa for aa, i in RESIDUE_TO_ID.items()}

INVALID_AA = set("BXZJOU*-")  # common non-standard characters to reject outright


def encode(
    sequence: str,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[int]:
    """Encode a peptide sequence to a list of token ids.

    Raises ``KeyError`` on any non-standard residue so callers fail loudly
    rather than silently emitting ``<unk>``.
    """
    for ch in sequence:
        if ch not in RESIDUE_TO_ID:
            raise ValueError(f"Non-standard residue {ch!r} in sequence {sequence!r}")
    ids = [RESIDUE_TO_ID[ch] for ch in sequence]
    if add_bos:
        ids = [BOS_ID, *ids]
    if add_eos:
        ids = [*ids, EOS_ID]
    return ids


def decode(ids: Iterable[int], *, strip_special: bool = True) -> str:
    """Decode token ids back to a residue string, dropping specials by default."""
    chars = []
    for i in ids:
        if strip_special and i in (PAD_ID, BOS_ID, EOS_ID, MASK_ID):
            continue
        chars.append(ID_TO_RESIDUE[int(i)])
    return "".join(chars)


def is_valid_sequence(seq: str) -> bool:
    """True if ``seq`` only contains the 20 standard amino acids."""
    return len(seq) > 0 and all(ch in RESIDUE_TO_ID for ch in seq)


def vocab_size() -> int:
    return len(VOCAB)


__all__ = [
    "encode",
    "decode",
    "is_valid_sequence",
    "vocab_size",
    "RESIDUE_TO_ID",
    "ID_TO_RESIDUE",
    "INVALID_AA",
    "BOS_TOKEN",
    "EOS_TOKEN",
]
