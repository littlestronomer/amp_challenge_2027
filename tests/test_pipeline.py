"""End-to-end tests for the pool-loading + blending pipeline (numpy-only path).

These run the actual ``blend_libraries`` CLI entry point on tiny synthetic
pools, asserting correctness (filters, membership) and byte-level
reproducibility — the property the competition validator checks.
"""

from __future__ import annotations

from pathlib import Path

import blend_libraries  # inserted into sys.path by tests/conftest.py

from amp_challenge_2027.data import iter_fasta
from amp_challenge_2027.pipeline import load_pool_fastas


def _write_fasta(path: Path, seqs: list[str]) -> None:
    with open(path, "w") as f:
        for i, s in enumerate(seqs, start=1):
            f.write(f">s{i}\n{s}\n")


SEQS_A = ["KLLAKLLAKL", "GHIKLMNPQRST", "ACDEFGHIK", "FGHIKLMNPQ", "KLLAKLLAGV"]
SEQS_B = ["KLLAKLLAKL", "WFEAKLLAKL", "GHIKLMNPQRSA", "ACDEFGHIKM", "VRLAKLLA"]


def test_load_pool_fastas_cap_and_determinism(tmp_path: Path):
    fa = tmp_path / "pool.fasta"
    _write_fasta(fa, SEQS_A)
    full = load_pool_fastas([fa])
    capped1 = load_pool_fastas([fa], cap_per_source=2, seed=7)
    capped2 = load_pool_fastas([fa], cap_per_source=2, seed=7)
    assert len(full) == len(SEQS_A)
    assert capped1 == capped2  # same seed → same seeded shuffle → same head
    assert len(capped1) == 2
    assert set(capped1) <= set(SEQS_A)


def test_blend_end_to_end_reproducible(tmp_path: Path):
    pa = tmp_path / "a.fasta"
    pb = tmp_path / "b.fasta"
    _write_fasta(pa, SEQS_A)
    _write_fasta(pb, SEQS_B)

    argv_common = [
        "--pools",
        str(pa),
        str(pb),
        "--library-size",
        "6",
        "--top-k",
        "3",
        "--w-activity",
        "0",  # no classifier artifact in the test env
        "--w-conformity",
        "1.0",
        "--w-precision",
        "0",
    ]

    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    blend_libraries.main([*argv_common, "--out-dir", str(out1)])
    blend_libraries.main([*argv_common, "--seed", "43", "--out-dir", str(out2)])

    lib1 = [s for _, s in iter_fasta(out1 / "library.fasta")]
    top1 = [s for _, s in iter_fasta(out1 / "top.fasta")]

    union_clean = set(SEQS_A) | set(SEQS_B)
    assert len(lib1) == 6
    assert set(lib1) <= union_clean
    assert len(set(lib1)) == len(lib1), "library must be deduplicated"
    assert len(top1) == 3
    assert set(top1) <= set(lib1)

    # Same seed → byte-identical outputs.
    out3 = tmp_path / "out3"
    blend_libraries.main([*argv_common, "--out-dir", str(out3)])
    assert (out3 / "library.fasta").read_bytes() == (out1 / "library.fasta").read_bytes()
    assert (out3 / "top.fasta").read_bytes() == (out1 / "top.fasta").read_bytes()

    # A different seed may reorder the library; content must still be valid.
    lib2 = [s for _, s in iter_fasta(out2 / "library.fasta")]
    assert set(lib2) <= union_clean and len(lib2) == 6
