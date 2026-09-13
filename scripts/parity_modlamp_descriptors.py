"""Parity check: local ConformityScore-scale descriptors vs seqme/modlamp.

Computes charge, hydrophobicity and hydrophobic moment for a deterministic
sample of the reference set plus fixed anchor peptides using the local
implementations in ``amp_challenge_2027.conditioning``. Where modlamp is
installed (SSH: the locked ``seqme[aa_descriptors]`` environment), computes
the exact seqme predictor call pattern and reports per-axis max absolute
differences. Also reports full-reference marginal statistics and the fraction
of reference sequences clamped into edge bins under the default conditioning
edges — the gate for freezing the bin constants before conditioned SFT.

Local runs (no modlamp) still emit anchors and reference statistics; the
parity block records that modlamp was unavailable. Nothing here trains or
changes artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import REPO_ROOT, code_identity, sha256, write_json  # noqa: E402

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA  # noqa: E402

# Fixed anchors (valid 8-50 aa peptides covering distinct property regimes).
ANCHORS = [
    "GLFDIVKKVVGALGSL",
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "KKKKKKKK",
    "FLPAIWAAAKFL",
    "WLRRIRKIAAHR",
]


def _load_reference(path: Path) -> list[str]:
    from amp_challenge_2027.data import iter_fasta

    return [seq for _, seq in iter_fasta(path)]


def _local_values(sequences: list[str]) -> dict[str, list[float]]:
    from amp_challenge_2027.conditioning import (
        charge_modlamp,
        hydrophobic_moment_modlamp,
        hydrophobicity_modlamp,
    )

    return {
        "charge": [charge_modlamp(seq) for seq in sequences],
        "hydrophobicity": [hydrophobicity_modlamp(seq) for seq in sequences],
        "hydrophobic_moment": [hydrophobic_moment_modlamp(seq) for seq in sequences],
    }


def _modlamp_values(sequences: list[str]) -> dict[str, list[float]]:
    """The exact seqme predictor call pattern, via modlamp."""
    from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

    charge_desc = GlobalDescriptor(list(sequences))
    charge_desc.calculate_charge(ph=7.0)
    hydro_desc = PeptideDescriptor(list(sequences))
    hydro_desc.load_scale("eisenberg")
    hydro_desc.calculate_global()
    moment_desc = PeptideDescriptor(list(sequences))
    moment_desc.load_scale("eisenberg")
    moment_desc.calculate_moment(window=11, angle=100, modality="mean")
    return {
        "charge": [float(v) for v in charge_desc.descriptor.squeeze(axis=-1)],
        "hydrophobicity": [float(v) for v in hydro_desc.descriptor.squeeze(axis=-1)],
        "hydrophobic_moment": [float(v) for v in moment_desc.descriptor.squeeze(axis=-1)],
    }


def _stats(values: list[float]) -> dict:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()), "n": len(values)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--sample", type=int, default=400, help="Sampled reference sequences (plus anchors)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON record (e.g. experiments/modlamp_parity_v1.json)")
    args = parser.parse_args(argv)

    import numpy as np

    from amp_challenge_2027.conditioning import (
        NUM_HMOMENT_BINS,
        NUM_HYDRO_BINS,
        NUM_CHARGE_BINS,
        axis_bin,
        charge_bin,
        hmoment_bin,
        hydrophobicity_bin,
    )

    reference = _load_reference(args.reference)
    if not reference:
        raise ValueError("reference set is empty")
    rng = np.random.default_rng(args.seed)
    sample = [reference[i] for i in rng.permutation(len(reference))[: args.sample]]
    sample = ANCHORS + sample

    local = _local_values(sample)

    parity: dict = {"available": False, "tolerance": args.tolerance, "axes": {}}
    try:
        from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor  # noqa: F401
    except ModuleNotFoundError:
        print("[parity] modlamp not installed; local-only run (parity unavailable)", flush=True)
    else:
        remote = _modlamp_values(sample)
        overall = True
        for axis in ("charge", "hydrophobicity", "hydrophobic_moment"):
            diffs = [abs(a - b) for a, b in zip(local[axis], remote[axis])]
            max_diff = max(diffs)
            passed = max_diff <= args.tolerance
            overall = overall and passed
            parity["axes"][axis] = {"max_abs_diff": max_diff, "pass": passed}
            print(f"[parity] {axis}: max |local - modlamp| = {max_diff:.3e} -> {'PASS' if passed else 'FAIL'}", flush=True)
        parity["available"] = True
        parity["pass"] = overall

    # Full-reference statistics and edge-clamp fractions for the default bins.
    full = _local_values(reference)
    reference_stats = {axis: _stats(values) for axis, values in full.items()}
    clamp = {
        "charge": {
            "num_bins": NUM_CHARGE_BINS,
            "lower": sum(1 for seq in reference if charge_bin(seq) == 0) / len(reference),
            "upper": sum(1 for seq in reference if charge_bin(seq) == NUM_CHARGE_BINS - 1) / len(reference),
        },
        "hydro": {
            "num_bins": NUM_HYDRO_BINS,
            "lower": sum(1 for seq in reference if hydrophobicity_bin(seq) == 0) / len(reference),
            "upper": sum(1 for seq in reference if hydrophobicity_bin(seq) == NUM_HYDRO_BINS - 1) / len(reference),
        },
        "hmoment": {
            "num_bins": NUM_HMOMENT_BINS,
            "lower": sum(1 for seq in reference if hmoment_bin(seq) == 0) / len(reference),
            "upper": sum(1 for seq in reference if hmoment_bin(seq) == NUM_HMOMENT_BINS - 1) / len(reference),
        },
    }
    for axis, info in clamp.items():
        print(
            f"[parity] reference {axis} edge-clamp: lower {info['lower']:.4f} / upper {info['upper']:.4f} "
            f"({info['num_bins']} bins)",
            flush=True,
        )
    for axis, stats in reference_stats.items():
        print(f"[parity] reference {axis}: mean {stats['mean']:+.4f} ± {stats['std']:.4f}", flush=True)

    anchors = [
        {
            "sequence": seq,
            "charge": local["charge"][i],
            "hydrophobicity": local["hydrophobicity"][i],
            "hydrophobic_moment": local["hydrophobic_moment"][i],
        }
        for i, seq in enumerate(ANCHORS)
    ]
    record = {
        "kind": "modlamp_parity_v1",
        "code": code_identity(),
        "reference_sha256": sha256(args.reference),
        "sample": {"size": args.sample, "seed": args.seed, "anchors": len(ANCHORS)},
        "parity": parity,
        "reference_stats": reference_stats,
        "clamp": clamp,
        "anchors": anchors,
        "note": "Parity requires modlamp (SSH). Edge-clamp fractions above ~2% on the "
        "reference mean the default bin edges clip real mass; adjust constants and "
        "re-run before any conditioned SFT.",
    }
    if args.out is not None:
        if not args.out.resolve().is_relative_to(REPO_ROOT):
            raise ValueError("--out must live inside the repository")
        write_json(args.out, record)
        print(f"[parity] wrote {args.out}", flush=True)

    if parity["available"] and not parity["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
