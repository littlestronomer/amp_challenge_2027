"""Honest AUROC of the DEPLOYED panel scorer (hub backbone + trained head).

The training-time val AUROCs (0.863 random / 0.808 clustered) describe the
fully-trained members — backbone fine-tuned. The artifact that actually
ships and scores candidates is ``classifier_panel.pt`` (head-only) rebuilt
on the hub backbone; every selection decision used THAT scorer. This script
reconstructs the promoted member's exact validation split offline and scores
the deployed scorer on it — the number that belongs in the claims.

Run (box):
    uv run python scripts/eval_deployed_panel.py \
        [--labels data/processed/activity_labels_full.csv] [--member 2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from train_reward_classifier import compute_panel_metrics, load_panel_data


def member_val_records(labels_csv: Path, *, seed: int) -> list[dict]:
    """Replicate _train_single's panel random split for one member seed."""
    records = load_panel_data(labels_csv)
    rng = np.random.default_rng(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    return shuffled[n_train:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deployed panel scorer AUROC on its val split.")
    parser.add_argument(
        "--labels", type=Path, default=Path("data/processed/activity_labels_full.csv")
    )
    parser.add_argument("--member", type=int, default=2, help="promoted member index (seed = 42+i)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    from amp_challenge_2027.config import PANEL_GENERA
    from amp_challenge_2027.score import PanelScorer

    val = member_val_records(args.labels, seed=42 + args.member)
    print(f"[deployed-eval] member {args.member} val set reconstructed: {len(val)} sequences")

    scorer = PanelScorer.load(device=args.device)
    if scorer is None:
        raise SystemExit("[deployed-eval] PanelScorer unavailable")
    assert scorer.genera == PANEL_GENERA, "genera order drifted vs training"

    seqs = [r["sequence"] for r in val]
    probs = scorer._probs(seqs)  # calibrated, deployed-forward probabilities
    logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - probs, 1e-7, 1 - 1e-7))

    targets = np.stack([r["targets"] for r in val])
    masks = np.stack([r["mask"] for r in val])
    m = compute_panel_metrics(logits, targets, masks, PANEL_GENERA)
    print(f"\n[deployed-eval] DEPLOYED scorer macro AUROC: {m['macro_auroc']:.3f}")
    for genus, auroc in sorted(m["per_genus_auroc"].items()):
        print(f"    {genus:<16} {auroc:.3f}")
    print(
        "\n[deployed-eval] reference points: fine-tuned member AUROC 0.863 "
        "(random split) / 0.808 (clustered). This number describes the exact "
        "scorer that ranked the submitted top-100."
    )


if __name__ == "__main__":
    main()
