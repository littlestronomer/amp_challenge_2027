"""Train the ESM-2 LoRA reward ensemble (per-strain MIC + hemolysis).

Produces ``REWARD_ENSEMBLE_SIZE`` models under ``checkpoint/reward/model_{i}/``,
each trained with a different seed so the ensemble averages out individual bias.

Targets:
  - MIC head:  per-strain log-MIC regression (target = log10(mic_uM)).
    DBAASP reports species, not strains, so each measurement supervises every
    panel head of that genus (all 20 heads get data; see build_mic_targets).
  - Hemo head: binary hemolysis classification (HC50 ≤ ceiling → risky).

Run:
    uv run --extra ml python scripts/train_reward.py \\
        --mic data/processed/mic.csv \\
        --hemolysis data/processed/hemolysis.csv \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import (
    BACTERIAL_PANEL,
    DEFAULT_SEED,
    NUM_STRAINS,
    PROCESSED_DATA_DIR,
    REWARD_DIR,
    REWARD_ENSEMBLE_SIZE,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_csv_rows(path: Path) -> list[dict]:
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_mic_targets(rows: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Group MIC rows by sequence → (sequence, strain_target_vector) pairs.

    Each strain vector holds log10(MIC_uM) where measurements exist, else NaN
    (masked out of the loss). DBAASP reports target *species*, not strains, so
    a measurement is applied to **every panel head of that genus** — this keeps
    all 20 heads supervised (previously 8 heads received no data at all).
    Known limitation: MDR and non-MDR strains of one genus share the same
    target; strain-resolved data (e.g. the AIC221/AIC222 pairs) can refine this
    later.

    Curation rules owned by your collaborator. Default: raw log10(MIC_uM),
    clamped at ``MIC_CEILING_UM``, non-positive values dropped (they are
    placeholders/censoring artifacts, and log10 would fail), missing = NaN.
    """
    from amp_challenge_2027.config import MIC_CEILING_UM

    genus_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (genus, _strain, _mdr) in enumerate(BACTERIAL_PANEL):
        genus_to_idxs[genus].append(i)

    by_seq: dict[str, list[tuple[list[int], float]]] = defaultdict(list)
    for r in rows:
        seq = r["sequence"].strip().upper()
        if not tok.is_valid_sequence(seq):
            continue
        genus = r["target_organism"].strip()
        idxs = genus_to_idxs.get(genus)
        if not idxs:
            continue  # organism not in our 20-strain panel
        try:
            mic = float(r["mic_value_um"])
        except (ValueError, KeyError):
            continue
        if mic <= 0:
            continue  # placeholder/censored; log10 would crash
        mic = min(mic, MIC_CEILING_UM)
        by_seq[seq].append((idxs, math.log10(mic)))

    out: list[tuple[str, np.ndarray]] = []
    for seq, measurements in by_seq.items():
        vec = np.full(NUM_STRAINS, np.nan, dtype=np.float32)
        for idxs, val in measurements:
            for idx in idxs:
                vec[idx] = val if np.isnan(vec[idx]) else 0.5 * (vec[idx] + val)
        out.append((seq, vec))
    print(f"[reward] built MIC targets for {len(out)} unique sequences")
    return out


def build_hemo_targets(rows: list[dict]) -> list[tuple[str, int]]:
    """Hemolysis classification targets: HC50 ≤ ceiling → risky (1)."""
    from amp_challenge_2027.config import HC50_CEILING_UM

    out = []
    for r in rows:
        seq = r["sequence"].strip().upper()
        if not tok.is_valid_sequence(seq):
            continue
        try:
            hc = float(r["hc50_um"])
        except (ValueError, KeyError):
            continue
        if hc <= 0:
            continue
        out.append((seq, 1 if hc <= HC50_CEILING_UM else 0))
    print(f"[reward] built hemolysis targets for {len(out)} sequences")
    return out


# ---------------------------------------------------------------------------
# Training one ensemble member
# ---------------------------------------------------------------------------


def train_one_member(
    member_idx: int,
    mic_data: list[tuple[str, np.ndarray]],
    hemo_data: list[tuple[str, int]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    out_dir: Path,
) -> None:
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.reward import build_reward_model, save_reward_model
    from amp_challenge_2027.training import enable_determinism

    enable_determinism(seed)

    model, tokenizer = build_reward_model(device=device)
    hemo_by_seq = {s: y for s, y in hemo_data}
    mic_by_seq = dict(mic_data)  # O(1) lookups instead of a scan per sample
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Sorted union → deterministic batch composition across runs (set order
    # depends on PYTHONHASHSEED otherwise).
    all_seqs = sorted({s for s, _ in mic_data} | set(hemo_by_seq))
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(all_seqs))
        total_loss, n = 0.0, 0
        for start in range(0, len(all_seqs), batch_size):
            idx = perm[start : start + batch_size]
            batch_seqs = [all_seqs[i] for i in idx]
            enc = tokenizer(
                batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=52
            ).to(device)

            mic_logits, hemo_logits = model(enc["input_ids"], enc["attention_mask"])
            # MIC regression: masked MSE over available strain labels.
            mic_target = torch.full((len(batch_seqs), NUM_STRAINS), float("nan"), device=device)
            mic_mask = torch.zeros_like(mic_target)
            for i, s in enumerate(batch_seqs):
                vec = mic_by_seq.get(s)
                if vec is not None:
                    valid = ~np.isnan(vec)
                    mic_target[i, valid] = torch.tensor(vec[valid], device=device)
                    mic_mask[i, valid] = 1.0
            mic_target = torch.nan_to_num(mic_target, nan=0.0)
            mic_loss = ((mic_logits - mic_target) ** 2 * mic_mask).sum() / mic_mask.sum().clamp(
                min=1
            )

            # Hemolysis: BCE on sequences that have a label.
            hemo_target = torch.tensor(
                [hemo_by_seq.get(s, -1) for s in batch_seqs], dtype=torch.float32, device=device
            )
            hemo_mask = (hemo_target >= 0).float()
            hemo_target = hemo_target.clamp(min=0)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                hemo_logits, hemo_target, reduction="none"
            )
            hemo_loss = (bce * hemo_mask).sum() / hemo_mask.sum().clamp(min=1)

            loss = mic_loss + hemo_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(
            f"[reward] member {member_idx} epoch {epoch + 1}/{epochs} loss {total_loss / max(n, 1):.4f}"
        )

    save_reward_model(model, out_dir / f"model_{member_idx}")
    print(f"[reward] saved member {member_idx} → {out_dir / f'model_{member_idx}'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ESM-2 LoRA reward ensemble.")
    parser.add_argument("--mic", type=Path, default=PROCESSED_DATA_DIR / "mic.csv")
    parser.add_argument("--hemolysis", type=Path, default=PROCESSED_DATA_DIR / "hemolysis.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ensemble-size", type=int, default=REWARD_ENSEMBLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=REWARD_DIR)
    args = parser.parse_args()

    if not args.mic.exists():
        print(
            f"[reward] MIC data not found: {args.mic}. Run fetch_data + build_datasets first.",
            file=sys.stderr,
        )
        sys.exit(1)

    mic = build_mic_targets(_load_csv_rows(args.mic))
    hemo = build_hemo_targets(_load_csv_rows(args.hemolysis)) if args.hemolysis.exists() else []
    if not mic and not hemo:
        print("[reward] no supervised targets available; aborting.", file=sys.stderr)
        sys.exit(1)

    for i in range(args.ensemble_size):
        train_one_member(
            i,
            mic,
            hemo,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + i,
            device=args.device,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
