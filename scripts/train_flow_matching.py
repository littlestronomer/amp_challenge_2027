"""Train the latent flow-matching (OT-CFM) generator in ESM-2 embedding space.

This is the second generator (the first being the AR decoder in
``train_generator.py``). The two are intentionally independent so you can
compare them head-to-head via seqme + reward scores — which is itself a
result the benchmark cares about.

Pipeline:
    1. Cache ESM-2 embeddings for all training peptides (one-time, GPU-fast).
    2. Train the denoiser to predict OT-CFM velocities v(x_t, t) = x_1 - x_0.
    3. Save the denoiser; ``flow_generate`` integrates the flow at inference.

Run:
    uv run --extra ml python scripts/train_flow_matching.py \
        --data data/processed/generative.csv --epochs 50

Compute note: embedding ~100k peptides with ESM-2 8M takes minutes on a single
GPU. The cache is written to disk so re-runs are free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import (
    CHECKPOINT_DIR,
    DEFAULT_SEED,
    MAX_LENGTH,
    PROCESSED_DATA_DIR,
)
from amp_challenge_2027.flow_matching import (
    FlowMatchingConfig,
    build_denoiser,
    load_esm2,
    ot_cfm_loss,
    save_flow_model,
)

FLOW_CHECKPOINT_DIR = CHECKPOINT_DIR / "flow_matching"


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def build_embedding_cache(
    sequences: list[str],
    model_id: str,
    *,
    device: str,
    out_path: Path,
    batch_size: int = 128,
    max_length: int = MAX_LENGTH + 2,
) -> tuple[Path, Path]:
    """Compute and cache ESM-2 embeddings for all sequences.

    Writes an ``(N, L, D)`` float16 numpy array plus a **lengths array**
    (``<out_path>.lengths.npy``, int64 token counts per sequence). The lengths
    drive the valid-position mask in the CFM loss and the decode cut — without
    them the denoiser trains on <pad> embeddings and every generated peptide
    comes out at full padded length. Re-runs are free if both files exist.
    """
    import torch

    lengths_path = out_path.with_suffix(".lengths.npy")
    if out_path.exists() and lengths_path.exists():
        print(f"[flow] embedding cache found at {out_path}; skipping")
        return out_path, lengths_path
    if out_path.exists():
        print("[flow] embedding cache present but lengths missing; rebuilding")

    print(f"[flow] embedding {len(sequences)} sequences with {model_id}")
    model, tokenizer = load_esm2(model_id, device=device)

    # Pad every batch to the SAME max_length so concatenation works.
    all_embeds = []
    all_lengths = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",  # pad ALL to max_length, not batch-max
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        all_embeds.append(out.last_hidden_state.cpu().to(torch.float16).numpy())
        # True token count (incl. <cls>/<eos>) from the attention mask.
        all_lengths.append(enc["attention_mask"].sum(dim=1).cpu().to(torch.int64).numpy())
        if start % (batch_size * 20) == 0:
            print(f"  embedded {start + len(batch)}/{len(sequences)}")

    embeds = np.concatenate(all_embeds, axis=0)
    lengths = np.concatenate(all_lengths, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeds)
    np.save(lengths_path, lengths)
    print(f"[flow] saved embedding cache: {out_path} shape={embeds.shape} (+ lengths)")
    return out_path, lengths_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_flow_matching(
    data_path: Path,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-4,
    esm_model: str = "facebook/esm2_t6_8M_UR50D",
    esm_embed_dim: int = 320,
    seed: int = DEFAULT_SEED,
    device: str = "cuda",
    out_dir: Path = FLOW_CHECKPOINT_DIR,
    cache_path: Path | None = None,
) -> None:
    import torch
    from torch.optim import AdamW

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 1. Load data ------------------------------------------------------
    import csv

    sequences: list[str] = []
    with open(data_path, newline="") as f:
        for row in csv.DictReader(f):
            seq = row["sequence"].strip().upper()
            if tok.is_valid_sequence(seq):
                sequences.append(seq)
    if not sequences:
        print("[flow] no valid sequences; aborting", file=sys.stderr)
        sys.exit(1)
    print(f"[flow] loaded {len(sequences)} peptides from {data_path}")

    # --- 2. Cache embeddings ----------------------------------------------
    cache_path = cache_path or (out_dir / f"embeds_{esm_model.split('/')[-1]}.npy")
    _, lengths_path = build_embedding_cache(
        sequences, esm_model, device=device, out_path=cache_path
    )
    embeds = np.load(cache_path)  # (N, L, D) float16
    lengths = np.load(lengths_path)  # (N,) int64 token counts
    N, L, D = embeds.shape
    print(f"[flow] cache shape: {embeds.shape}; mean tokens {lengths.mean():.1f}")
    if len(lengths) != N:
        print("[flow] FATAL: lengths/cache mismatch; delete the cache and re-run", file=sys.stderr)
        sys.exit(1)

    # --- 3. Build denoiser -------------------------------------------------
    cfg = FlowMatchingConfig(
        esm_model=esm_model,
        esm_embed_dim=D,  # read actual dim from the cache
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        max_length=L,
        num_flow_steps=50,
    )
    denoiser = build_denoiser(cfg).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"[flow] denoiser: {n_params / 1e6:.1f}M params")

    optimizer = AdamW(denoiser.parameters(), lr=lr, weight_decay=0.01)

    # --- 4. Train ----------------------------------------------------------
    rng = np.random.default_rng(seed)
    embeds_t = torch.from_numpy(embeds).to(device).float()  # (N, L, D)
    lengths_t = torch.from_numpy(lengths).to(device)  # (N,)
    positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
    step = 0
    for epoch in range(epochs):
        denoiser.train()
        perm = rng.permutation(N)
        total_loss, n_batches = 0.0, 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            x_1 = embeds_t[idx]  # (B, L, D)
            mask = (positions < lengths_t[idx].unsqueeze(1)).float()  # (B, L)
            loss = ot_cfm_loss(denoiser, x_1, cfg_cond_drop=cfg.cfg_cond_drop, mask=mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1
            if step % 100 == 0:
                print(f"[flow] epoch {epoch + 1} step {step} loss {loss.item():.5f}")
        avg = total_loss / max(n_batches, 1)
        print(f"[flow] epoch {epoch + 1}/{epochs} avg loss {avg:.5f}")

    save_flow_model(denoiser, cfg, out_dir)
    print(f"[flow] saved flow-matching model to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the latent flow-matching generator.")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_DIR / "generative.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--esm-model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="frozen ESM-2 encoder to flow-match in (8M is fast; 650M is higher-fidelity)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=FLOW_CHECKPOINT_DIR)
    args = parser.parse_args()

    train_flow_matching(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        esm_model=args.esm_model,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
