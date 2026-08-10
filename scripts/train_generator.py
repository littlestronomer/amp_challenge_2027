"""Train the custom PeptideDecoder generator (SFT + RL).

Production training infrastructure:
    - Determinism via ``training.enable_determinism`` (critical for the validator).
    - Mixed precision (bf16/fp16) via autocast.
    - Real ``DataLoader`` with async tokenization + dynamic padding collation.
    - Mid-training checkpointing + resume (model + optimizer + scheduler + step).
    - JSONL metrics logging (+ optional W&B).
    - Cosine LR + linear warmup via a real ``LambdaLR`` scheduler.
    - Early stopping on validation loss.
    - All architecture knobs exposed (``--num-layers``, ``--hidden-size``, ...).

Architecture is config-driven: one trainer covers dense / MoE × standard /
attnres / block_attnres. Flip CLI flags to A/B test variants.

Run:
    uv run --extra ml python scripts/train_generator.py sft \\
        --data data/processed/generative.csv --epochs 10 --precision bf16
    uv run --extra ml python scripts/train_generator.py rl \\
        --sft-checkpoint checkpoint/generator --epochs 5 --lora-rank 16

    # Variants:
    ... sft --ffn moe --moe-num-experts 8 --moe-active 2
    ... sft --residual block_attnres --attnres-blocks 4
    ... sft --num-layers 8 --hidden-size 512  # larger config
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import DEFAULT_SEED, GENERATOR_DIR, MAX_LENGTH, PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_generative_dataset(path: Path) -> list[str]:
    """Load curated peptides as a list of sequences."""
    import csv

    sequences: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq = row["sequence"].strip().upper()
            if seq and tok.is_valid_sequence(seq):
                sequences.append(seq)
    print(f"[train] loaded {len(sequences)} peptides from {path}")
    if not sequences:
        print("[train] no data; aborting SFT", file=sys.stderr)
        sys.exit(1)
    return sequences


def _tokenizer_fn(seq: str) -> list[int]:
    """Tokenize for the PeptideDataset: <bos> seq <eos>, no padding."""
    return tok.encode(seq, add_bos=True, add_eos=True)


# ---------------------------------------------------------------------------
# Stage 1: SFT (production training loop)
# ---------------------------------------------------------------------------


def train_sft(
    data_path: Path,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    seed: int = DEFAULT_SEED,
    device: str = "cuda",
    out_dir: Path = GENERATOR_DIR,
    # Architecture
    residual: str = "standard",
    ffn: str = "dense",
    moe_num_experts: int = 8,
    moe_active: int = 2,
    attnres_blocks: int = 4,
    num_layers: int = 6,
    hidden_size: int = 384,
    num_heads: int = 6,
    # Training infrastructure
    precision: str = "bf16",
    grad_accum: int = 1,
    save_every: int = 1000,
    eval_every: int = 500,
    patience: int | None = None,
    log_dir: str | None = None,
    wandb_project: str | None = None,
    num_workers: int = 2,
    resume: bool = True,
) -> None:
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.generator import build_model, save_model
    from amp_challenge_2027.model import DecoderConfig
    from amp_challenge_2027.training import (
        Logger,
        build_cosine_scheduler,
        enable_determinism,
        latest_checkpoint,
        load_checkpoint,
        make_dataloader,
        save_checkpoint,
    )

    # --- Determinism (critical for the validator) -------------------------
    enable_determinism(seed)

    sequences = load_generative_dataset(data_path)
    cfg = DecoderConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        residual=residual,
        ffn=ffn,
        moe_num_experts=moe_num_experts,
        moe_num_active=moe_active,
        attnres_num_blocks=attnres_blocks,
    )
    model, _ = build_model(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{residual}+{ffn}"
    print(f"[train] model: {n_params/1e6:.1f}M params, {tag}, {num_layers}L × {hidden_size}H")

    # --- Train/val split ---------------------------------------------------
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sequences))
    split = int(0.95 * len(sequences))
    train_seqs = [sequences[i] for i in perm[:split]]
    val_seqs = [sequences[i] for i in perm[split:]]

    train_loader = make_dataloader(
        train_seqs, _tokenizer_fn, batch_size=batch_size, pad_id=tok.PAD_ID,
        max_length=MAX_LENGTH + 2, num_workers=num_workers, shuffle=True, seed=seed,
    )
    val_loader = make_dataloader(
        val_seqs, _tokenizer_fn, batch_size=batch_size, pad_id=tok.PAD_ID,
        max_length=MAX_LENGTH + 2, num_workers=num_workers, shuffle=False, seed=seed,
    ) if val_seqs else None

    # --- Optimizer + scheduler ---------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_seqs) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = int(warmup_ratio * total_steps)
    scheduler = build_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup)

    # --- Logging + checkpointing -------------------------------------------
    ckpt_dir = Path(out_dir) / "checkpoints"
    logger = Logger(log_dir=log_dir, wandb_project=wandb_project, run_name=tag,
                    config=cfg.to_dict() | {"lr": lr, "batch_size": batch_size})

    autocast_ctx = _make_autocast(precision)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None

    start_epoch, start_step, best_val = 0, 0, float("inf")
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            meta = load_checkpoint(ckpt, model, optimizer, map_location=device)
            start_epoch = meta["epoch"]
            start_step = meta["step"]
            best_val = meta.get("best_val") or float("inf")
            # Advance the scheduler to the resumed step without triggering the
            # "step before optimizer.step" warning: set the LR directly instead
            # of calling scheduler.step() in a loop.
            for _ in range(start_step):
                scheduler.step()
            # Advance scheduler state counter to match (avoids the warning).
            optimizer.step()  # no-op on zero grad; silences PyTorch's ordering check
            optimizer.zero_grad(set_to_none=True)
            print(f"[train] resumed from {ckpt} at epoch {start_epoch} step {start_step}")

    # --- Training loop -----------------------------------------------------
    step = start_step
    patience_left = patience or 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        nb = 0
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            with autocast_ctx():
                out = model(input_ids)
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = (lm_loss + out.aux_loss) / grad_accum

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step every grad_accum micro-batches.
            if (batch_idx + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                step += 1
                running += lm_loss.item()
                nb += 1

                if step % 100 == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.log({"train/lm_loss": lm_loss.item(),
                                "train/aux_loss": out.aux_loss.item(), "lr": lr_now}, step=step)
                    print(f"[train] ep {epoch+1} step {step} lm_loss {lm_loss.item():.4f} "
                          f"aux {out.aux_loss.item():.4f} lr {lr_now:.2e}")

                if save_every and step % save_every == 0:
                    save_checkpoint(ckpt_dir / f"ckpt_{step}.pt", model=model,
                                    optimizer=optimizer, step=step, epoch=epoch, best_val=best_val)
                    save_model(model, out_dir, config=cfg)  # keep inference ckpt fresh

                if eval_every and val_loader is not None and step % eval_every == 0:
                    val_loss = _evaluate(model, val_loader, device, autocast_ctx)
                    logger.log({"val/lm_loss": val_loss,
                                "val/ppl": math.exp(min(val_loss, 20))}, step=step)
                    print(f"[train] step {step} val lm_loss {val_loss:.4f} "
                          f"ppl {math.exp(min(val_loss,20)):.1f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_model(model, out_dir, config=cfg)
                        print(f"[train] new best val {best_val:.4f}; saved inference checkpoint")
                    elif patience is not None:
                        patience_left -= 1
                        if patience_left <= 0:
                            print(f"[train] early stopping at step {step} (patience={patience})")
                            save_model(model, out_dir, config=cfg)
                            logger.close()
                            return

        avg = running / max(nb, 1)
        print(f"[train] epoch {epoch+1}/{epochs} avg lm_loss {avg:.4f} "
              f"ppl {math.exp(min(avg,20)):.2f}")
        save_checkpoint(ckpt_dir / f"ckpt_epoch{epoch+1}.pt", model=model,
                        optimizer=optimizer, step=step, epoch=epoch + 1, best_val=best_val)

    save_model(model, out_dir, config=cfg)
    print(f"[train] saved generator ({tag}, {n_params/1e6:.1f}M) to {out_dir}")
    logger.close()


# ---------------------------------------------------------------------------
# Stage 2: REINFORCE toward MIC (LoRA)
# ---------------------------------------------------------------------------


def train_rl(
    sft_checkpoint: Path,
    *,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-5,
    kl_beta: float = 0.1,
    lora_rank: int = 16,
    max_len: int = MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    device: str = "cuda",
    out_dir: Path = GENERATOR_DIR,
    precision: str = "bf16",
    log_dir: str | None = None,
) -> None:
    """REINFORCE with a KL anchor to the SFT policy, using LoRA."""
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.generator import load_model, sample_sequences, save_model
    from amp_challenge_2027.lora import inject_lora, merge_and_strip_lora
    from amp_challenge_2027.reward import RewardEnsemble
    from amp_challenge_2027.training import Logger, enable_determinism

    enable_determinism(seed)

    sft_model, cfg = load_model(sft_checkpoint, map_location=device)
    sft_model.to(device)
    sft_eval, _ = load_model(sft_checkpoint, map_location=device)
    sft_eval.to(device).eval()
    for p in sft_eval.parameters():
        p.requires_grad = False

    model = inject_lora(sft_model, rank=lora_rank, alpha=lora_rank * 2)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[rl] LoRA injected: {n_tr/1e6:.2f}M / {n_tot/1e6:.1f}M trainable")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    ensemble = RewardEnsemble.load_or_fallback(device="cpu")
    if not ensemble.is_trained:
        print("[train] WARNING: no trained reward ensemble; RL uses fallback property reward")

    logger = Logger(log_dir=log_dir, run_name="rl",
                    config={"lora_rank": lora_rank, "kl_beta": kl_beta})
    autocast_ctx = _make_autocast(precision)
    gen = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        with autocast_ctx():
            samples = sample_sequences(
                model, n_sequences=batch_size, device=device, max_length=max_len, generator=gen,
            )
        valid = [s for s in samples if tok.is_valid_sequence(s)]
        if not valid:
            continue
        rewards = ensemble.score_batch(valid)
        r = torch.tensor([o.score for o in rewards], dtype=torch.float32, device=device)
        adv = (r - r.mean()) / (r.std() + 1e-6)

        with autocast_ctx():
            logp_pol = _sequence_logprobs(model, valid, device=device)
        with torch.no_grad():
            logp_ref = _sequence_logprobs(sft_eval, valid, device=device)
        kl = (logp_pol - logp_ref).clamp(max=20).mean()
        loss = -(adv * logp_pol).mean() + kl_beta * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        logger.log({"rl/reward_mean": r.mean().item(), "rl/kl": kl.item(),
                    "rl/loss": loss.item()}, step=epoch)
        print(f"[rl] ep {epoch+1}/{epochs} reward {r.mean().item():.3f} "
              f"kl {kl.item():.4f} loss {loss.item():.4f} (valid {len(valid)}/{batch_size})")

    merge_and_strip_lora(model)
    save_model(model, out_dir, config=cfg)
    print(f"[rl] saved RL-fine-tuned generator (LoRA merged) to {out_dir}")
    logger.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_autocast(precision: str):
    """Return an autocast context factory (or nullcontext for fp32)."""
    from contextlib import nullcontext

    import torch

    if precision == "fp32":
        return nullcontext
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _evaluate(model, val_loader, device: str, autocast_ctx) -> float:
    """Mean next-token loss on the validation set."""
    import torch

    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with autocast_ctx():
                out = model(input_ids)
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def _sequence_logprobs(model, sequences: list[str], *, device: str):
    """Sum of per-token log-probs assigned by ``model`` to each sequence."""
    import torch

    encoded = [tok.encode(s, add_bos=True, add_eos=True) for s in sequences]
    L = max(len(e) for e in encoded)
    input_ids = torch.full((len(encoded), L), tok.PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((len(encoded), L), dtype=torch.bool, device=device)
    for i, e in enumerate(encoded):
        input_ids[i, : len(e)] = torch.tensor(e, device=device)
        mask[i, : len(e)] = True
    logits = model(input_ids).logits
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    tok_logp = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask[:, 1:].float()).sum(dim=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom AMP generator (SFT + RL).")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_sft = sub.add_parser("sft", help="causal-LM supervised fine-tuning on MarLys")
    p_sft.add_argument("--data", type=Path, default=PROCESSED_DATA_DIR / "generative.csv")
    p_sft.add_argument("--epochs", type=int, default=10)
    p_sft.add_argument("--batch-size", type=int, default=128)
    p_sft.add_argument("--lr", type=float, default=3e-4)
    p_sft.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_sft.add_argument("--device", type=str, default="cuda")
    p_sft.add_argument("--out-dir", type=Path, default=GENERATOR_DIR)
    # Architecture
    p_sft.add_argument("--residual", type=str, default="standard",
                       choices=["standard", "attnres", "block_attnres"])
    p_sft.add_argument("--ffn", type=str, default="dense", choices=["dense", "moe"])
    p_sft.add_argument("--num-layers", type=int, default=6)
    p_sft.add_argument("--hidden-size", type=int, default=384)
    p_sft.add_argument("--num-heads", type=int, default=6)
    p_sft.add_argument("--moe-num-experts", type=int, default=8)
    p_sft.add_argument("--moe-active", type=int, default=2)
    p_sft.add_argument("--attnres-blocks", type=int, default=4)
    # Training infra
    p_sft.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p_sft.add_argument("--grad-accum", type=int, default=1)
    p_sft.add_argument("--save-every", type=int, default=1000)
    p_sft.add_argument("--eval-every", type=int, default=500)
    p_sft.add_argument("--patience", type=int, default=None)
    p_sft.add_argument("--log-dir", type=str, default=None)
    p_sft.add_argument("--wandb-project", type=str, default=None)
    p_sft.add_argument("--num-workers", type=int, default=2)
    p_sft.add_argument("--no-resume", action="store_true")

    p_rl = sub.add_parser("rl", help="REINFORCE fine-tuning toward MIC (LoRA)")
    p_rl.add_argument("--sft-checkpoint", type=Path, default=GENERATOR_DIR)
    p_rl.add_argument("--epochs", type=int, default=5)
    p_rl.add_argument("--batch-size", type=int, default=512)
    p_rl.add_argument("--lr", type=float, default=1e-5)
    p_rl.add_argument("--kl-beta", type=float, default=0.1)
    p_rl.add_argument("--lora-rank", type=int, default=16)
    p_rl.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_rl.add_argument("--device", type=str, default="cuda")
    p_rl.add_argument("--out-dir", type=Path, default=GENERATOR_DIR)
    p_rl.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p_rl.add_argument("--log-dir", type=str, default=None)

    args = parser.parse_args()
    if args.stage == "sft":
        train_sft(
            args.data,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            device=args.device, out_dir=args.out_dir,
            residual=args.residual, ffn=args.ffn,
            num_layers=args.num_layers, hidden_size=args.hidden_size, num_heads=args.num_heads,
            moe_num_experts=args.moe_num_experts, moe_active=args.moe_active,
            attnres_blocks=args.attnres_blocks,
            precision=args.precision, grad_accum=args.grad_accum,
            save_every=args.save_every, eval_every=args.eval_every, patience=args.patience,
            log_dir=args.log_dir, wandb_project=args.wandb_project,
            num_workers=args.num_workers, resume=not args.no_resume,
        )
    elif args.stage == "rl":
        train_rl(
            args.sft_checkpoint,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            kl_beta=args.kl_beta, lora_rank=args.lora_rank, seed=args.seed,
            device=args.device, out_dir=args.out_dir,
            precision=args.precision, log_dir=args.log_dir,
        )


if __name__ == "__main__":
    main()
