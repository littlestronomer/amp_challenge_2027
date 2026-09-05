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
import hashlib
import json
import math
import sys
from pathlib import Path

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import (
    DEFAULT_SEED,
    GENERATOR_DIR,
    MAX_LENGTH,
    PROCESSED_DATA_DIR,
    RL_GENERATOR_DIR,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_generative_dataset(path: Path, *, conditioning: str = "none"):
    """Load curated peptides; with ``conditioning="charge"`` also compute charge bins.

    Returns ``list[str]`` for unconditional training, or ``(sequences,
    charge_bins)`` for charge-conditioned training. The CSV's ``charge`` column
    is deliberately ignored — bins are recomputed with the modlamp (Bjellqvist)
    charge so training labels live in the same scale as the ConformityScore.
    """
    import csv

    from amp_challenge_2027.conditioning import charge_bin

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
    if conditioning != "charge":
        return sequences
    charge_bins = [charge_bin(s) for s in sequences]
    import numpy as np

    hist = np.bincount(charge_bins, minlength=21)
    print(
        f"[train] charge bins: min={min(charge_bins)} max={max(charge_bins)} "
        f"mode_bin={int(hist.argmax())} (histogram {hist.tolist()})"
    )
    return sequences, charge_bins


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
    split_seed: int | None = None,
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
    # Charge conditioning
    conditioning: str = "none",
) -> None:
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.generator import build_model, save_model
    from amp_challenge_2027.model import DecoderConfig
    from amp_challenge_2027.training import (
        EarlyStoppingState,
        Logger,
        build_cosine_scheduler,
        enable_determinism,
        latest_checkpoint,
        load_checkpoint,
        make_charge_dataloader,
        make_dataloader,
        save_checkpoint,
    )

    # --- Determinism (critical for the validator) -------------------------
    enable_determinism(seed)

    if conditioning not in ("none", "charge"):
        raise ValueError(f"unknown conditioning {conditioning!r}; expected 'none' or 'charge'")

    if conditioning == "charge":
        sequences, charge_bins = load_generative_dataset(data_path, conditioning="charge")
    else:
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
        conditioning=conditioning,
    )
    model, _ = build_model(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{residual}+{ffn}" + (f"+cond-{conditioning}" if conditioning != "none" else "")
    print(f"[train] model: {n_params / 1e6:.1f}M params, {tag}, {num_layers}L × {hidden_size}H")

    # --- Train/val split ---------------------------------------------------
    import numpy as np

    effective_split_seed = seed if split_seed is None else split_seed
    rng = np.random.default_rng(effective_split_seed)
    perm = rng.permutation(len(sequences))
    split = int(0.95 * len(sequences))
    train_seqs = [sequences[i] for i in perm[:split]]
    val_seqs = [sequences[i] for i in perm[split:]]

    # The corpus and split are fixed across initialization seeds when
    # --split-seed is supplied. Resume must use the same experiment inputs.
    out_dir = Path(out_dir)
    manifest = {
        "format_version": 2,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "seed": seed,
        "split_seed": effective_split_seed,
        "train_sha256": hashlib.sha256("\n".join(train_seqs).encode()).hexdigest(),
        "val_sha256": hashlib.sha256("\n".join(val_seqs).encode()).hexdigest(),
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "model": cfg.to_dict(),
    }
    manifest_path = out_dir / "training_manifest.json"
    previous = latest_checkpoint(out_dir / "checkpoints")
    if resume and previous is not None:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Resume corpus, split, model, or trainer version differs. "
                "Start this experiment in a new --out-dir; historical runs remain usable for sweeps."
            )
    elif previous is not None or (out_dir / "model.pt").exists():
        raise ValueError("Output already contains model artifacts; use a new --out-dir for a fresh run")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    print(f"[train] split seed {effective_split_seed}: {len(train_seqs)} train / {len(val_seqs)} val")

    if conditioning == "charge":
        train_bins = [charge_bins[i] for i in perm[:split]]
        val_bins = [charge_bins[i] for i in perm[split:]]
        train_loader = make_charge_dataloader(
            train_seqs,
            train_bins,
            _tokenizer_fn,
            batch_size=batch_size,
            pad_id=tok.PAD_ID,
            max_length=MAX_LENGTH + 2,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
        )
        val_loader = (
            make_charge_dataloader(
                val_seqs,
                val_bins,
                _tokenizer_fn,
                batch_size=batch_size,
                pad_id=tok.PAD_ID,
                max_length=MAX_LENGTH + 2,
                num_workers=num_workers,
                shuffle=False,
                seed=seed,
            )
            if val_seqs
            else None
        )
    else:
        train_loader = make_dataloader(
            train_seqs,
            _tokenizer_fn,
            batch_size=batch_size,
            pad_id=tok.PAD_ID,
            max_length=MAX_LENGTH + 2,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
        )
        val_loader = (
            make_dataloader(
                val_seqs,
                _tokenizer_fn,
                batch_size=batch_size,
                pad_id=tok.PAD_ID,
                max_length=MAX_LENGTH + 2,
                num_workers=num_workers,
                shuffle=False,
                seed=seed,
            )
            if val_seqs
            else None
        )

    # --- Optimizer + scheduler ---------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_seqs) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = int(warmup_ratio * total_steps)
    scheduler = build_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup)

    # --- Logging + checkpointing -------------------------------------------
    ckpt_dir = Path(out_dir) / "checkpoints"
    logger = Logger(
        log_dir=log_dir,
        wandb_project=wandb_project,
        run_name=tag,
        config=cfg.to_dict() | {"lr": lr, "batch_size": batch_size},
    )

    autocast_ctx = _make_autocast(precision)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None

    start_epoch, start_step = 0, 0
    stopping = EarlyStoppingState(patience=patience)
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            meta = load_checkpoint(ckpt, model, optimizer, map_location=device)
            start_epoch = meta["epoch"]
            start_step = meta["step"]
            if meta.get("best_val") is not None:
                stopping.best_loss = meta["best_val"]
            stopping.best_step = meta["extra"].get("best_step")
            stopping.bad_evaluations = meta["extra"].get("bad_evaluations", 0)
            if stopping.best_step is not None and not (out_dir / "model.pt").exists():
                raise ValueError("Cannot resume: the saved best inference model is missing")
            # Advance the scheduler to the start of the RESUMED EPOCH's step
            # range, not to ``start_step``: mid-epoch checkpoints restart that
            # whole epoch, so stepping to start_step would double-count the
            # pre-checkpoint portion of it against the cosine schedule.
            sched_target = int(meta["extra"].get("epoch_start_step", start_step))
            for _ in range(sched_target):
                scheduler.step()
            # One silent optimizer step so LambdaLR's ordering check doesn't
            # warn about scheduler.step() before optimizer.step().
            optimizer.step()  # no-op on empty grads
            optimizer.zero_grad(set_to_none=True)
            print(
                f"[train] resumed from {ckpt} at epoch {start_epoch} "
                f"(scheduler advanced to step {sched_target})"
            )

    # --- Training loop -----------------------------------------------------
    step = start_step

    def _save_training(path: Path, epoch: int, epoch_start: int) -> None:
        save_checkpoint(
            path, model=model, optimizer=optimizer, step=step, epoch=epoch,
            best_val=stopping.best_loss if math.isfinite(stopping.best_loss) else None,
            extra={
                "epoch_start_step": epoch_start,
                "best_step": stopping.best_step,
                "bad_evaluations": stopping.bad_evaluations,
            },
        )

    def _optimizer_step(epoch: int) -> None:
        """Clip, step optimizer+scheduler, log, checkpoint, early-stopping eval."""
        nonlocal step
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

        if step % 100 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.log(
                {
                    "train/lm_loss": lm_loss.item(),
                    "train/aux_loss": out.aux_loss.item(),
                    "lr": lr_now,
                },
                step=step,
            )
            print(
                f"[train] ep {epoch + 1} step {step} lm_loss {lm_loss.item():.4f} "
                f"aux {out.aux_loss.item():.4f} lr {lr_now:.2e}"
            )

        if eval_every and val_loader is not None and step % eval_every == 0:
            val_loss = _evaluate(model, val_loader, device, autocast_ctx)
            logger.log({"val/lm_loss": val_loss, "val/ppl": math.exp(min(val_loss, 20))}, step=step)
            print(
                f"[train] step {step} val lm_loss {val_loss:.4f} "
                f"ppl {math.exp(min(val_loss, 20)):.1f}"
            )
            if stopping.observe(val_loss, step):
                save_model(model, out_dir, config=cfg)
                (out_dir / "selection.json").write_text(json.dumps({
                    "criterion": "validation_loss", "step": step, "val_loss": val_loss,
                }, indent=2) + "\n")
                print(f"[train] new best val {stopping.best_loss:.4f}; saved inference checkpoint")
            if stopping.should_stop:
                print(f"[train] early stopping at step {step} (patience={patience})")
                _save_training(ckpt_dir / f"ckpt_stop_{step}.pt", epoch, epoch_start_step)
                raise _EarlyStop()

        if save_every and step % save_every == 0:
            _save_training(ckpt_dir / f"ckpt_{step}.pt", epoch, epoch_start_step)

    for epoch in range(start_epoch, epochs):
        epoch_start_step = step
        try:
            model.train()
            running = 0.0
            nb = 0
            n_micro = 0
            for batch_idx, batch in enumerate(train_loader):
                if conditioning == "charge":
                    input_ids, labels, charge = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    charge = charge.to(device)
                else:
                    input_ids, labels = batch
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    charge = None
                # Zero grads at the START of each accumulation window (not every
                # micro-batch — that erased everything accumulated so far).
                if n_micro % grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    out = model(input_ids, charge=charge)
                    shift_logits = out.logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    lm_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    loss = (lm_loss + out.aux_loss) / grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                n_micro += 1

                # Step once per full accumulation window.
                if n_micro % grad_accum == 0:
                    _optimizer_step(epoch)
                    running += lm_loss.item()
                    nb += 1

            # Flush a trailing partial window so its gradients aren't lost.
            if n_micro % grad_accum != 0:
                _optimizer_step(epoch)
                running += lm_loss.item()
                nb += 1
        except _EarlyStop:
            break

        avg = running / max(nb, 1)
        print(
            f"[train] epoch {epoch + 1}/{epochs} avg lm_loss {avg:.4f} "
            f"ppl {math.exp(min(avg, 20)):.2f}"
        )
        _save_training(ckpt_dir / f"ckpt_epoch{epoch + 1}.pt", epoch + 1, step)

    save_model(model, out_dir / "last", config=cfg)
    if stopping.best_step is None:
        save_model(model, out_dir, config=cfg)
        (out_dir / "selection.json").write_text(json.dumps({
            "criterion": "last_no_validation", "step": step, "val_loss": None,
        }, indent=2) + "\n")
    print(
        f"[train] saved last generator to {out_dir / 'last'}; "
        f"inference model at {out_dir} (best validation step={stopping.best_step})"
    )
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
    out_dir: Path = RL_GENERATOR_DIR,
    precision: str = "bf16",
    log_dir: str | None = None,
    resume: bool = True,
) -> None:
    """REINFORCE with a KL anchor to the SFT policy, using LoRA.

    Output goes to ``checkpoint/generator_rl`` by default — never over the SFT
    checkpoint it fine-tunes (that would poison any later RL re-run's KL
    anchor). Per-epoch checkpoints under ``<out_dir>/checkpoints/`` allow
    resume; the sampling RNG state is not checkpointed, so a resumed run
    continues from fresh samples.
    """
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.generator import load_model, sample_sequences, save_model
    from amp_challenge_2027.lora import inject_lora, merge_and_strip_lora
    from amp_challenge_2027.reward import RewardEnsemble
    from amp_challenge_2027.training import (
        Logger,
        enable_determinism,
        latest_checkpoint,
        load_checkpoint,
        save_checkpoint,
    )

    enable_determinism(seed)

    if Path(out_dir).resolve() == Path(sft_checkpoint).resolve():
        raise SystemExit(
            f"[rl] refusing to write RL output over its own SFT anchor ({out_dir}); "
            "pass --out-dir (default: checkpoint/generator_rl)."
        )

    sft_model, cfg = load_model(sft_checkpoint, map_location=device)
    sft_model.to(device)
    sft_eval, _ = load_model(sft_checkpoint, map_location=device)
    sft_eval.to(device).eval()
    for p in sft_eval.parameters():
        p.requires_grad = False

    model = inject_lora(sft_model, rank=lora_rank, alpha=lora_rank * 2)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[rl] LoRA injected: {n_tr / 1e6:.2f}M / {n_tot / 1e6:.1f}M trainable")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    ckpt_dir = Path(out_dir) / "checkpoints"
    start_epoch = 0
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            meta = load_checkpoint(ckpt, model, optimizer, map_location=device)
            start_epoch = meta["epoch"]
            print(f"[rl] resumed from {ckpt} at epoch {start_epoch}")

    ensemble = RewardEnsemble.load_or_fallback(device="cpu")
    if not ensemble.is_trained:
        print("[train] WARNING: no trained reward ensemble; RL uses fallback property reward")

    logger = Logger(
        log_dir=log_dir, run_name="rl", config={"lora_rank": lora_rank, "kl_beta": kl_beta}
    )
    autocast_ctx = _make_autocast(precision)
    gen = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(start_epoch, epochs):
        # Rollouts must come from the CURRENT POLICY IN EVAL MODE with no
        # autograd: dropout-on sampling breaks REINFORCE's assumption that the
        # replayed log-probs describe the sampling distribution, and building a
        # graph through 50 sampling forwards wastes memory we never backprop.
        model.eval()
        with torch.no_grad(), autocast_ctx():
            samples = sample_sequences(
                model,
                n_sequences=batch_size,
                device=device,
                max_length=max_len,
                generator=gen,
            )
        valid = [s for s in samples if tok.is_valid_sequence(s)]
        if len(valid) < 2:
            print(
                f"[rl] ep {epoch + 1}: only {len(valid)} valid sample(s); skipping update "
                "(group-relative advantages need ≥2)"
            )
            continue
        rewards = ensemble.score_batch(valid)
        r = torch.tensor([o.score for o in rewards], dtype=torch.float32, device=device)
        adv = (r - r.mean()) / (r.std() + 1e-6)
        assert torch.isfinite(adv).all(), (
            "non-finite advantages; aborting before corrupting weights"
        )

        model.train()
        with autocast_ctx():
            logp_pol = _sequence_logprobs(model, valid, device=device)
        with torch.no_grad():
            logp_ref = _sequence_logprobs(sft_eval, valid, device=device)
        kl = (logp_pol - logp_ref).clamp(max=20).mean()
        loss = -(adv * logp_pol).mean() + kl_beta * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        logger.log(
            {"rl/reward_mean": r.mean().item(), "rl/kl": kl.item(), "rl/loss": loss.item()},
            step=epoch + 1,
        )
        print(
            f"[rl] ep {epoch + 1}/{epochs} reward {r.mean().item():.3f} "
            f"kl {kl.item():.4f} loss {loss.item():.4f} (valid {len(valid)}/{batch_size})"
        )

        save_checkpoint(
            ckpt_dir / f"ckpt_epoch{epoch + 1}.pt",
            model=model,
            optimizer=optimizer,
            step=epoch + 1,
            epoch=epoch + 1,
        )

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


class _EarlyStop(Exception):
    """Unwinds the SFT epoch loop when patience runs out (see _optimizer_step)."""


def _evaluate(model, val_loader, device: str, autocast_ctx) -> float:
    """Mean next-token loss on the validation set (handles both batch formats)."""
    import torch

    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:  # charge-conditioned: (input_ids, labels, charge)
                input_ids, labels, charge = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                charge = charge.to(device)
            else:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(device), labels.to(device)
                charge = None
            with autocast_ctx():
                out = model(input_ids, charge=charge)
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
    p_sft.add_argument(
        "--split-seed", type=int, default=None,
        help="Fixed train/validation split seed (default: same as --seed for legacy compatibility)",
    )
    p_sft.add_argument("--device", type=str, default="cuda")
    p_sft.add_argument("--out-dir", type=Path, default=GENERATOR_DIR)
    # Architecture
    p_sft.add_argument(
        "--residual", type=str, default="standard", choices=["standard", "attnres", "block_attnres"]
    )
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
    p_sft.add_argument(
        "--conditioning",
        type=str,
        default="none",
        choices=["none", "charge"],
        help=(
            "Train a charge-conditioned generator: condition on the peptide's modlamp "
            "net charge (21 bins, -8..+12). At generation time, sample bins from the "
            "reference charge distribution to reproduce it (see generate --charge-conditioned)."
        ),
    )

    p_rl = sub.add_parser("rl", help="REINFORCE fine-tuning toward MIC (LoRA)")
    p_rl.add_argument("--sft-checkpoint", type=Path, default=GENERATOR_DIR)
    p_rl.add_argument("--epochs", type=int, default=5)
    p_rl.add_argument("--batch-size", type=int, default=512)
    p_rl.add_argument("--lr", type=float, default=1e-5)
    p_rl.add_argument("--kl-beta", type=float, default=0.1)
    p_rl.add_argument("--lora-rank", type=int, default=16)
    p_rl.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_rl.add_argument("--device", type=str, default="cuda")
    p_rl.add_argument(
        "--out-dir",
        type=Path,
        default=RL_GENERATOR_DIR,
        help="RL output (default checkpoint/generator_rl; never the SFT dir)",
    )
    p_rl.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p_rl.add_argument("--log-dir", type=str, default=None)
    p_rl.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    if args.stage == "sft":
        train_sft(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            split_seed=args.split_seed,
            device=args.device,
            out_dir=args.out_dir,
            residual=args.residual,
            ffn=args.ffn,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            moe_num_experts=args.moe_num_experts,
            moe_active=args.moe_active,
            attnres_blocks=args.attnres_blocks,
            precision=args.precision,
            grad_accum=args.grad_accum,
            save_every=args.save_every,
            eval_every=args.eval_every,
            patience=args.patience,
            log_dir=args.log_dir,
            wandb_project=args.wandb_project,
            num_workers=args.num_workers,
            resume=not args.no_resume,
            conditioning=args.conditioning,
        )
    elif args.stage == "rl":
        train_rl(
            args.sft_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            kl_beta=args.kl_beta,
            lora_rank=args.lora_rank,
            seed=args.seed,
            device=args.device,
            out_dir=args.out_dir,
            precision=args.precision,
            log_dir=args.log_dir,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
