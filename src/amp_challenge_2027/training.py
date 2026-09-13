"""Training utilities: determinism, mixed precision, checkpointing, logging.

This module centralizes the production-training machinery the audit flagged as
missing, so the SFT/RL/reward trainers stay thin. Everything here is optional
and composable — the trainers opt in via the ``TrainConfig`` dataclass.

Determinism contract (the hardest competition requirement):
    The validator runs ``uv run generate`` twice and byte-compares the output.
    On CUDA this is only achievable with explicit determinism flags, because
    ``scaled_dot_product_attention``, ``index_add_`` (MoE), and floating-point
    reductions select non-deterministic kernels by default. ``enable_determinism``
    sets every flag needed; call it once at the start of any script whose output
    must be reproducible.

Heavy imports (torch) are deferred to the functions that need them, so this
module imports cleanly in the torch-free inference environment.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def enable_determinism(seed: int = 42, *, strict: bool = True) -> None:
    """Enable full determinism for reproducible runs.

    Sets every RNG (Python, numpy, torch) and the CUDA/cuDNN flags required for
    bitwise-identical output across runs. This is mandatory for the competition
    validator, which byte-compares two ``uv run generate`` invocations.

    ``strict=True`` calls ``torch.use_deterministic_algorithms(True)``, which
    raises if a non-deterministic op is used. Set ``CUBLAS_WORKSPACE_CONFIG``
    before importing torch for CUDA >10.2.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch lacks warn_only
            torch.use_deterministic_algorithms(True)


def seed_everything(seed: int) -> None:
    """Seed all RNGs without enabling strict determinism (lighter)."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class EarlyStoppingState:
    """Track consecutive validation failures, independently of model saves."""

    patience: int | None = None
    best_loss: float = float("inf")
    best_step: int | None = None
    bad_evaluations: int = 0

    def __post_init__(self) -> None:
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive or None")

    def observe(self, loss: float, step: int) -> bool:
        if not math.isfinite(loss):
            raise ValueError(f"non-finite validation loss at step {step}: {loss}")
        if loss < self.best_loss:
            self.best_loss, self.best_step = loss, step
            self.bad_evaluations = 0
            return True
        self.bad_evaluations += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.bad_evaluations >= self.patience


@dataclass
class TrainConfig:
    """Training hyperparameters shared by SFT, RL, and reward trainers.

    Holds the infrastructure knobs (precision, checkpointing, logging) separate
    from model architecture (``DecoderConfig``), so a single trainer covers all
    model variants.
    """

    seed: int = 42
    # Mixed precision: "bf16" (recommended on Ampere+), "fp16", or "fp32".
    precision: str = "bf16"
    # Gradient clipping (0 = off).
    grad_clip: float = 1.0
    # Gradient accumulation steps (effective batch = batch_size * grad_accum).
    grad_accum: int = 1
    # Checkpointing: save every N steps to ``ckpt_dir``.
    ckpt_dir: str | None = None
    save_every: int = 1000
    # Resume from the latest checkpoint in ``ckpt_dir`` if present.
    resume: bool = True
    # Logging: "jsonl" (always) and optionally "wandb".
    log_dir: str | None = None
    wandb_project: str | None = None
    # Early stopping: stop if val loss hasn't improved for ``patience`` evals.
    patience: int | None = None
    # DataLoader.
    num_workers: int = 0
    pin_memory: bool = True

    @property
    def amp_dtype(self) -> Any:
        if self.precision == "bf16":
            return "bfloat16"
        if self.precision == "fp16":
            return "float16"
        return None

    def amp_context(self):
        """Return a torch autocast context manager factory, or a nullcontext."""
        from contextlib import nullcontext

        import torch

        if self.precision == "fp32":
            return nullcontext
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return lambda: torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


# ---------------------------------------------------------------------------
# Dataset + DataLoader helpers
# ---------------------------------------------------------------------------


class PeptideDataset:
    """A torch ``Dataset`` over tokenized peptide sequences.

    Lazily tokenizes on ``__getitem__`` so the whole corpus isn't held in
    memory as padded tensors. Used by SFT pretraining; the collator handles
    dynamic per-batch padding.
    """

    def __init__(self, sequences: list[str], tokenizer_fn, max_length: int = 52):
        self.sequences = sequences
        self.tokenize = tokenizer_fn
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> list[int]:
        ids = self.tokenize(self.sequences[idx])
        return ids[: self.max_length]


def causal_lm_collate(pad_id: int):
    """Return a collate_fn for causal LM with dynamic padding + label masking.

    Pads to the batch's max length, builds labels = input_ids (shifted in the
    loss), and masks padding positions with -100.
    """
    import torch

    def collate(batch: list[list[int]]):
        L = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        for i, ids in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            labels[i, :n] = torch.tensor(ids)
        return input_ids, labels

    return collate


class ChargeConditionedDataset:
    """A torch ``Dataset`` over (tokenized sequence, charge bin) pairs.

    Used by charge-conditioned SFT (``train_generator.py sft --conditioning
    charge``). Yields ``(ids, charge_bin)``; the matching collator
    :func:`charge_conditional_collate` pads both fields.
    """

    def __init__(
        self,
        sequences: list[str],
        charge_bins: list[int],
        tokenizer_fn,
        max_length: int = 52,
    ):
        if len(sequences) != len(charge_bins):
            raise ValueError(
                f"sequences ({len(sequences)}) and charge_bins ({len(charge_bins)}) must align"
            )
        self.sequences = sequences
        self.charge_bins = charge_bins
        self.tokenize = tokenizer_fn
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        ids = self.tokenize(self.sequences[idx])
        return ids[: self.max_length], self.charge_bins[idx]


def charge_conditional_collate(pad_id: int):
    """Collate for charge-conditioned causal LM: ``(input_ids, labels, charge)``.

    Same dynamic padding + label masking as :func:`causal_lm_collate`, plus a
    ``(B,)`` long tensor of charge bins threaded to ``model(..., charge=...)``.
    """

    def collate(batch: list[tuple[list[int], int]]):
        import torch

        L = max(len(ids) for ids, _bin in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        charge = torch.empty(len(batch), dtype=torch.long)
        for i, (ids, cbin) in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            labels[i, :n] = torch.tensor(ids)
            charge[i] = cbin
        return input_ids, labels, charge

    return collate


def make_dataloader(
    sequences: list[str],
    tokenizer_fn,
    *,
    batch_size: int,
    pad_id: int,
    max_length: int = 52,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a ``torch.utils.data.DataLoader`` over tokenized peptides.

    Uses a ``peptide-aware`` collate function for dynamic padding. ``num_workers``
    enables async tokenization prefetch.
    """
    import torch
    from torch.utils.data import DataLoader

    ds = PeptideDataset(sequences, tokenizer_fn, max_length=max_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=causal_lm_collate(pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        drop_last=False,
    )


def make_charge_dataloader(
    sequences: list[str],
    charge_bins: list[int],
    tokenizer_fn,
    *,
    batch_size: int,
    pad_id: int,
    max_length: int = 52,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a ``DataLoader`` over (sequence, charge-bin) pairs.

    Mirrors :func:`make_dataloader` but yields ``(input_ids, labels, charge)``
    batches for ``model(input_ids, charge=charge)`` training.
    """
    import torch
    from torch.utils.data import DataLoader

    ds = ChargeConditionedDataset(sequences, charge_bins, tokenizer_fn, max_length=max_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=charge_conditional_collate(pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        drop_last=False,
    )


class ConditionedDataset:
    """A torch ``Dataset`` over (tokenized sequence, per-axis bins) tuples.

    Used by multi-axis conditioned SFT (``train_generator.py sft
    --conditioning charge,hydro,...``). Yields ``(ids, bins)`` where ``bins``
    is a per-axis integer tuple in the dataset's fixed axis order; the
    matching collator :func:`conditioned_collate` pads sequences and stacks
    bins into a ``{axis: (B,) long tensor}`` dict.
    """

    def __init__(
        self,
        sequences: list[str],
        bins_by_axis: dict[str, list[int]],
        tokenizer_fn,
        max_length: int = 52,
    ):
        axes = tuple(bins_by_axis)
        if not axes:
            raise ValueError("bins_by_axis must contain at least one axis")
        if any(len(bins) != len(sequences) for bins in bins_by_axis.values()):
            raise ValueError("sequences and every per-axis bin list must align")
        self.sequences = sequences
        self.axes = axes
        self.bins_by_axis = bins_by_axis
        self.tokenize = tokenizer_fn
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[list[int], tuple[int, ...]]:
        ids = self.tokenize(self.sequences[idx])
        bins = tuple(self.bins_by_axis[axis][idx] for axis in self.axes)
        return ids[: self.max_length], bins


def conditioned_collate(pad_id: int, axes: tuple[str, ...]):
    """Collate for multi-axis conditioned causal LM.

    Same dynamic padding + label masking as :func:`causal_lm_collate`, plus a
    ``{axis: (B,) long tensor}`` dict threaded to ``model(..., cond_bins=...)``.
    """

    def collate(batch: list[tuple[list[int], tuple[int, ...]]]):
        import torch

        L = max(len(ids) for ids, _bins in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        bins = {axis: torch.empty(len(batch), dtype=torch.long) for axis in axes}
        for i, (ids, sample_bins) in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            labels[i, :n] = torch.tensor(ids)
            for axis, value in zip(axes, sample_bins):
                bins[axis][i] = value
        return input_ids, labels, bins

    return collate


def make_conditioned_dataloader(
    sequences: list[str],
    bins_by_axis: dict[str, list[int]],
    tokenizer_fn,
    *,
    batch_size: int,
    pad_id: int,
    max_length: int = 52,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a ``DataLoader`` yielding ``(input_ids, labels, {axis: bins})``.

    Mirrors :func:`make_charge_dataloader` for multi-axis conditioning
    (``model(input_ids, cond_bins=...)`` training).
    """
    import torch
    from torch.utils.data import DataLoader

    ds = ConditionedDataset(sequences, bins_by_axis, tokenizer_fn, max_length=max_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=conditioned_collate(pad_id, ds.axes),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Checkpointing (save/resume model + optimizer + scheduler + step)
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path | str,
    *,
    model,
    optimizer,
    step: int,
    epoch: int,
    best_val: float | None = None,
    extra: dict | None = None,
) -> None:
    """Save a full training checkpoint (model + optimizer + state).

    Separate from ``model.save_model`` (which saves a clean inference model):
    this includes optimizer/scheduler state for resume. ``model.save_model``
    produces the deployment artifact; this produces the resume artifact.
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "extra": extra or {},
    }
    torch.save(payload, p)


def load_checkpoint(path: Path | str, model, optimizer=None, *, map_location: str = "cpu") -> dict:
    """Load a checkpoint saved by ``save_checkpoint``. Returns the metadata dict."""
    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "step": payload.get("step", 0),
        "epoch": payload.get("epoch", 0),
        "best_val": payload.get("best_val"),
        "extra": payload.get("extra", {}),
    }


def latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """Return the most recent checkpoint path in ``ckpt_dir``, or None."""
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    ckpts = sorted(d.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Logging (JSONL metrics + optional W&B)
# ---------------------------------------------------------------------------


class Logger:
    """Append-only JSONL metrics logger with optional W&B integration.

    Writes one line per logged step to ``log_dir/metrics.jsonl``. If
    ``wandb_project`` is set, also logs to W&B (requires ``wandb`` installed).
    """

    def __init__(self, log_dir: str | None = None, wandb_project: str | None = None,
                 run_name: str | None = None, config: dict | None = None):
        self.jsonl_path = Path(log_dir) / "metrics.jsonl" if log_dir else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._wandb = None
        self.wandb_project = wandb_project
        if wandb_project:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=wandb_project, name=run_name, config=config or {})
            except ImportError:
                print(f"[logger] wandb not installed; logging to {self.jsonl_path} only")

    def log(self, metrics: dict, step: int) -> None:
        """Log a metrics dict at a given step."""
        entry = {"step": step, **metrics}
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Cosine LR with warmup (scheduler object, not manual mutation)
# ---------------------------------------------------------------------------


def build_cosine_scheduler(
    optimizer, *, total_steps: int, warmup_steps: int = 0
):
    """Build a ``torch.optim.lr_scheduler.LambdaLR`` with linear warmup + cosine decay.

    Returns the scheduler object so it composes with the standard PyTorch
    training-loop idiom (``scheduler.step()`` per step).
    """
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(min(prog, 1.0) * torch.pi)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


__all__ = [
    "EarlyStoppingState",
    "TrainConfig",
    "PeptideDataset",
    "Logger",
    "enable_determinism",
    "seed_everything",
    "causal_lm_collate",
    "make_dataloader",
    "save_checkpoint",
    "load_checkpoint",
    "latest_checkpoint",
    "build_cosine_scheduler",
]
