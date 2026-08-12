"""Train a binary activity classifier as the Phase-2 reward model.

Uses frozen ESM-2 + a small classification head to predict whether a peptide is
antimicrobially active. Trained on the DBAASP high/low activity data from the
HydrAMP starter kit:

  - E. coli: 929 active + 394 inactive
  - S. aureus: 586 active + 299 inactive

The classifier serves two purposes:
  1. Replaces the fallback property scorer in generate.py — ranks candidates
     by predicted activity instead of charge/hydrophobicity heuristic.
  2. Becomes the reward signal for RL fine-tuning (REINFORCE toward activity).

Run:
    uv run --extra ml python scripts/train_reward_classifier.py \\
        --data data/processed/activity_labels.csv --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import DEFAULT_SEED, REWARD_DIR

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_activity_data(path: Path) -> list[dict]:
    """Load the activity_labels.csv into training records.

    Returns a list of {sequence, label (0=inactive, 1=active), organism}.
    """
    import csv

    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq = row["sequence"].strip().upper()
            label = 1 if row["label"].strip().lower() == "active" else 0
            records.append({
                "sequence": seq,
                "label": label,
                "organism": row.get("organism", "").strip(),
            })
    print(f"[reward] loaded {len(records)} labeled sequences from {path}")
    active = sum(1 for r in records if r["label"] == 1)
    inactive = len(records) - active
    print(f"[reward]   active: {active}, inactive: {inactive}")
    return records


# ---------------------------------------------------------------------------
# Model: frozen ESM-2 + classification head
# ---------------------------------------------------------------------------


def build_classifier(esm_model: str, device: str = "cpu"):
    """Build a frozen ESM-2 encoder + trainable classification head.

    Returns (model, tokenizer). The ESM-2 backbone is frozen; only the
    classification head (pooler → dense → binary output) trains.
    """
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(esm_model)
    esm = AutoModel.from_pretrained(esm_model)
    hidden = esm.config.hidden_size

    for p in esm.parameters():
        p.requires_grad = False

    class ActivityClassifier(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size // 2)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_size // 2, 1)

        def forward(self, input_ids, attention_mask):
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            # Mean-pool over sequence (mask-aware).
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            x = self.drop(self.act(self.dense(pooled)))
            return self.classifier(x).squeeze(-1)

    model = ActivityClassifier(hidden)
    model.esm = esm  # attach frozen encoder
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_path: Path,
    *,
    esm_model: str = "facebook/esm2_t6_8M_UR50D",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = DEFAULT_SEED,
    device: str = "cuda",
    out_dir: Path = REWARD_DIR,
) -> None:
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.training import build_cosine_scheduler, enable_determinism

    enable_determinism(seed)

    records = load_activity_data(data_path)
    if not records:
        print("[reward] no data; aborting", file=sys.stderr)
        sys.exit(1)

    # Stratified train/val split (80/20)
    rng = np.random.default_rng(seed)
    active = [r for r in records if r["label"] == 1]
    inactive = [r for r in records if r["label"] == 0]
    rng.shuffle(active)
    rng.shuffle(inactive)
    n_train_active = int(0.8 * len(active))
    n_train_inactive = int(0.8 * len(inactive))
    train = active[:n_train_active] + inactive[:n_train_inactive]
    val = active[n_train_active:] + inactive[n_train_inactive:]
    rng.shuffle(train)
    rng.shuffle(val)
    print(f"[reward] train: {len(train)} (act={n_train_active}, inact={n_train_inactive})")
    print(f"[reward] val: {len(val)}")

    model, tokenizer = build_classifier(esm_model, device=device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[reward] model: {n_trainable/1e6:.2f}M trainable / {n_total/1e6:.1f}M total ({esm_model})")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = epochs * math.ceil(len(train) / batch_size)
    scheduler = build_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(0.05 * total_steps))

    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        rng.shuffle(train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(train), batch_size):
            batch = train[start : start + batch_size]
            seqs = [r["sequence"] for r in batch]
            labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
            enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(val), batch_size):
                batch = val[start : start + batch_size]
                seqs = [r["sequence"] for r in batch]
                labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
                enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
                logits = model(enc["input_ids"], enc["attention_mask"])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(batch)

        avg_train = total_loss / max(n_batches, 1)
        avg_val = val_loss / max(math.ceil(len(val) / batch_size), 1)
        val_acc = val_correct / max(val_total, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = avg_val
            _save_classifier(model, tokenizer, esm_model, out_dir)
            improved = " ← saved (best)"

        print(
            f"[reward] epoch {epoch+1}/{epochs} "
            f"train_loss {avg_train:.4f} val_loss {avg_val:.4f} "
            f"val_acc {val_acc:.3f} lr {lr_now:.2e}{improved}"
        )

    print(f"\n[reward] best val accuracy: {best_val_acc:.3f}")
    print(f"[reward] saved to {out_dir}")


def _save_classifier(model, tokenizer, esm_model: str, out_dir: Path) -> None:
    """Save the classifier head + config."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save only the trainable head (ESM-2 is reloaded from HF at inference).
    head_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("esm.")
    }
    torch.save(head_state, out_dir / "classifier_head.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "esm_model": esm_model,
        "type": "binary_activity_classifier",
    }, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase-2 activity classifier (reward model).")
    parser.add_argument("--data", type=Path, default=Path("data/processed/activity_labels.csv"))
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t6_8M_UR50D",
                        help="ESM-2 backbone (8M fast, 650M for quality)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=REWARD_DIR)
    args = parser.parse_args()

    train(
        args.data,
        esm_model=args.esm_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
