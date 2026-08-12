"""Train a binary activity classifier as the Phase-2 reward model.

Uses ESM-2 (partially fine-tuned) + classification head to predict whether a
peptide is antimicrobially active. Trained on DBAASP high/low activity data.

Key design choices for the small-data regime (~2000 sequences):
  - Unfreezes the top N ESM-2 layers (not just a frozen backbone) — the 8M
    model's frozen embeddings are too weak to separate active from inactive.
  - Class-weighted BCE loss to handle the 2:1 active/inactive imbalance.
  - Reports AUROC and F1, not just accuracy (accuracy is misleading on
    imbalanced data — majority-class baseline is 67.6%).

Run:
    uv run --extra ml python scripts/train_reward_classifier.py \\
        --data data/processed/activity_labels.csv --epochs 50 \\
        --esm-model facebook/esm2_t12_35M_UR50D --unfreeze-layers 4
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
    """Load activity_labels.csv → [{sequence, label (0/1), organism}]."""
    import csv

    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq = row["sequence"].strip().upper()
            label = 1 if row["label"].strip().lower() == "active" else 0
            records.append({"sequence": seq, "label": label})
    active = sum(1 for r in records if r["label"] == 1)
    print(f"[reward] loaded {len(records)} sequences (active={active}, inactive={len(records)-active})")
    return records


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_classifier(esm_model: str, unfreeze_layers: int, device: str = "cpu"):
    """Build ESM-2 (top N layers trainable) + classification head.

    ``unfreeze_layers=0`` = fully frozen (original behavior).
    ``unfreeze_layers=4`` = top 4 transformer layers + embeddings trainable.
    This is critical for small datasets where frozen 8M embeddings are too weak.
    """
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(esm_model)
    esm = AutoModel.from_pretrained(esm_model)
    hidden = esm.config.hidden_size
    n_layers = esm.config.num_hidden_layers

    # Freeze all, then unfreeze top N layers.
    for p in esm.parameters():
        p.requires_grad = False
    if unfreeze_layers > 0:
        # Unfreeze embeddings + top N encoder layers.
        if hasattr(esm, "embeddings"):
            for p in esm.embeddings.parameters():
                p.requires_grad = True
        layers = getattr(esm.encoder, "layer", None) or getattr(esm.encoder, "layers", None)
        if layers is not None:
            for layer in layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    class ActivityClassifier(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.2)
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            x = self.drop(self.act(self.dense(pooled)))
            x = self.drop(self.act(self.dense(x)))
            return self.classifier(x).squeeze(-1)

    model = ActivityClassifier(hidden)
    model.esm = esm
    model.to(device)
    return model, tokenizer, n_layers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(logits, labels):
    """Compute accuracy, AUROC, F1, precision, recall."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    metrics = {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auroc"] = 0.5
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_path: Path,
    *,
    esm_model: str = "facebook/esm2_t12_35M_UR50D",
    unfreeze_layers: int = 4,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 5e-5,
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

    # Stratified 80/20 split.
    rng = np.random.default_rng(seed)
    active = [r for r in records if r["label"] == 1]
    inactive = [r for r in records if r["label"] == 0]
    rng.shuffle(active)
    rng.shuffle(inactive)
    na, ni = int(0.8 * len(active)), int(0.8 * len(inactive))
    train_recs = active[:na] + inactive[:ni]
    val_recs = active[na:] + inactive[ni:]
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    print(f"[reward] train: {len(train_recs)} (act={na}, inact={ni})")
    print(f"[reward] val: {len(val_recs)}")

    # Class weights (inverse frequency) to handle the 2:1 imbalance.
    n_active = sum(1 for r in train_recs if r["label"] == 1)
    n_inactive = len(train_recs) - n_active
    pos_weight = torch.tensor([n_inactive / max(n_active, 1)], device=device)
    print(f"[reward] class weight (pos): {pos_weight.item():.3f}")

    model, tokenizer, n_layers = build_classifier(esm_model, unfreeze_layers, device=device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[reward] model: {n_trainable/1e6:.2f}M trainable / {n_total/1e6:.1f}M total "
        f"({esm_model}, {n_layers}L, unfrozen={unfreeze_layers})"
    )

    # Single LR for all trainable params — works fine for small models.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = epochs * math.ceil(len(train_recs) / batch_size)
    scheduler = build_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(0.1 * total_steps))

    best_auroc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        rng.shuffle(train_recs)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(train_recs), batch_size):
            batch = train_recs[start : start + batch_size]
            seqs = [r["sequence"] for r in batch]
            labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
            enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        # --- Validate ---
        model.eval()
        all_logits, all_labels = [], []
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_recs), batch_size):
                batch = val_recs[start : start + batch_size]
                seqs = [r["sequence"] for r in batch]
                labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
                enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
                logits = model(enc["input_ids"], enc["attention_mask"])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                val_loss_total += loss.item()
                val_batches += 1
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        m = compute_metrics(np.array(all_logits), np.array(all_labels))
        avg_train = total_loss / max(n_batches, 1)
        avg_val = val_loss_total / max(val_batches, 1)

        improved = ""
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            _save_classifier(model, tokenizer, esm_model, unfreeze_layers, out_dir)
            improved = " ← saved (best)"

        if (epoch + 1) % 5 == 0 or improved:
            print(
                f"[reward] epoch {epoch+1}/{epochs} "
                f"train_loss {avg_train:.4f} val_loss {avg_val:.4f} "
                f"acc {m['acc']:.3f} auroc {m['auroc']:.3f} "
                f"f1 {m['f1']:.3f}{improved}"
            )

    print(f"\n[reward] best AUROC: {best_auroc:.3f}")
    print(f"[reward] saved to {out_dir}")


def _save_classifier(model, tokenizer, esm_model: str, unfreeze_layers: int, out_dir: Path) -> None:
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "classifier.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "esm_model": esm_model,
        "unfreeze_layers": unfreeze_layers,
        "type": "binary_activity_classifier",
    }, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase-2 activity classifier (reward model).")
    parser.add_argument("--data", type=Path, default=Path("data/processed/activity_labels.csv"))
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t12_35M_UR50D",
                        help="ESM-2 backbone (35M recommended for this data size)")
    parser.add_argument("--unfreeze-layers", type=int, default=4,
                        help="Number of top ESM-2 layers to fine-tune (0=frozen)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=REWARD_DIR)
    args = parser.parse_args()

    train(
        args.data,
        esm_model=args.esm_model,
        unfreeze_layers=args.unfreeze_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
