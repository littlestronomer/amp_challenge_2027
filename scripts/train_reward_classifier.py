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

from amp_challenge_2027.config import DEFAULT_SEED, PROCESSED_DATA_DIR, REWARD_DIR

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
            raw = row["label"].strip().lower()
            # Handle both string ("active"/"inactive") and numeric (1/0) labels.
            if raw in ("active", "1", "1.0", "true"):
                label = 1
            elif raw in ("inactive", "0", "0.0", "false"):
                label = 0
            else:
                continue  # skip unknown labels
            records.append({"sequence": seq, "label": label})
    active = sum(1 for r in records if r["label"] == 1)
    print(
        f"[reward] loaded {len(records)} sequences (active={active}, inactive={len(records) - active})"
    )
    return records


def load_panel_data(path: Path) -> list[dict]:
    """Aggregate activity_labels_full.csv → per-sequence multi-hot targets.

    Long format (sequence, organism, mic_um, band, label). Each sequence
    becomes ``{sequence, targets[G], mask[G]}`` over ``PANEL_GENERA`` order:
    active → 1, inactive → 0, blank/absent → masked. Conflicting rows for the
    same (sequence, genus) resolve to active (conservative for a reward).
    """
    import csv

    from amp_challenge_2027 import tokenizer as tok
    from amp_challenge_2027.config import MAX_LENGTH, MIN_LENGTH, PANEL_GENERA
    from amp_challenge_2027.data import SPECIES_TO_PANEL

    genus_index = {g: i for i, g in enumerate(PANEL_GENERA)}
    per_seq: dict[str, dict[int, float]] = {}
    skipped_organism = 0
    skipped_sequence = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().lower()
            if label not in ("active", "inactive"):
                continue  # masked band — carries no training evidence
            seq = row["sequence"].strip().upper()
            # tok.is_valid_sequence is alphabet-only; the competition length
            # bounds live here (mirrors select.is_valid_sequence).
            if not tok.is_valid_sequence(seq) or not (MIN_LENGTH <= len(seq) <= MAX_LENGTH):
                skipped_sequence += 1
                continue
            organism = (row.get("organism") or "").strip()
            genus = (
                organism
                if organism in genus_index
                else next(
                    (g for k, g in SPECIES_TO_PANEL.items() if k.lower() in organism.lower()), ""
                )
            )
            if genus not in genus_index:
                skipped_organism += 1
                continue
            per_seq.setdefault(seq, {})[genus_index[genus]] = 1.0 if label == "active" else 0.0

    import numpy as np

    n_genera = len(genus_index)
    records = []
    for seq in sorted(per_seq):
        evidence = per_seq[seq]
        # Zero-filled targets + separate mask. (NaN targets would poison the
        # loss even under masking: 0 * NaN = NaN.) The mask is the only
        # source of truth for "has evidence".
        targets = np.zeros(n_genera, dtype=np.float32)
        mask = np.zeros(n_genera, dtype=np.float32)
        for idx, val in evidence.items():
            targets[idx] = val
            mask[idx] = 1.0
        records.append({"sequence": seq, "targets": targets, "mask": mask})

    covered = np.stack([r["mask"] for r in records]).sum(axis=0).astype(int) if records else []
    print(
        f"[reward] panel data: {len(records)} sequences × {n_genera} genera "
        f"(unmapped organisms skipped: {skipped_organism}, invalid sequences: {skipped_sequence})"
    )
    print("[reward] per-genus coverage:", dict(zip(PANEL_GENERA, covered)))
    return records


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_classifier(
    esm_model: str, unfreeze_layers: int, device: str = "cpu", num_outputs: int = 1
):
    """Build ESM-2 (top N layers trainable) + classification head.

    ``unfreeze_layers=0`` = fully frozen (original behavior).
    ``unfreeze_layers=4`` = top 4 transformer layers + embeddings trainable.
    This is critical for small datasets where frozen 8M embeddings are too weak.
    ``num_outputs``: 1 for the binary head; ``len(PANEL_GENERA)`` for panel mode.
    Forward returns ``(B, num_outputs)`` logits — binary callers squeeze.
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
        def __init__(self, hidden_size: int, n_out: int):
            super().__init__()
            self.n_out = n_out
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.2)
            self.classifier = nn.Linear(hidden_size, n_out)

        def forward(self, input_ids, attention_mask):
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            x = self.drop(self.act(self.dense(pooled)))
            x = self.drop(self.act(self.dense(x)))
            return self.classifier(x)  # (B, n_out); binary callers squeeze(-1)

    model = ActivityClassifier(hidden, num_outputs)
    model.esm = esm
    model.to(device)
    return model, tokenizer, n_layers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(logits, labels):
    """Compute accuracy, AUROC, F1, precision, recall, average precision."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
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
    try:
        metrics["ap"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["ap"] = float(np.mean(labels))
    return metrics


def masked_bce(logits, targets, mask, pos_weight=None):
    """Mean BCE over unmasked entries only (train_reward.py hemolysis pattern).

    Genera without evidence for a sequence contribute zero loss — the panel
    labels are sparse by construction (a peptide is measured against a few
    organisms, not the whole panel).
    """
    import torch

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


def compute_panel_metrics(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    val_mask: np.ndarray,
    genera: list[str],
) -> dict:
    """Macro AUROC over genera that have both classes in the masked val set."""
    from sklearn.metrics import roc_auc_score

    aurocs: dict[str, float] = {}
    for i, genus in enumerate(genera):
        m = val_mask[:, i] > 0
        y = val_targets[m, i]
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            continue  # single-class genus: AUROC undefined
        try:
            aurocs[genus] = float(roc_auc_score(y, 1 / (1 + np.exp(-val_logits[m, i]))))
        except ValueError:
            continue
    macro = float(np.mean(list(aurocs.values()))) if aurocs else 0.5
    return {"macro_auroc": macro, "per_genus_auroc": aurocs}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature minimizing val BCE (grid + local refine). Pure numpy.

    Calibrated probabilities matter because the composite scorer thresholds and
    z-scores activity across candidates; a systematically overconfident head
    compresses the useful part of the score distribution.
    """

    def nll(t: float) -> float:
        p = 1 / (1 + np.exp(-logits / t))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    grid = np.concatenate(
        [
            np.linspace(0.25, 4.0, 76),
            1.0 / np.linspace(0.25, 1.0, 16),  # T < 1 side too
        ]
    )
    best = float(grid[np.argmin([nll(t) for t in grid])])
    for step in (0.05, 0.01):
        local = np.linspace(max(best - 10 * step, 0.1), best + 10 * step, 21)
        best = float(local[np.argmin([nll(t) for t in local])])
    return round(best, 3)


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
    ensemble_size: int = 1,
    calibrate: bool = True,
    panel: bool = False,
    split: str = "random",
    cluster_threshold: float = 0.7,
) -> None:
    """Train one or more classifier members; promote the best by val AUROC.

    Binary mode promotes ``classifier.pt``; panel mode (``panel=True``) trains
    the genus-level multi-hot head and promotes ``classifier_panel.pt`` with
    ``{"task": "panel", ...}`` in its config — the two artifacts never collide.

    With ``ensemble_size > 1`` each member trains under ``<out_dir>/member{i}/``
    with seed ``seed + i``; the winner's weights become the promoted artifact
    and a fitted temperature is stored in ``config.json`` (applied at inference
    by ``score.ActivityScorer`` / ``score.PanelScorer``). All members are
    summarized in ``members.json``.
    """
    import json
    import shutil

    from amp_challenge_2027.config import PANEL_GENERA

    artifact = "classifier_panel.pt" if panel else "classifier.pt"
    summary = []
    best: tuple[float, int, float] | None = None
    for member in range(ensemble_size):
        res = _train_single(
            data_path,
            esm_model=esm_model,
            unfreeze_layers=unfreeze_layers,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + member,
            device=device,
            save_dir=Path(out_dir) / f"member{member}",
            panel=panel,
            split=split,
            cluster_threshold=cluster_threshold,
        )
        metric_key = "best_macro_auroc" if panel else "best_auroc"
        temperature = (
            round(float(fit_temperature(res["val_logits"], res["val_labels"])), 3)
            if calibrate
            else 1.0
        )
        print(
            f"[reward] member {member}: best val AUROC {res[metric_key]:.3f} "
            f"(calibration T={temperature})"
        )
        summary.append(
            {
                "member": member,
                "seed": seed + member,
                "val_auroc": round(float(res[metric_key]), 4),
                "temperature": temperature,
            }
        )
        if best is None or res[metric_key] > best[0]:
            best = (res[metric_key], member, temperature)

    assert best is not None
    _, winner, temperature = best
    src = Path(out_dir) / f"member{winner}"
    for fname in (artifact, "config.json"):
        shutil.copy2(src / fname, Path(out_dir) / fname)
    cfg_path = Path(out_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature"] = temperature
    cfg["val_auroc"] = round(float(best[0]), 4)
    if panel:
        cfg["task"] = "panel"
        cfg["genera"] = list(PANEL_GENERA)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (Path(out_dir) / "members.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n[reward] promoted member {winner} "
        f"(val AUROC {best[0]:.3f}, T={temperature}) → {out_dir}"
    )


def greedy_identity_clusters(sequences: list[str], *, threshold: float = 0.7) -> dict[str, str]:
    """Greedy leader clustering by Levenshtein ratio ≥ ``threshold``.

    Returns a seq → cluster-leader map. Deterministic: sequences are visited
    in (length, lexicographic) order and join the first leader above the
    threshold. A length-difference prefilter skips impossible pairs cheaply
    (ratio ≤ 2·min(l₁,l₂)/(l₁+l₂), so widely different lengths can never hit
    the threshold even with a perfect prefix match).
    """
    import Levenshtein

    assign: dict[str, str] = {}
    leaders: list[str] = []
    for seq in sorted(set(sequences), key=lambda s: (len(s), s)):
        for leader in leaders:
            shorter, longer = (seq, leader) if len(seq) <= len(leader) else (leader, seq)
            if 2 * len(shorter) / (len(shorter) + len(longer)) < threshold:
                continue
            if Levenshtein.ratio(seq, leader) >= threshold:
                assign[seq] = leader
                break
        else:
            leaders.append(seq)
            assign[seq] = seq
    return assign


def cluster_split_records(
    records: list[dict], *, threshold: float = 0.7, seed: int = 42, train_frac: float = 0.8
) -> tuple[list[dict], list[dict], int]:
    """Split records by identity cluster so no homolog straddles train/val.

    A random sequence-level split inflates val AUROC whenever near-duplicate
    peptide variants (single-mutation families, common in DBAASP) land on
    both sides; this split is the honest generalization measure. Returns
    (train, val, n_clusters); both lists are seeded-shuffled.
    """
    assign = greedy_identity_clusters([r["sequence"] for r in records], threshold=threshold)
    reps = sorted(set(assign.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(reps)
    k = max(1, int(train_frac * len(reps)))
    train_reps = set(reps[:k])
    train = [r for r in records if assign[r["sequence"]] in train_reps]
    val = [r for r in records if assign[r["sequence"]] not in train_reps]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, len(reps)


def _train_single(
    data_path: Path,
    *,
    esm_model: str,
    unfreeze_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    save_dir: Path,
    panel: bool = False,
    split: str = "random",
    cluster_threshold: float = 0.7,
) -> dict:
    """Train one classifier member. Returns best-val-AUROC bookkeeping.

    Binary mode optimizes flat AUROC over ``{0,1}`` labels; panel mode
    optimizes macro AUROC over the ``PANEL_GENERA`` multi-hot targets with
    per-entry masking (masked BCE loss, train_reward.py hemolysis pattern).
    """
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.config import PANEL_GENERA
    from amp_challenge_2027.training import build_cosine_scheduler, enable_determinism

    enable_determinism(seed)

    if panel:
        records = load_panel_data(data_path)
    else:
        records = load_activity_data(data_path)
    if not records:
        print("[reward] no data; aborting", file=sys.stderr)
        sys.exit(1)

    # 80/20 split. Clustered: identity clusters (honest generalization — no
    # homolog on both sides). Random: binary stratified by class, panel plain
    # shuffle (multi-hot labels don't partition into two clean strata).
    rng = np.random.default_rng(seed)
    if split == "clustered":
        train_recs, val_recs, n_clusters = cluster_split_records(
            records, threshold=cluster_threshold, seed=seed
        )
        print(
            f"[reward] CLUSTERED split: {n_clusters} clusters "
            f"(threshold {cluster_threshold:g}) → train: {len(train_recs)}  val: {len(val_recs)}"
        )
        if not val_recs:
            print("[reward] clustered split left val empty; aborting", file=sys.stderr)
            sys.exit(1)
    elif panel:
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_train = int(0.8 * len(shuffled))
        train_recs, val_recs = shuffled[:n_train], shuffled[n_train:]
        print(f"[reward] train: {len(train_recs)}  val: {len(val_recs)}")
    else:
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

    n_outputs = len(PANEL_GENERA) if panel else 1

    # Class weights (inverse frequency). Binary: per dataset; panel: over the
    # flattened unmasked entries.
    if panel:
        all_t = np.concatenate([r["targets"][r["mask"] > 0] for r in train_recs])
        n_pos = float(all_t.sum())
        pos_weight = torch.tensor([float(len(all_t) - n_pos) / max(n_pos, 1.0)], device=device)
    else:
        n_active = sum(1 for r in train_recs if r["label"] == 1)
        n_inactive = len(train_recs) - n_active
        pos_weight = torch.tensor([n_inactive / max(n_active, 1)], device=device)
    print(f"[reward] class weight (pos): {pos_weight.item():.3f}")

    model, tokenizer, n_layers = build_classifier(
        esm_model, unfreeze_layers, device=device, num_outputs=n_outputs
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[reward] model: {n_trainable / 1e6:.2f}M trainable / {n_total / 1e6:.1f}M total "
        f"({esm_model}, {n_layers}L, unfrozen={unfreeze_layers}, outputs={n_outputs})"
    )

    # Single LR for all trainable params — works fine for small models.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = epochs * math.ceil(len(train_recs) / batch_size)
    scheduler = build_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=int(0.1 * total_steps)
    )

    best_auroc = 0.0
    best_val_logits: np.ndarray | None = None
    best_val_labels: np.ndarray | None = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        rng.shuffle(train_recs)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(train_recs), batch_size):
            batch = train_recs[start : start + batch_size]
            seqs = [r["sequence"] for r in batch]
            enc = tokenizer(
                seqs, return_tensors="pt", padding=True, truncation=True, max_length=52
            ).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"])
            if panel:
                targets = torch.stack([torch.from_numpy(r["targets"]).to(device) for r in batch])
                mask = torch.stack([torch.from_numpy(r["mask"]).to(device) for r in batch])
                loss = masked_bce(logits, targets, mask, pos_weight=pos_weight)
            else:
                labels = torch.tensor(
                    [r["label"] for r in batch], dtype=torch.float32, device=device
                )
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), labels, pos_weight=pos_weight
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
        all_masks = []
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_recs), batch_size):
                batch = val_recs[start : start + batch_size]
                seqs = [r["sequence"] for r in batch]
                enc = tokenizer(
                    seqs, return_tensors="pt", padding=True, truncation=True, max_length=52
                ).to(device)
                logits = model(enc["input_ids"], enc["attention_mask"])
                if panel:
                    targets = torch.stack(
                        [torch.from_numpy(r["targets"]).to(device) for r in batch]
                    )
                    mask = torch.stack([torch.from_numpy(r["mask"]).to(device) for r in batch])
                    loss = masked_bce(logits, targets, mask)
                    all_masks.append(mask.cpu().numpy())
                    all_labels.append(targets.cpu().numpy())
                else:
                    labels = torch.tensor(
                        [r["label"] for r in batch], dtype=torch.float32, device=device
                    )
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), labels
                    )
                    all_labels.append(labels.cpu().numpy())
                val_loss_total += loss.item()
                val_batches += 1
                all_logits.append(logits.cpu().numpy())

        val_logits = np.concatenate(all_logits)
        val_labels = np.concatenate(all_labels) if panel else np.array(all_labels)
        val_masks = np.concatenate(all_masks) if panel else None

        if panel:
            m = compute_panel_metrics(val_logits, val_labels, val_masks, PANEL_GENERA)
            score_val = m["macro_auroc"]
            shown = f"macro_auroc {m['macro_auroc']:.3f}"
        else:
            m = compute_metrics(val_logits.squeeze(-1), val_labels)
            score_val = m["auroc"]
            shown = f"auroc {m['auroc']:.3f}"
        avg_train = total_loss / max(n_batches, 1)
        avg_val = val_loss_total / max(val_batches, 1)

        improved = ""
        if score_val > best_auroc:
            best_auroc = score_val
            if panel:
                # Flatten unmasked entries for the temperature fit.
                flat_logits = val_logits[val_masks > 0]
                flat_labels = val_labels[val_masks > 0]
                best_val_logits, best_val_labels = flat_logits, flat_labels
            else:
                best_val_logits, best_val_labels = val_logits.squeeze(-1), val_labels
            _save_classifier(model, tokenizer, esm_model, unfreeze_layers, save_dir, panel=panel)
            improved = " ← saved (best)"

        if (epoch + 1) % 5 == 0 or improved:
            extra = f" genera={len(m['per_genus_auroc'])}/{len(PANEL_GENERA)}" if panel else ""
            print(
                f"[reward] epoch {epoch + 1}/{epochs} "
                f"train_loss {avg_train:.4f} val_loss {avg_val:.4f} "
                f"{shown}{extra}{improved}"
            )
        if panel and (epoch + 1) % 10 == 0 and m["per_genus_auroc"]:
            per = "  ".join(
                f"{g.split()[0]}:{a:.2f}" for g, a in sorted(m["per_genus_auroc"].items())
            )
            print(f"[reward]   per-genus AUROC: {per}")

    print(f"\n[reward] best AUROC: {best_auroc:.3f}")
    print(f"[reward] saved to {save_dir}")
    return {
        "best_auroc": best_auroc,
        "best_macro_auroc": best_auroc,
        "val_logits": best_val_logits if best_val_logits is not None else np.zeros(0),
        "val_labels": best_val_labels if best_val_labels is not None else np.zeros(0),
    }


def _save_classifier(
    model,
    tokenizer,
    esm_model: str,
    unfreeze_layers: int,
    out_dir: Path,
    panel: bool = False,
) -> None:
    """Save the classification HEAD only (ESM weights come from the hub).

    A full state dict includes the ESM-2 backbone (~140 MB for t12) which
    exceeds GitHub's 100 MB file limit. The backbone is rebuilt from
    ``esm_model`` at load time; only the small head is stored (~2 MB).
    Binary mode → ``classifier.pt``; panel mode → ``classifier_panel.pt``.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = "classifier_panel.pt" if panel else "classifier.pt"
    head_sd = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("esm.")
    }
    torch.save(head_sd, out_dir / artifact)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "esm_model": esm_model,
                "unfreeze_layers": unfreeze_layers,
                "type": "panel_activity_classifier" if panel else "binary_activity_classifier",
                "task": "panel" if panel else "binary",
                "checkpoint_format": "head-only",
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase-2 activity classifier (binary default, --panel for genus multi-hot)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="labels CSV; default activity_labels.csv (binary) or "
        "activity_labels_full.csv (--panel)",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="train the genus-level multi-hot panel head "
        "(loader: activity_labels_full.csv; artifact: classifier_panel.pt)",
    )
    parser.add_argument(
        "--split",
        choices=["random", "clustered"],
        default="random",
        help="val split: random (legacy; near-duplicate variants straddle "
        "train/val, optimistic AUROC) or clustered (Levenshtein identity "
        "clusters — the honest generalization measure)",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.7,
        help="Levenshtein ratio for identity clustering under --split clustered",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="facebook/esm2_t12_35M_UR50D",
        help="ESM-2 backbone (35M recommended for this data size)",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=4,
        help="Number of top ESM-2 layers to fine-tune (0=frozen)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=REWARD_DIR)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="train N members (seeds seed..seed+N-1); promote the best by val AUROC",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip temperature scaling on the validation set",
    )
    args = parser.parse_args()

    if args.data is None:
        args.data = (
            PROCESSED_DATA_DIR / "activity_labels_full.csv"
            if args.panel
            else PROCESSED_DATA_DIR / "activity_labels.csv"
        )
    if not args.data.exists():
        print(f"[reward] labels file missing: {args.data}", file=sys.stderr)
        sys.exit(1)
    if args.split == "clustered" and args.out_dir == REWARD_DIR:
        print(
            "[reward] refusing --split clustered into the DEPLOYED reward dir "
            f"({REWARD_DIR}) — promotion would overwrite classifier_panel.pt. "
            "Point --out-dir at an evaluation-only dir (e.g. checkpoint/reward_clustered).",
            file=sys.stderr,
        )
        sys.exit(1)

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
        ensemble_size=args.ensemble_size,
        calibrate=not args.no_calibrate,
        panel=args.panel,
        split=args.split,
        cluster_threshold=args.cluster_threshold,
    )


if __name__ == "__main__":
    main()
