"""Frozen-feature predictor training; no dataset splitting or test access here."""

from __future__ import annotations

import numpy as np

from amp_challenge_2027.generalization import macro_rank


def build_head(hidden: int, outputs: int, architecture: str):
    from torch import nn

    if architecture not in {"linear", "mlp"}:
        raise ValueError("Unknown head architecture")

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            if architecture == "mlp":
                self.dense = nn.Linear(hidden, hidden)
                self.act = nn.GELU()
                self.drop = nn.Dropout(.2)
            self.classifier = nn.Linear(hidden, outputs)

        def forward(self, pooled):
            if architecture == "mlp":
                # Exact production head: the SAME dense layer is applied twice.
                pooled = self.drop(self.act(self.dense(pooled)))
                pooled = self.drop(self.act(self.dense(pooled)))
            return self.classifier(pooled)

    return Head()


def fit_head(train_x, train_y, train_mask, val_x, val_y, val_mask, *, protocol, seed, architecture, device):
    """Select epochs on validation macro AUROC; neither calibration nor test is accepted."""
    import torch

    from amp_challenge_2027.training import enable_determinism

    enable_determinism(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if (not len(train_x) or not len(val_x) or train_y.shape != train_mask.shape
            or val_y.shape != val_mask.shape or train_x.shape[0] != train_y.shape[0]
            or val_x.shape[0] != val_y.shape[0]):
        raise ValueError("Invalid train/validation arrays")
    pos = (train_y * train_mask).sum(0)
    neg = ((1 - train_y) * train_mask).sum(0)
    if (pos == 0).any() or (neg == 0).any():
        raise ValueError("Training outputs need both classes")
    if macro_rank(np.zeros_like(val_y), val_y, val_mask) is None:
        raise ValueError("No defined validation AUROC")
    model = build_head(train_x.shape[1], train_y.shape[1], architecture).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    tx, ty, tm, vx = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (train_x, train_y, train_mask, val_x)]
    weight = torch.as_tensor(neg / pos, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best_score, best_state, best_logits = -np.inf, None, None
    history, stale = [], 0
    for epoch in range(1, protocol["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train_x))
        loss_sum = observed = 0.
        for start in range(0, len(order), protocol["head_batch_size"]):
            ids = torch.as_tensor(order[start:start + protocol["head_batch_size"]], device=device)
            raw = model(tx[ids])
            count = tm[ids].sum()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(raw, ty[ids], pos_weight=weight,
                                                                         reduction="none") * tm[ids]).sum() / count.clamp(min=1)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(count)
            observed += float(count)
        model.eval()
        with torch.inference_mode():
            logits = model(vx).cpu().numpy()
        score = macro_rank(logits, val_y, val_mask)
        history.append({"epoch": epoch, "train_weighted_bce": loss_sum / observed, "validation_macro_auroc": score})
        if score > best_score:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_logits = logits.copy()
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[generalization] {architecture}/seed{seed} epoch{epoch}: validation AUROC={score:.4f}", flush=True)
        if stale >= protocol["patience"]:
            break
    return best_state, history, best_logits


def predict_head(state, features, outputs, architecture, *, device):
    import torch

    model = build_head(features.shape[1], outputs, architecture).float().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        result = model(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite head predictions")
    return result


class FrozenEncoder:
    """Pinned FP32 frozen ESM mean pooling, including special tokens as deployed."""

    def __init__(self, model_id, revision, *, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from amp_challenge_2027.training import enable_determinism

        enable_determinism(42)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).float().to(device).eval()
        if self.model.config.hidden_size != 480 or self.model.config._commit_hash != revision:
            raise ValueError("Backbone commit or embedding dimension differs")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.info = {"model": model_id, "revision": revision, "hidden_size": 480,
                     "dtype": "float32", "max_length": 52, "pooling": "masked mean incl BOS/EOS",
                     "mode": "frozen eval; no feature fitting, normalization or augmentation"}

    def encode(self, sequences, batch_size):
        import torch

        result = np.empty((len(sequences), 480), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start:start + batch_size]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
                if enc["input_ids"].shape[1] > 52:
                    raise ValueError("Sequence would be truncated; dataset and tokenization disagree")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.)
                result[start:start + len(batch)] = pooled.cpu().numpy()
                if start == 0 or start // batch_size % 25 == 0:
                    print(f"[generalization] embedded {min(start + batch_size, len(sequences))}/{len(sequences)}", flush=True)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen embeddings")
        return result
