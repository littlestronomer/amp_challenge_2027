"""Reward surrogates: ESM-2 LoRA models for MIC and hemolysis prediction.

These models serve two purposes:
  1. As the **RL reward signal** during generator fine-tuning (activity reward,
     hemolysis penalty → the safety window HC50/MIC50).
  2. As the **ranker** for selecting the top-100 candidates.

Architecture: frozen ESM-2 encoder + LoRA adapters (q,v) + a small multi-task
head. The head predicts per-strain MIC (regression, log-µM) and a hemolysis
risk. Using an ensemble (3 models, different seeds) reduces any single model's
bias — the same principle the organizers apply with their Phase-1 surrogate
ensemble.

This module is **optional at inference time**: ``generate.py`` tries to load a
trained ensemble from ``checkpoint/reward/``; if absent, it falls back to a
deterministic property-based scorer so the entry point still runs end-to-end.

Heavy imports (torch, transformers, peft) are deferred to the functions that
need them, so importing this module is cheap and always succeeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import (
    ESM2_LORA_ALPHA,
    ESM2_LORA_DROPOUT,
    ESM2_LORA_RANK,
    ESM2_MODEL,
    NUM_STRAINS,
    REWARD_DIR,
)

# ---------------------------------------------------------------------------
# Reward API (used by generate.py and the RL trainer)
# ---------------------------------------------------------------------------


@dataclass
class RewardOutput:
    """Per-sequence reward components."""

    activity: float  # mean predicted activity across strains (higher = better)
    hemolysis_penalty: float  # predicted hemolysis risk (lower = safer)
    safety_window: float  # activity - hemolysis (higher = better)
    per_strain: np.ndarray | None  # optional per-strain activity, shape (NUM_STRAINS,)

    @property
    def score(self) -> float:
        """Combined ranking score: activity with a hemolysis penalty."""
        return float(self.activity - self.hemolysis_penalty)


class RewardEnsemble:
    """Ensemble of ESM-2 LoRA reward models + a deterministic fallback scorer.

    Call ``RewardEnsemble.load_or_fallback()`` from the inference path. If no
    trained ensemble is present, you get a property-based fallback so the
    pipeline always produces a ranking.
    """

    def __init__(self, models: list | None = None, *, device: str = "cpu"):
        self.models = models  # list of trained ESMMICReward modules
        self.device = device

    @property
    def is_trained(self) -> bool:
        return self.models is not None and len(self.models) > 0

    def score_batch(self, sequences: list[str]) -> list[RewardOutput]:
        if self.is_trained:
            return self._score_trained(sequences)
        return self._score_fallback(sequences)

    # --- trained path -------------------------------------------------------

    def _score_trained(self, sequences: list[str]) -> list[RewardOutput]:
        """Average predictions across the ensemble.

        Each model returns per-strain log-MIC (lower = more potent) and a
        hemolysis probability. We convert to "activity" = -mean(log MIC) so
        higher is better, consistent with the fallback.
        """
        per_strain_acc = np.zeros((len(sequences), NUM_STRAINS), dtype=np.float32)
        hemo = np.zeros(len(sequences), dtype=np.float32)
        for model in self.models:
            ps, hm = _predict_one_model(model, sequences, device=self.device)
            per_strain_acc += ps
            hemo += hm
        per_strain_acc /= max(len(self.models), 1)
        hemo /= max(len(self.models), 1)

        out: list[RewardOutput] = []
        for i in range(len(sequences)):
            activity = float(np.mean(per_strain_acc[i]))
            hemolysis = float(hemo[i])
            out.append(
                RewardOutput(
                    activity=activity,
                    hemolysis_penalty=hemolysis,
                    safety_window=activity - hemolysis,
                    per_strain=per_strain_acc[i],
                )
            )
        return out

    # --- fallback path ------------------------------------------------------

    def _score_fallback(self, sequences: list[str]) -> list[RewardOutput]:
        """Deterministic property-based scorer (no torch, no ESM-2).

        Approximates "AMP-likeness" from charge, amphipathicity, and
        hydrophobicity using the biophysical rules in ``props.py``. Not a
        learned model — purely for keeping the pipeline runnable before the
        reward models are trained.
        """
        from amp_challenge_2027.props import compute_properties

        out: list[RewardOutput] = []
        for seq in sequences:
            p = compute_properties(seq)
            # Cationic + amphipathic peptides tend to be active; extreme
            # hydrophobicity tends to be hemolytic.
            activity = 0.5 * min(max(p.charge, 0.0), 8.0) / 8.0
            activity += 0.5 * min(p.hydrophobic_moment, 0.5) / 0.5
            hemo_risk = max(0.0, (p.hydrophobicity_kd + 1.0) / 2.5)  # ~0..1
            hemo_risk = min(hemo_risk, 1.0)
            out.append(
                RewardOutput(
                    activity=float(activity),
                    hemolysis_penalty=float(hemo_risk * 0.5),
                    safety_window=float(activity - hemo_risk * 0.5),
                    per_strain=None,
                )
            )
        return out

    # --- loading -----------------------------------------------------------

    @classmethod
    def load_or_fallback(
        cls, checkpoint_dir: Path | str | None = None, *, device: str = "cpu"
    ) -> RewardEnsemble:
        """Load a trained ensemble if present; otherwise return the fallback scorer.

        ``generate.py`` calls this so it works both before and after reward
        training. The fallback is deterministic and torch-free.
        """
        ckpt_dir = Path(checkpoint_dir or REWARD_DIR)
        if not ckpt_dir.exists() or not any(ckpt_dir.glob("model_*")):
            return cls(models=None, device=device)
        try:
            return _load_ensemble(ckpt_dir, device=device)
        except Exception as e:  # pragma: no cover - defensive at inference time
            print(f"[reward] could not load trained ensemble ({e}); using fallback scorer")
            return cls(models=None, device=device)


# ---------------------------------------------------------------------------
# Model definition (module-level so build + load share one definition)
# ---------------------------------------------------------------------------


def _build_modules():
    """Construct and return (RewardHead, ESMMICReward) nn.Module classes.

    Deferred so torch is only imported when actually building a model.
    """
    import torch
    from torch import nn

    class RewardHead(nn.Module):
        """Multi-task head: per-strain MIC regression + hemolysis classification."""

        def __init__(self, hidden_size: int, num_strains: int):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(0.1)
            self.mic_head = nn.Linear(hidden_size, num_strains)  # log-MIC per strain
            self.hemo_head = nn.Linear(hidden_size, 1)  # hemolysis logit

        def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.dropout(self.act(self.dense(pooled)))
            return self.mic_head(x), self.hemo_head(x).squeeze(-1)

    class ESMMICReward(nn.Module):
        """Frozen ESM-2 (with LoRA) + reward head."""

        def __init__(self, esm, head: RewardHead):
            super().__init__()
            self.esm = esm
            self.head = head

        def forward(self, input_ids, attention_mask):
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (B, L, H)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.head(pooled)

    return RewardHead, ESMMICReward


def build_reward_model(device: str = "cpu"):
    """Construct one ESM-2 LoRA reward model. Returns (model, tokenizer).

    The base ESM-2 is frozen; LoRA adapters on q/v make only ~1% of params
    trainable. The head predicts NUM_STRAINS MIC values (regression) + one
    hemolysis logit.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    RewardHead, ESMMICReward = _build_modules()

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    esm = AutoModel.from_pretrained(ESM2_MODEL)
    hidden = esm.config.hidden_size

    for p in esm.parameters():
        p.requires_grad = False
    lora_cfg = LoraConfig(
        r=ESM2_LORA_RANK,
        lora_alpha=ESM2_LORA_ALPHA,
        lora_dropout=ESM2_LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    esm = get_peft_model(esm, lora_cfg)
    model = ESMMICReward(esm, RewardHead(hidden, NUM_STRAINS))
    model.to(device)
    return model, tokenizer


def _predict_one_model(model, sequences: list[str], *, device: str = "cpu"):
    """Run one trained model over a batch; return (activity[N,S], hemolysis[N])."""
    import torch

    model.eval()
    tokenizer = _get_tokenizer()
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=52,
    ).to(device)
    with torch.no_grad():
        mic_logits, hemo_logits = model(enc["input_ids"], enc["attention_mask"])
    # mic_logits: (N, S) in log-µM. Lower = more potent. Convert to activity.
    activity = (-mic_logits.float().cpu().numpy())  # higher = better
    hemolysis = torch.sigmoid(hemo_logits).float().cpu().numpy()
    return activity, hemolysis


_TOKENIZER_CACHE: object | None = None


def _get_tokenizer():
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(ESM2_MODEL)
    return _TOKENIZER_CACHE


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_reward_model(model, path: Path | str) -> None:
    """Save a trained reward model: LoRA adapters + head weights + config."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if hasattr(model.esm, "save_pretrained"):
        model.esm.save_pretrained(p / "esm_lora")  # type: ignore[arg-type]
    import torch

    torch.save(model.head.state_dict(), p / "head.pt")
    (p / "config.json").write_text(
        json.dumps({"esm_model": ESM2_MODEL, "num_strains": NUM_STRAINS})
    )


def _load_ensemble(checkpoint_dir: Path | str, *, device: str = "cpu") -> RewardEnsemble:
    """Load all ``model_*`` subdirectories under ``checkpoint_dir``."""
    ckpt = Path(checkpoint_dir)
    model_dirs = sorted(ckpt.glob("model_*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model_* directories under {ckpt}")
    models = []
    for md in model_dirs:
        try:
            models.append(_load_one(md, device=device))
        except Exception as e:
            print(f"[reward] skipping {md.name}: {e}")
    if not models:
        raise RuntimeError("No reward models could be loaded")
    return RewardEnsemble(models=models, device=device)


def _load_one(model_dir: Path, *, device: str = "cpu"):
    """Load a single reward model: rebuild base + merge LoRA + load head."""
    import torch
    from peft import PeftModel
    from transformers import AutoModel

    RewardHead, ESMMICReward = _build_modules()
    cfg = json.loads((model_dir / "config.json").read_text())
    hidden = AutoModel.from_pretrained(cfg["esm_model"]).config.hidden_size

    head = RewardHead(hidden, cfg["num_strains"])
    head.load_state_dict(torch.load(model_dir / "head.pt", map_location=device))

    esm = AutoModel.from_pretrained(cfg["esm_model"])
    esm = PeftModel.from_pretrained(esm, model_dir / "esm_lora")

    model = ESMMICReward(esm, head)
    model.to(device)
    model.eval()
    return model


__all__ = [
    "RewardOutput",
    "RewardEnsemble",
    "build_reward_model",
    "save_reward_model",
]
