"""Multi-objective candidate scoring: activity, conformity, precision proxy.

The Phase-1 protocol scores a library along several orthogonal axes. Ranking
candidates by the activity classifier alone over-fits one axis (seed44 showed
this: Recall 0.913 but Conformity 0.497). This module provides three
independent, deterministic component scorers and a weighted combiner so the
selection step can trade axes explicitly:

  activity    Phase-2 ESM-2 binary activity classifier probability
              (``checkpoint/reward/classifier.pt``; higher = likely active).

  conformity  Density quantile of each candidate inside the reference property
              distribution — a numpy port of seqme's ``ConformityScore`` idea:
              Gaussian KDE (silverman bandwidth) over per-peptide property
              vectors, score = fraction of reference points no denser than the
              candidate (higher = deeper inside the reference support). Uses the
              modlamp-exact Bjellqvist charge (``conditioning.charge_modlamp``)
              plus Kyte-Doolittle mean hydrophobicity and the Eisenberg helical
              moment from ``props.py``.

  precision   Mean cosine similarity to the k nearest reference ESM-2 mean-
              pooled embeddings — a cheap proxy for seqme's ``Precision``
              (higher = closer to the reference manifold in embedding space).
              Reference embeddings are computed once and cached under
              ``data/cache/``.

``CompositeScorer`` z-normalizes each available component across the current
batch and returns the weight-averaged sum. Components whose heavy dependencies
are missing are simply left out (weights renormalize), so the same call site
works on the minimal validator env and on the GPU training box.

All scorers are deterministic given their inputs; heavy imports (torch,
transformers) are deferred into methods.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from amp_challenge_2027.conditioning import charge_modlamp
from amp_challenge_2027.config import (
    DATA_DIR,
    ESM2_MODEL,
    MDR_PANEL_GENERA,
    REWARD_DIR,
    REWARD_HEMO_DIR,
)
from amp_challenge_2027.props import hydrophobic_moment, mean_hydrophobicity

EMBED_CACHE_DIR = DATA_DIR / "cache"
KNN_K = 5
_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Property space for the conformity component
# ---------------------------------------------------------------------------


def _property_matrix(sequences: list[str]) -> np.ndarray:
    """Per-sequence (modlamp charge, KD hydrophobicity, Eisenberg moment)."""
    rows = np.empty((len(sequences), 3), dtype=np.float64)
    for i, seq in enumerate(sequences):
        rows[i, 0] = charge_modlamp(seq)
        rows[i, 1] = mean_hydrophobicity(seq)
        rows[i, 2] = hydrophobic_moment(seq)
    return rows


def _robust_sd(x: np.ndarray) -> float:
    """min(std, IQR/1.349) — the Gaussian efficiency core of silverman's rule."""
    sd = float(x.std())
    q25, q75 = np.percentile(x, [25, 75])
    iqr_sd = float(q75 - q25) / 1.349
    cand = min(v for v in (sd, iqr_sd) if v > 0) if (sd > 0 or iqr_sd > 0) else 0.0
    return cand if cand > 0 else max(abs(float(x.mean())), 1e-6)


def _silverman_bandwidths(points: np.ndarray) -> np.ndarray:
    """Diagonal silverman bandwidths for a Gaussian product-kernel KDE."""
    n, d = points.shape
    factor = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
    h = np.array([factor * _robust_sd(points[:, j]) for j in range(d)])
    return np.maximum(h, 1e-6)


def _log_densities(
    queries: np.ndarray, ref: np.ndarray, h: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    """Log KDE density of each query row under the reference kernel mixture.

    Squared Mahalanobis distances are computed as ||q/h||² + ||r/h||² − 2(q/h)·(r/h)
    in float32 chunks so a (N × ~39k) problem stays memory-bounded.
    """
    n_ref = ref.shape[0]
    d = ref.shape[1]
    qh = (queries / h).astype(np.float32)
    rh = (ref / h).astype(np.float32)
    log_h_sum = float(np.sum(np.log(h)))
    const = -log_h_sum - 0.5 * d * _LOG_2PI - math.log(n_ref)

    out = np.empty(len(queries), dtype=np.float64)
    rh_sq = (rh * rh).sum(axis=1)  # (R,)
    for start in range(0, len(queries), chunk):
        qc = qh[start : start + chunk]
        d2 = ((qc * qc).sum(axis=1)[:, None] + rh_sq[None, :] - 2.0 * (qc @ rh.T)).astype(
            np.float32
        )
        np.maximum(d2, 0.0, out=d2)
        # logsumexp_j(-0.5*d2_j) with the min-distance shift for stability.
        m = d2.min(axis=1, keepdims=True)
        lse = np.log(np.exp(-0.5 * (d2 - m)).sum(axis=1))
        out[start : start + chunk] = -0.5 * m[:, 0] + lse + const
    return out


class ConformityScorer:
    """Reference-density quantile score; higher = more reference-typical."""

    name = "conformity"

    def __init__(self, reference_seqs: list[str], *, sample: int = 12000, seed: int = 42) -> None:
        if not reference_seqs:
            raise ValueError("ConformityScorer needs a non-empty reference set")
        if sample and len(reference_seqs) > sample:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(reference_seqs), size=sample, replace=False))
            reference_seqs = [reference_seqs[i] for i in idx]
        self._ref_props = _property_matrix(list(reference_seqs))
        self._h = _silverman_bandwidths(self._ref_props)
        self._ref_logden = _log_densities(self._ref_props, self._ref_props, self._h)

    def score(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.zeros(0, dtype=np.float32)
        q_logden = _log_densities(_property_matrix(sequences), self._ref_props, self._h)
        # Fraction of reference points at most as dense as the candidate.
        out = np.empty(len(sequences), dtype=np.float32)
        chunk = 1024
        for start in range(0, len(sequences), chunk):
            q = q_logden[start : start + chunk]
            out[start : start + chunk] = (q[:, None] >= self._ref_logden[None, :]).mean(axis=1)
        return out


# ---------------------------------------------------------------------------
# Activity component (Phase-2 ESM-2 binary classifier)
# ---------------------------------------------------------------------------


def _build_activity_module(num_outputs: int = 1):
    """Return (ActivityClassifier, torch, nn) with deferred imports.

    ``num_outputs``: 1 for the binary head (forward squeezes to (B,));
    ``len(PANEL_GENERA)`` for the panel head (forward returns (B, G)).
    """
    import torch
    from torch import nn

    class ActivityClassifier(nn.Module):
        """ESM-2 backbone + dense head; mirrors train_reward_classifier.py."""

        def __init__(self, hidden_size: int, n_out: int = 1):
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
            out = self.classifier(x)
            return out.squeeze(-1) if self.n_out == 1 else out

    return ActivityClassifier, torch, nn


class ActivityScorer:
    """Probability from the trained binary activity classifier (or None).

    If ``config.json`` carries a ``temperature`` (fitted on validation by
    ``train_reward_classifier.py --ensemble-size``), logits are divided by it
    before the sigmoid so probabilities are calibrated, not just ranked.
    """

    name = "activity"

    def __init__(self, model, tokenizer, device: str, temperature: float = 1.0) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._temperature = max(float(temperature), 1e-3)

    @classmethod
    def load(cls, *, device: str = "cpu") -> ActivityScorer | None:
        ckpt_path = REWARD_DIR / "classifier.pt"
        config_path = REWARD_DIR / "config.json"
        if not ckpt_path.exists() or not config_path.exists():
            return None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            config = json.loads(config_path.read_text())
            esm_id = config["esm_model"]
            ActivityClassifier, _, _ = _build_activity_module()
            tokenizer = AutoTokenizer.from_pretrained(esm_id)
            esm = AutoModel.from_pretrained(esm_id)
            model = ActivityClassifier(esm.config.hidden_size)
            model.esm = esm
            sd = torch.load(ckpt_path, map_location=device)
            if any(k.startswith("esm.") for k in sd):
                # Legacy full checkpoint (backbone included).
                model.load_state_dict(sd)
            else:
                # Head-only checkpoint (train_reward_classifier >= v2): backbone
                # comes from the hub above; verify nothing unexpected is present.
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if unexpected:
                    raise RuntimeError(f"unexpected head keys: {sorted(unexpected)[:4]}")
            model.to(device).eval()
            return cls(model, tokenizer, device, temperature=config.get("temperature", 1.0))
        except Exception as e:
            print(f"[score] activity classifier unavailable ({e}); dropping component")
            return None

    def score(self, sequences: list[str]) -> np.ndarray:
        import torch

        probs = np.empty(len(sequences), dtype=np.float32)
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                enc = self._tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=52
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                probs[start : start + batch_size] = (
                    torch.sigmoid(logits / self._temperature).cpu().numpy()
                )
        return probs


# ---------------------------------------------------------------------------
# Panel breadth (genus-level multi-hot activity, MDR-weighted variant)
# ---------------------------------------------------------------------------


class PanelScorer:
    """Activity-breadth components from the genus-level panel classifier.

    Two composite-scorer components from ONE batched forward pass:
      breadth      fraction of panel genera with calibrated p(active) > 0.5
      mdr_breadth  same over the genera containing an MDR panel strain

    Loads ``checkpoint/reward/classifier_panel.pt`` (trained by
    ``train_reward_classifier.py --panel``). ``load()`` returns None when the
    artifact is absent so composite wiring simply drops the components — the
    validator environment and pre-training runs keep working unchanged.
    """

    name = "breadth"

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        genera: list[str],
        mdr_genera: frozenset[str],
        temperature: float = 1.0,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self.genera = list(genera)
        self.mdr_genera = frozenset(mdr_genera)
        self._temperature = max(float(temperature), 1e-3)

    @classmethod
    def load(
        cls, *, device: str = "cpu", checkpoint_dir: Path | str | None = None
    ) -> PanelScorer | None:
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else REWARD_DIR
        ckpt_path = ckpt_dir / "classifier_panel.pt"
        config_path = ckpt_dir / "config.json"
        if not ckpt_path.exists() or not config_path.exists():
            return None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            config = json.loads(config_path.read_text())
            if config.get("task") != "panel":
                raise RuntimeError("config task != 'panel'")
            genera = list(config["genera"])
            PanelClassifier, _, _ = _build_activity_module(num_outputs=len(genera))
            tokenizer = AutoTokenizer.from_pretrained(config["esm_model"])
            esm = AutoModel.from_pretrained(config["esm_model"])
            model = PanelClassifier(esm.config.hidden_size, len(genera))
            model.esm = esm
            sd = torch.load(ckpt_path, map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected head keys: {sorted(unexpected)[:4]}")
            model.to(device).eval()
            mdr = frozenset(MDR_PANEL_GENERA) & set(genera)
            return cls(
                model,
                tokenizer,
                device,
                genera,
                mdr,
                temperature=config.get("temperature", 1.0),
            )
        except Exception as e:
            print(f"[score] panel classifier unavailable ({e}); dropping component")
            return None

    def _probs(self, sequences: list[str]) -> np.ndarray:
        """Calibrated per-genus probabilities, shape (N, G)."""
        import torch

        out = np.empty((len(sequences), len(self.genera)), dtype=np.float32)
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                enc = self._tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=52
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                out[start : start + batch_size] = (
                    torch.sigmoid(logits / self._temperature).cpu().numpy()
                )
        return out

    def breadth(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.zeros(0, dtype=np.float32)
        return (self._probs(sequences) > 0.5).mean(axis=1).astype(np.float32)

    def mdr_breadth(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.zeros(0, dtype=np.float32)
        idx = [i for i, genus in enumerate(self.genera) if genus in self.mdr_genera]
        if not idx:
            return np.zeros(len(sequences), dtype=np.float32)
        return (self._probs(sequences)[:, idx] > 0.5).mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Hemolysis safety (Phase-2 risk control)
# ---------------------------------------------------------------------------


class HemoScorer:
    """Predicted hemolysis risk from the HC50 head → SAFETY component.

    Direction convention: the head is trained by the generic binary trainer on
    ``hemolysis_labels.csv`` where label "active" == RISKY (min HC50 ≤ the
    ceiling), so the raw sigmoid output is p(risky). The composite-scorer
    component is ``safety = 1 - p(risky)`` (higher = safer), so a positive
    ``--w-safety`` weight penalizes hemolytic candidates like every other
    higher-is-better component.

    Loads ``classifier.pt`` + ``config.json`` from ``REWARD_HEMO_DIR``
    (``checkpoint/reward_hemo/``, a separate dir so training the hemo head can
    never touch the activity artifacts). ``load()`` returns None when absent —
    the component simply drops, exactly like the other optional scorers.
    """

    name = "safety"

    def __init__(self, model, tokenizer, device: str, temperature: float = 1.0) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._temperature = max(float(temperature), 1e-3)

    @classmethod
    def load(
        cls, *, device: str = "cpu", checkpoint_dir: Path | str | None = None
    ) -> HemoScorer | None:
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else REWARD_HEMO_DIR
        ckpt_path = ckpt_dir / "classifier.pt"
        config_path = ckpt_dir / "config.json"
        if not ckpt_path.exists() or not config_path.exists():
            return None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            config = json.loads(config_path.read_text())
            BinaryClassifier, _, _ = _build_activity_module(num_outputs=1)
            tokenizer = AutoTokenizer.from_pretrained(config["esm_model"])
            esm = AutoModel.from_pretrained(config["esm_model"])
            model = BinaryClassifier(esm.config.hidden_size, 1)
            model.esm = esm
            sd = torch.load(ckpt_path, map_location=device)
            if any(k.startswith("esm.") for k in sd):
                model.load_state_dict(sd)
            else:
                _, unexpected = model.load_state_dict(sd, strict=False)
                if unexpected:
                    raise RuntimeError(f"unexpected head keys: {sorted(unexpected)[:4]}")
            model.to(device).eval()
            return cls(model, tokenizer, device, temperature=config.get("temperature", 1.0))
        except Exception as e:
            print(f"[score] hemolysis scorer unavailable ({e}); dropping component")
            return None

    def p_risky(self, sequences: list[str]) -> np.ndarray:
        """Calibrated p(hemolytic at ≤ ceiling), shape (N,). Higher = riskier."""
        import torch

        out = np.empty(len(sequences), dtype=np.float32)
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                enc = self._tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=52
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                out[start : start + batch_size] = (
                    torch.sigmoid(logits / self._temperature).cpu().numpy()
                )
        return out

    def score(self, sequences: list[str]) -> np.ndarray:
        """SAFETY = 1 - p(risky). Higher = safer (composite convention)."""
        return (1.0 - self.p_risky(sequences)).astype(np.float32)


# ---------------------------------------------------------------------------
# Precision proxy (kNN similarity in ESM-2 embedding space)
# ---------------------------------------------------------------------------


def _embedder_cache_key(esm_model: str, reference_seqs: list[str]) -> str:
    digest = hashlib.sha1()
    digest.update(esm_model.encode())
    digest.update(b"\0")
    for seq in sorted(reference_seqs):
        digest.update(seq.encode())
        digest.update(b";")
    return digest.hexdigest()[:16]


class PrecisionProxyScorer:
    """Mean top-k cosine similarity to reference ESM-2 embeddings."""

    name = "precision"

    def __init__(
        self,
        reference_seqs: list[str],
        *,
        esm_model: str = ESM2_MODEL,
        device: str = "cpu",
        k: int = KNN_K,
        cache: bool = True,
    ) -> None:
        if not reference_seqs:
            raise ValueError("PrecisionProxyScorer needs a non-empty reference set")
        self._esm_model = esm_model
        self._device = device
        self._k = k
        self._reference_seqs = list(reference_seqs)
        self._ref_emb: np.ndarray | None = None
        self._tokenizer = None
        self._model = None
        self._cache_path = (
            EMBED_CACHE_DIR / f"ref_embeddings_{_embedder_cache_key(esm_model, reference_seqs)}.npy"
        )
        if cache and self._cache_path.exists():
            try:
                emb = np.load(self._cache_path)
                if emb.shape[0] == len(set(reference_seqs)):
                    self._ref_emb = emb
                    print(
                        f"[score] loaded cached reference embeddings {emb.shape} ← {self._cache_path.name}"
                    )
            except Exception as e:
                print(f"[score] ignoring corrupt embedding cache ({e})")

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._esm_model)
        model = AutoModel.from_pretrained(self._esm_model).to(self._device).eval()
        self._torch = torch
        self._model = model

    def _embed(self, sequences: list[str]) -> np.ndarray:
        self._ensure_model()
        torch = self._torch
        out = np.empty((len(sequences), self._model.config.hidden_size), dtype=np.float32)
        batch_size = 128
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                enc = self._tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=52
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                res = self._model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (res.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                out[start : start + batch_size] = pooled.cpu().numpy()
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-8)

    def _reference_embeddings(self) -> np.ndarray:
        if self._ref_emb is not None:
            return self._ref_emb
        unique = sorted(set(self._reference_seqs))
        emb = self._embed(unique)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._cache_path, emb)
        print(f"[score] cached {emb.shape} reference embeddings → {self._cache_path.name}")
        self._ref_emb = emb
        return emb

    def score(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.zeros(0, dtype=np.float32)
        ref = self._reference_embeddings()
        q = self._embed(sequences)
        k = min(self._k, ref.shape[0])
        sims = np.empty(len(sequences), dtype=np.float32)
        chunk = 2048
        for start in range(0, len(sequences), chunk):
            block = q[start : start + chunk] @ ref.T
            sims[start : start + chunk] = np.sort(block, axis=1)[:, -k:].mean(axis=1)
        return sims


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

ComponentFn = Callable[[list[str]], np.ndarray]


class CompositeScorer:
    """Weighted z-score combination of available component scorers."""

    def __init__(self, components: list[tuple[str, float, ComponentFn]]) -> None:
        if not components:
            raise ValueError("CompositeScorer needs at least one component")
        self.components = components

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.components]

    def score(self, sequences: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        parts: dict[str, np.ndarray] = {}
        total_w = 0.0
        combined = np.zeros(len(sequences), dtype=np.float64)
        for name, weight, fn in self.components:
            vals = np.asarray(fn(sequences), dtype=np.float64)
            parts[name] = vals.astype(np.float32)
            sd = vals.std()
            z = (vals - vals.mean()) / (sd if sd > 1e-8 else 1.0)
            combined += weight * z
            total_w += weight
        return (combined / total_w).astype(np.float32), parts


__all__ = [
    "ConformityScorer",
    "ActivityScorer",
    "PanelScorer",
    "HemoScorer",
    "PrecisionProxyScorer",
    "CompositeScorer",
    "_property_matrix",
    "_silverman_bandwidths",
    "_log_densities",
]
