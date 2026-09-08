"""Pure reconstruction/measurement helpers; no retraining or claim of a new holdout."""

from __future__ import annotations

import math

import numpy as np


def reconstruct_split(records: list[dict], *, panel: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror _train_single's RANDOM split, retaining loaded row order/duplicates.

    This only reconstructs membership under the stated algorithm and current
    records. A recorded model seed alone does not establish the original split.
    """
    rng = np.random.default_rng(seed)
    if panel:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        cut = int(.8 * len(indices))
        train, val = indices[:cut], indices[cut:]
    else:
        positive = [i for i, row in enumerate(records) if row["label"] == 1]
        negative = [i for i, row in enumerate(records) if row["label"] == 0]
        if len(positive) + len(negative) != len(records):
            raise ValueError("Binary reconstruction requires 0/1 labels")
        rng.shuffle(positive)
        rng.shuffle(negative)
        np_, nn = int(.8 * len(positive)), int(.8 * len(negative))
        train, val = positive[:np_] + negative[:nn], positive[np_:] + negative[nn:]
        rng.shuffle(train)
        rng.shuffle(val)
    if not train or not val:
        raise ValueError("Reconstructed train/validation split is empty")
    return np.array(train, dtype=int), np.array(val, dtype=int)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite logits")
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))


def discrimination(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Tie-aware AUROC/AP. Empty or single-class strata have no rank metric."""
    y, scores = np.asarray(labels), np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid discrimination inputs")
    n, positive = len(y), int(y.sum())
    if positive == 0 or positive == n:
        return {"auroc": None, "average_precision": None,
                "rank_metric_status": "empty" if not n else "single_class"}
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    auc = (ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * (n - positive))
    order = np.argsort(-scores, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(scores[order])), n - 1]
    tp = np.cumsum(y[order])[end]
    ap = np.sum(np.diff(np.r_[0, tp]) / positive * tp / (end + 1))
    return {"auroc": float(auc), "average_precision": float(ap), "rank_metric_status": "defined"}


def reliability(labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> list[dict]:
    """Equal-width bins; Wilson intervals are descriptive iid-row intervals only."""
    y, p = np.asarray(labels), np.asarray(probabilities)
    if bins <= 0 or y.shape != p.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid reliability inputs")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probabilities")
    assignment = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = assignment == i
        n = int(mask.sum())
        prevalence = float(y[mask].mean()) if n else None
        lower = upper = None
        if n:
            z = 1.959963984540054
            denominator = 1 + z * z / n
            center = (prevalence + z * z / (2 * n)) / denominator
            radius = z * math.sqrt(prevalence * (1 - prevalence) / n + z * z / (4 * n * n)) / denominator
            lower, upper = max(0., center - radius), min(1., center + radius)
        rows.append({"bin": i, "lower": i / bins, "upper": (i + 1) / bins, "n": n,
                     "mean_probability": float(p[mask].mean()) if n else None, "positive_fraction": prevalence,
                     "iid_row_wilson95_lower": lower, "iid_row_wilson95_upper": upper, "low_support": n < 30})
    return rows


def binary_report(labels: np.ndarray, logits: np.ndarray, *, temperature: float,
                  training_labels: np.ndarray) -> tuple[dict, list[dict]]:
    y, logits, train = np.asarray(labels), np.asarray(logits), np.asarray(training_labels)
    rank = discrimination(y, logits)
    if train.ndim != 1 or not np.isin(train, [0, 1]).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid training labels or temperature")
    z = logits.astype(np.float64) / temperature
    p = sigmoid(z)
    bins = reliability(y, p)
    n = len(y)
    train_p = float(train.mean()) if len(train) else None
    base_loss = base_brier = None
    if n and train_p is not None:
        base_brier = float(np.mean((y - train_p) ** 2))
        clipped = np.clip(train_p, 1e-7, 1 - 1e-7)
        base_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    ece = sum(row["n"] * abs(row["mean_probability"] - row["positive_fraction"])
              for row in bins if row["n"]) / n if n else None
    return {**rank, "n": n, "positives": int(y.sum()), "negatives": n - int(y.sum()),
            "temperature": temperature, "validation_prevalence": float(y.mean()) if n else None,
            "training_prevalence": train_p, "mean_probability": float(p.mean()) if n else None,
            "brier": float(np.mean((p - y) ** 2)) if n else None,
            "log_loss": float(np.mean(np.logaddexp(0, z) - y * z)) if n else None,
            "ece_10_equal_width": ece, "training_prevalence_baseline_brier": base_brier,
            "training_prevalence_baseline_log_loss": base_loss}, bins


def prediction_report(logits: np.ndarray, targets: np.ndarray, masks: np.ndarray,
                      training_targets: np.ndarray, training_masks: np.ndarray,
                      *, temperature: float, outputs: list[str]) -> tuple[dict, list[dict], list[dict]]:
    for values, mask in ((targets, masks), (training_targets, training_masks)):
        if (values.ndim != 2 or values.shape != mask.shape or values.shape[1] != len(outputs)
                or not np.isin(values, [0, 1]).all() or not np.isin(mask, [0, 1]).all()):
            raise ValueError("Invalid target/mask schema")
    if logits.shape != targets.shape or not np.isfinite(logits).all():
        raise ValueError("Invalid prediction array")
    rows, bins = [], []
    for mode, temp in (("uncalibrated", 1.), ("stored_temperature", temperature)):
        for i, name in enumerate(outputs):
            keep, train_keep = masks[:, i] == 1, training_masks[:, i] == 1
            row, calibration = binary_report(targets[keep, i], logits[keep, i], temperature=temp,
                                             training_labels=training_targets[train_keep, i])
            rows.append({"mode": mode, "output": name, **row})
            bins.extend({"mode": mode, "output": name, **item} for item in calibration)
    summary = {}
    for mode in ("uncalibrated", "stored_temperature"):
        selected = [row for row in rows if row["mode"] == mode]
        metrics = {}
        for key in ("auroc", "average_precision", "brier", "log_loss", "ece_10_equal_width",
                    "training_prevalence_baseline_brier", "training_prevalence_baseline_log_loss"):
            available = [row[key] for row in selected if row[key] is not None]
            metrics[f"macro_{key}"] = float(np.mean(available)) if available else None
            metrics[f"outputs_with_{key}"] = len(available)
        summary[mode] = metrics
    return summary, rows, bins


def cross_split_similarity(training_sequences: list[str], validation_sequences: list[str]) -> tuple[dict, list[float]]:
    from Levenshtein import ratio

    train = sorted(set(training_sequences))
    if not train or not validation_sequences:
        raise ValueError("Cannot audit empty splits")
    nearest = {seq: max(ratio(seq, other) for other in train) for seq in sorted(set(validation_sequences))}
    values = [nearest[seq] for seq in validation_sequences]
    train_set = set(train)
    return {"train_rows": len(training_sequences), "train_unique_sequences": len(train),
            "validation_rows": len(values), "validation_unique_sequences": len(nearest),
            "validation_duplicate_rows": len(values) - len(nearest),
            "validation_rows_with_exact_training_sequence": sum(seq in train_set for seq in validation_sequences),
            "validation_rows_with_nearest_ratio_gt_0_7": sum(v > .7 for v in values),
            "validation_rows_with_nearest_ratio_gt_0_8": sum(v > .8 for v in values),
            "mean_nearest_training_ratio": float(np.mean(values)),
            "scope": "CURRENT reconstructed split only; not a historical leakage or independence certificate"}, values
