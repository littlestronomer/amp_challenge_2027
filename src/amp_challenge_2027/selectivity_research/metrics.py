from __future__ import annotations

import numpy as np
from amp_challenge_2027.config import MDR_PANEL_GENERA

from .contracts import validate_solution


def summarize(frame, indices: np.ndarray, genera: list[str], *, incumbent: np.ndarray | None = None) -> dict:
    validate_solution(indices, len(frame), len(indices))
    panel = frame[[f"panel:{g}" for g in genera]].to_numpy(dtype=float)[indices]
    mdr = [j for j, g in enumerate(genera) if g in MDR_PANEL_GENERA]
    seqs = frame.iloc[indices]["sequence"].tolist()
    from Levenshtein import ratio
    distances = [1 - ratio(a, b) for i, a in enumerate(seqs) for b in seqs[:i]]
    out = {
        "n": len(indices), "activity_mean": float(frame.iloc[indices]["activity"].mean()),
        "risk_mean": float(frame.iloc[indices]["hemolysis_risk"].mean()),
        "risk_median": float(frame.iloc[indices]["hemolysis_risk"].median()),
        "risk_p75": float(np.percentile(frame.iloc[indices]["hemolysis_risk"], 75)),
        "risk_p95": float(np.percentile(frame.iloc[indices]["hemolysis_risk"], 95)),
        "risk_max": float(frame.iloc[indices]["hemolysis_risk"].max()),
        "panel_mean": float(panel.mean()), "breadth_mean": float((panel > .5).mean(axis=1).mean()),
        "mdr_mean": float((panel[:, mdr] > .5).mean(axis=1).mean()) if mdr else None,
        "pairwise_distance_mean": float(np.mean(distances)) if distances else 0.0,
        "risk_over_0_8": int((frame.iloc[indices]["hemolysis_risk"].to_numpy() > .8).sum()),
    }
    for j, genus in enumerate(genera):
        out[f"panel:{genus}:mean"] = float(panel[:, j].mean())
    if incumbent is not None:
        base = summarize(frame, incumbent, genera)
        out["activity_floor"] = base["activity_mean"] - .03
        out["distance_floor"] = base["pairwise_distance_mean"] - .02
        out["risk_delta_vs_incumbent"] = out["risk_mean"] - base["risk_mean"]
    return out
