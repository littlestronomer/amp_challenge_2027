from __future__ import annotations

import numpy as np


def baseline_indices(frame):
    values = np.flatnonzero(frame["incumbent_member"].to_numpy(dtype=bool))
    if len(values) != 100:
        raise ValueError(f"Expected 100 incumbent members, found {len(values)}")
    return values


def arrays(frame, genera):
    panel = frame[[f"panel:{g}" for g in genera]].to_numpy(dtype=float)
    from amp_challenge_2027.config import MDR_PANEL_GENERA
    mdr = [j for j, g in enumerate(genera) if g in MDR_PANEL_GENERA]
    return {
        "activity": frame["activity"].to_numpy(float), "risk": frame["hemolysis_risk"].to_numpy(float),
        "panel": panel, "breadth": (panel > .5).mean(axis=1),
        "mdr": (panel[:, mdr] > .5).mean(axis=1) if mdr else np.zeros(len(frame)),
    }


def linear_bounds(frame, genera, protocol=None):
    if protocol is None:
        protocol = {"activity_drop_max": .03, "panel_drop_max": .03,
                    "breadth_drop_max": .02, "mdr_drop_max": .02,
                    "distance_drop_max": .02}
    a = arrays(frame, genera); inc = baseline_indices(frame)
    panel_inc = a["panel"][inc].mean(axis=0)
    return a, inc, {
        "activity": float(a["activity"][inc].mean() - float(protocol["activity_drop_max"])),
        "panel": (panel_inc - float(protocol["panel_drop_max"])).tolist(),
        "breadth": float(a["breadth"][inc].mean() - float(protocol["breadth_drop_max"])),
        "mdr": float(a["mdr"][inc].mean() - float(protocol["mdr_drop_max"])),
        "risk_tail": int((a["risk"][inc] > float(protocol.get("risk_tail_threshold", .8))).sum()),
        "distance": float(_distance(frame, inc) - float(protocol["distance_drop_max"])),
    }


def _distance(frame, indices):
    from Levenshtein import ratio
    seqs = frame.iloc[indices]["sequence"].tolist()
    values = [1-ratio(a,b) for i,a in enumerate(seqs) for b in seqs[:i]]
    return float(np.mean(values)) if values else 0.0
