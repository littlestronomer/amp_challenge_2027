from __future__ import annotations

import json
from pathlib import Path


def validate_protocol(protocol: dict) -> dict:
    if protocol.get("kind") != "constrained_selectivity_v1":
        raise ValueError("Unsupported constrained-selectivity protocol kind")
    required = {"top_size", "activity_drop_max", "panel_drop_max", "breadth_drop_max",
                "mdr_drop_max", "distance_drop_max", "risk_tail_threshold", "risk_ceilings"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Protocol missing fields: {sorted(missing)}")
    if int(protocol["top_size"]) != 100:
        raise ValueError("This implementation requires top_size=100")
    for key in ("activity_drop_max", "panel_drop_max", "breadth_drop_max", "mdr_drop_max", "distance_drop_max"):
        value = float(protocol[key])
        if value < 0 or value > 1:
            raise ValueError(f"Invalid tolerance: {key}")
    if not 0 <= float(protocol["risk_tail_threshold"]) <= 1:
        raise ValueError("Invalid risk_tail_threshold")
    ceilings = [float(x) for x in protocol["risk_ceilings"]]
    if not ceilings or any(x < 0 or x > 1 for x in ceilings) or ceilings != sorted(set(ceilings)):
        raise ValueError("risk_ceilings must be sorted unique values in [0,1]")
    return protocol


def load_protocol(path: Path) -> dict:
    return validate_protocol(json.loads(path.read_text()))


def validate_pool_frame(frame, *, genera: list[str]) -> None:
    import numpy as np
    required = {"sequence", "library_index", "activity", "hemolysis_risk", "conformity", "precision"}
    required |= {f"panel:{g}" for g in genera}
    if not required <= set(frame.columns):
        raise ValueError(f"Pool is missing columns: {sorted(required - set(frame.columns))}")
    if frame["sequence"].duplicated().any() or frame["library_index"].duplicated().any():
        raise ValueError("Pool sequence/index identity is duplicated")
    numeric = [c for c in required if c not in {"sequence"}]
    for col in numeric:
        values = frame[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Pool contains non-finite values: {col}")


def validate_solution(indices, n: int, k: int = 100) -> None:
    import numpy as np
    values = np.asarray(indices, dtype=int)
    if values.shape != (k,) or len(set(values.tolist())) != k or (values < 0).any() or (values >= n).any():
        raise ValueError("Solution membership is not exactly a unique in-range top-k")
