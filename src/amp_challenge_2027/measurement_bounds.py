"""Conservative threshold implications; never interpret a censored bound as exact."""
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Bound:
    operator: str
    value: float

    def definitely_le(self, threshold: float) -> bool:
        return self.operator in ("=", "<", "<=") and self.value <= threshold

    def definitely_ge(self, threshold: float) -> bool:
        return self.operator in ("=", ">", ">=") and self.value >= threshold

    def definitely_gt(self, threshold: float) -> bool:
        return (self.operator == ">" and self.value >= threshold or
                self.operator in ("=", ">=") and self.value > threshold)


def parse_bound(text: str) -> Bound:
    normalized = text.strip().replace("≤", "<=").replace("≥", ">=")
    match = re.fullmatch(r"(<=|>=|<|>|=)?\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", normalized)
    if not match:
        raise ValueError(f"Unsupported measurement: {text!r}")
    value = float(match[2])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Concentration must be finite and positive")
    return Bound(match[1] or "=", value)


def activity_label(bound: Bound, success: float = 16, inactive: float = 32) -> str:
    if bound.definitely_le(success):
        return "active"
    if bound.definitely_gt(inactive):
        return "inactive"
    return "ambiguous"
