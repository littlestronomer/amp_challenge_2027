"""Hash-bound inference metadata; historical validation claims are not imported."""
import hashlib
import json
from pathlib import Path


def load_config(directory: Path, stem: str) -> dict:
    head = directory / f"{stem}.pt"
    known = json.loads(Path(__file__).with_suffix(".json").read_text())
    expected = known.get(hashlib.sha256(head.read_bytes()).hexdigest()) if head.exists() else None
    explicit = directory / f"{stem}_config.json"
    if explicit.exists():
        actual = json.loads(explicit.read_text())
        if expected is not None:
            mismatches = [key for key, value in expected.items() if actual.get(key) != value]
            if mismatches:
                raise ValueError(f"Audited head metadata mismatch in {explicit}: {mismatches}")
        return actual
    if expected is not None:
        return dict(expected)
    return json.loads((directory / "config.json").read_text())
